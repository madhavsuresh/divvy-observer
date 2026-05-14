"""Tile-level disabled-bike state + repair-time predictor.

This is a parallel system to ``tile_predictor`` (free + reserved bikes), kept
separate because the underlying generative process is different:

* The driver is the operator (Lyft / Divvy field crews), not the rider.
  Disability persists for hours/days until pickup or repair, so the relevant
  horizons are hours, not the 5-20 minutes of the free-bike flow model.

* Disabled is a near-absorbing state: a disabled bike doesn't transition back
  to free without operator intervention. The ``inventory_dp`` flow-rate
  machinery used by ``tile_predictor`` doesn't apply.

* Two distinct data sources contribute and are aggregated differently:

  - ``free_bike_status.is_disabled`` toggles on individual free-floating
    bikes. Sparse in practice (the operator likely flips bikes between
    states without the public feed always reflecting it) but lets us track
    per-bike dwell times directly.

  - ``station_status.num_bikes_disabled`` is a per-station count that's
    regularly non-zero. We see only count changes, not individual bikes,
    so dwell time becomes the aggregate "bike-hours of disability
    accumulated in this tile over the last 24h" (the integral of the
    disabled count over time).

Repair-time predictions are produced from observed repair events in the
historical (tile, hour-of-day) window, via a per-disabled-bike hazard rate
that's empirical-Bayes-shrunk toward the global rate at the same (hour,
dow). The resulting survival curve is reported at 1/6/24-hour horizons by
default — coarser than the free model on purpose.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable

import duckdb
import math
import pandas as pd

from . import tile, tile_predictor


HORIZONS_HOURS: tuple[float, ...] = (1.0, 6.0, 24.0)
EB_ALPHA = 50.0  # empirical-Bayes shrinkage prior weight; matches tile_predictor.
DEFAULT_LOOKBACK_DAYS = 28
# Disabled bikes can sit for days; only prune from the latest snapshot if we
# haven't seen the bike at all in this many hours (the operator has presumably
# picked it up).
MAX_DISABLED_BIKE_AGE_HOURS = 168  # 7 days


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _ids_placeholder(items: list[str]) -> str:
    return ",".join(["?"] * len(items))


# ---------------------------------------------------------------------------
# Per-tile free-floating disability hazard rate, feeds tile_predictor.
# ---------------------------------------------------------------------------

def free_disability_rate_priors(
    conn: duckdb.DuckDBPyConnection,
    tile_ids: Iterable[str],
    anchor_ts: datetime,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> pd.DataFrame:
    """Per-tile rate of free-floating bikes becoming disabled, per minute.

    Mirrors ``tile_predictor.tile_flow_priors``: counts fresh false→true
    transitions in ``free_bike_status`` at the matching (hour, dow), and
    EB-shrinks per-tile counts against the global mean using the same
    ``EB_ALPHA`` prior weight and the same minutes-per-(hour, dow) denominator.

    The returned rate is a *tile-level* events-per-minute (not per-bike) and
    should be added directly to ``depart_mean_per_min`` when composing the
    free-bike survival machinery — a free bike that runs out of battery leaves
    the rider-available pool the same way a ride does.

    Free-floating ``is_disabled`` transitions are sparsely reported by Divvy
    (the upstream feed doesn't always reflect operator toggles), so this rate
    is typically near zero. The integration becomes a near-no-op in practice
    but is correctly wired so that as coverage improves the model picks up
    the additional drain automatically.
    """
    tile_ids_list = list(dict.fromkeys(tile_ids))
    if not tile_ids_list:
        return pd.DataFrame(columns=["tile_id", "disability_rate_per_min"])
    placeholders = _ids_placeholder(tile_ids_list)
    start_ts = anchor_ts - pd.Timedelta(days=lookback_days)
    anchor_hour = int(anchor_ts.hour)
    anchor_dow = int(anchor_ts.weekday())

    per_tile = conn.execute(
        f"""
        WITH per_bike AS (
          SELECT
            bike_id, fetched_at, tile_id, is_disabled,
            LAG(is_disabled) OVER (PARTITION BY bike_id ORDER BY fetched_at) AS prev_disabled
          FROM free_bike_status
          WHERE tile_id IS NOT NULL
            AND fetched_at >= ?
        ),
        fresh_disabilities AS (
          SELECT tile_id, fetched_at
          FROM per_bike
          WHERE COALESCE(is_disabled, false) = true
            AND prev_disabled IS NOT NULL
            AND prev_disabled = false
            AND tile_id IN ({placeholders})
            AND EXTRACT(HOUR    FROM fetched_at) = ?
            AND (EXTRACT(ISODOW FROM fetched_at) - 1) = ?
        )
        SELECT tile_id, COUNT(*) AS n_events
        FROM fresh_disabilities
        GROUP BY tile_id
        """,
        [start_ts, *tile_ids_list, anchor_hour, anchor_dow],
    ).df()

    global_row = conn.execute(
        """
        WITH per_bike AS (
          SELECT
            bike_id, fetched_at, tile_id, is_disabled,
            LAG(is_disabled) OVER (PARTITION BY bike_id ORDER BY fetched_at) AS prev_disabled
          FROM free_bike_status
          WHERE tile_id IS NOT NULL
            AND fetched_at >= ?
        ),
        per_tile_counts AS (
          SELECT tile_id, COUNT(*) AS n_events
          FROM per_bike
          WHERE COALESCE(is_disabled, false) = true
            AND prev_disabled IS NOT NULL
            AND prev_disabled = false
            AND EXTRACT(HOUR    FROM fetched_at) = ?
            AND (EXTRACT(ISODOW FROM fetched_at) - 1) = ?
          GROUP BY tile_id
        )
        SELECT COALESCE(AVG(n_events), 0.0)::DOUBLE AS mean_events
        FROM per_tile_counts
        """,
        [start_ts, anchor_hour, anchor_dow],
    ).df()

    window_minutes = max(1.0, (lookback_days / 7.0) * 60.0)
    global_rate = float(global_row["mean_events"].iloc[0]) / window_minutes

    counts_by_tile = {
        str(r["tile_id"]): int(r["n_events"])
        for _, r in per_tile.iterrows()
    }

    rows = []
    for tid in tile_ids_list:
        n = counts_by_tile.get(tid, 0)
        rate = (EB_ALPHA * global_rate + n) / (EB_ALPHA + window_minutes)
        rows.append({"tile_id": tid, "disability_rate_per_min": rate})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Current per-tile disability state.
# ---------------------------------------------------------------------------

def _latest_station_status(
    conn: duckdb.DuckDBPyConnection, station_ids: list[str]
) -> pd.DataFrame:
    if not station_ids:
        return pd.DataFrame()
    placeholders = _ids_placeholder(station_ids)
    return conn.execute(
        f"""
        WITH latest AS (
          SELECT *, ROW_NUMBER() OVER (
            PARTITION BY station_id ORDER BY last_reported DESC
          ) AS rn
          FROM station_status
          WHERE station_id IN ({placeholders})
        )
        SELECT
          s.station_id, s.name, s.lat, s.lon, s.capacity,
          l.num_bikes_available, l.num_ebikes_available,
          l.num_bikes_disabled, l.num_docks_disabled,
          l.num_docks_available, l.last_reported
        FROM stations s
        JOIN latest l USING (station_id)
        WHERE l.rn = 1
        """,
        station_ids,
    ).df()


def _station_disability_transitions(
    conn: duckdb.DuckDBPyConnection,
    station_ids: list[str],
    *,
    now: datetime,
    hours: int = 24,
) -> pd.DataFrame:
    """Per-station deltas in num_bikes_disabled and num_docks_disabled.

    Each delta is signed: positive means a disability event (a bike was just
    flagged broken), negative means a repair event. We sum the positive and
    negative magnitudes separately because |delta| isn't quite the right
    aggregation: if a station's disabled count jumps 2→0, that's two repairs
    even though the net change is -2.
    """
    if not station_ids:
        return pd.DataFrame(
            columns=[
                "station_id",
                "disability_events",
                "repair_events",
                "dock_disability_events",
                "dock_repair_events",
                "bike_hours_disabled",
            ]
        )
    placeholders = _ids_placeholder(station_ids)
    return conn.execute(
        f"""
        WITH per_station AS (
          SELECT
            station_id, last_reported,
            num_bikes_disabled,
            num_docks_disabled,
            LAG(num_bikes_disabled) OVER (PARTITION BY station_id ORDER BY last_reported) AS prev_bikes_disabled,
            LAG(num_docks_disabled) OVER (PARTITION BY station_id ORDER BY last_reported) AS prev_docks_disabled,
            LAG(last_reported)      OVER (PARTITION BY station_id ORDER BY last_reported) AS prev_reported
          FROM station_status
          WHERE station_id IN ({placeholders})
            AND last_reported >= ? - INTERVAL ({hours}) HOUR
        )
        SELECT
          station_id,
          SUM(CASE
            WHEN prev_bikes_disabled IS NOT NULL AND num_bikes_disabled > prev_bikes_disabled
              THEN num_bikes_disabled - prev_bikes_disabled ELSE 0 END
          )::INTEGER AS disability_events,
          SUM(CASE
            WHEN prev_bikes_disabled IS NOT NULL AND num_bikes_disabled < prev_bikes_disabled
              THEN prev_bikes_disabled - num_bikes_disabled ELSE 0 END
          )::INTEGER AS repair_events,
          SUM(CASE
            WHEN prev_docks_disabled IS NOT NULL AND num_docks_disabled > prev_docks_disabled
              THEN num_docks_disabled - prev_docks_disabled ELSE 0 END
          )::INTEGER AS dock_disability_events,
          SUM(CASE
            WHEN prev_docks_disabled IS NOT NULL AND num_docks_disabled < prev_docks_disabled
              THEN prev_docks_disabled - num_docks_disabled ELSE 0 END
          )::INTEGER AS dock_repair_events,
          SUM(
            -- Trapezoidal-ish integral: average disabled count across each interval,
            -- multiplied by interval length in hours. Skips rows with no prior reading
            -- (the first row in the window).
            CASE
              WHEN prev_reported IS NULL THEN 0.0
              ELSE ((COALESCE(num_bikes_disabled, 0) + COALESCE(prev_bikes_disabled, 0)) / 2.0)
                 * EXTRACT(EPOCH FROM (last_reported - prev_reported)) / 3600.0
            END
          ) AS bike_hours_disabled
        FROM per_station
        GROUP BY station_id
        """,
        [*station_ids, now],
    ).df()


def _free_bike_disability_snapshot(
    conn: duckdb.DuckDBPyConnection,
    tile_ids: list[str],
    *,
    now: datetime,
) -> pd.DataFrame:
    """One row per currently-disabled free-floating bike in any of ``tile_ids``.

    Columns: tile_id, bike_id, name, lat, lon, disabled_since, last_event_at.
    ``disabled_since`` is the most recent false→true transition for that bike;
    if the bike has only ever been observed as disabled, it falls back to the
    first observed timestamp.
    """
    if not tile_ids:
        return pd.DataFrame()
    placeholders = _ids_placeholder(tile_ids)
    return conn.execute(
        f"""
        WITH per_bike AS (
          SELECT
            bike_id, fetched_at, name, lat, lon, tile_id, is_disabled,
            LAG(is_disabled) OVER (PARTITION BY bike_id ORDER BY fetched_at) AS prev_disabled,
            ROW_NUMBER() OVER (PARTITION BY bike_id ORDER BY fetched_at DESC) AS rn_desc
          FROM free_bike_status
          WHERE tile_id IS NOT NULL
            AND fetched_at >= ? - INTERVAL ({MAX_DISABLED_BIKE_AGE_HOURS}) HOUR
        ),
        latest_per_bike AS (
          SELECT bike_id, fetched_at AS last_event_at, name, lat, lon, tile_id
          FROM per_bike
          WHERE rn_desc = 1
            AND COALESCE(is_disabled, false) = true
            AND tile_id IN ({placeholders})
        ),
        disability_starts AS (
          -- Most recent false→true transition per bike (or first observation if
          -- the bike has only ever been observed disabled).
          SELECT bike_id, MAX(fetched_at) AS disabled_since
          FROM per_bike
          WHERE COALESCE(is_disabled, false) = true
            AND (prev_disabled IS NULL OR prev_disabled = false)
          GROUP BY bike_id
        )
        SELECT
          lp.tile_id, lp.bike_id, lp.name, lp.lat, lp.lon,
          lp.last_event_at, ds.disabled_since
        FROM latest_per_bike lp
        LEFT JOIN disability_starts ds USING (bike_id)
        """,
        [now, *tile_ids],
    ).df()


def _free_bike_disability_events(
    conn: duckdb.DuckDBPyConnection,
    tile_ids: list[str],
    *,
    now: datetime,
    hours: int = 24,
) -> pd.DataFrame:
    """Per-tile counts of free-floating disability / repair transitions.

    Counts only fresh transitions where we observed the opposite prior state —
    rows with no LAG (first appearance) don't contribute.
    """
    if not tile_ids:
        return pd.DataFrame(
            columns=["tile_id", "disability_events", "repair_events"]
        )
    placeholders = _ids_placeholder(tile_ids)
    return conn.execute(
        f"""
        WITH per_bike AS (
          SELECT
            bike_id, fetched_at, tile_id, is_disabled,
            LAG(is_disabled) OVER (PARTITION BY bike_id ORDER BY fetched_at) AS prev_disabled
          FROM free_bike_status
          WHERE tile_id IS NOT NULL
            AND fetched_at >= ? - INTERVAL ({hours}) HOUR
        )
        SELECT
          tile_id,
          SUM(CASE
            WHEN prev_disabled = false AND COALESCE(is_disabled, false) = true THEN 1 ELSE 0 END
          )::INTEGER AS disability_events,
          SUM(CASE
            WHEN prev_disabled = true AND COALESCE(is_disabled, false) = false THEN 1 ELSE 0 END
          )::INTEGER AS repair_events
        FROM per_bike
        WHERE tile_id IN ({placeholders})
        GROUP BY tile_id
        """,
        [now, *tile_ids],
    ).df()


def current_tile_disability_state(
    conn: duckdb.DuckDBPyConnection,
    tile_ids: Iterable[str],
) -> pd.DataFrame:
    """Return one row per tile_id with current disability state.

    Columns:
      tile_id, current_disabled_free_ebikes, current_disabled_docked_bikes,
      current_disabled_docks, disability_events_24h, repair_events_24h,
      dock_disability_events_24h, dock_repair_events_24h,
      bike_hours_disabled_24h, last_disability_event_at,
      disabled_free_bikes (list[dict]), disabled_stations (list[dict]),
      n_stations_in_tile.

    Note: ``free_bike_status.is_disabled`` is sparsely reported by Divvy, so
    free-floating disability volume will under-count true volume. Counts here
    are from our observed transitions, which is the lower bound. The docked
    side is more complete.
    """
    tile_ids_list = list(dict.fromkeys(tile_ids))
    if not tile_ids_list:
        return pd.DataFrame()
    now = _utc_now()

    bikes_df = _free_bike_disability_snapshot(conn, tile_ids_list, now=now)
    free_events_df = _free_bike_disability_events(conn, tile_ids_list, now=now, hours=24)

    station_map = tile_predictor.stations_in_tile_map(conn)
    all_station_ids = sorted({
        sid
        for tid in tile_ids_list
        for sid in station_map.get(tid, [])
    })
    station_status_df = _latest_station_status(conn, all_station_ids)
    station_status_by_id = {
        str(row["station_id"]): row.to_dict()
        for _, row in station_status_df.iterrows()
    }
    station_transitions_df = _station_disability_transitions(
        conn, all_station_ids, now=now, hours=24
    )
    station_transitions_by_id = {
        str(row["station_id"]): row.to_dict()
        for _, row in station_transitions_df.iterrows()
    }

    rows = []
    for tid in tile_ids_list:
        tile_bikes = bikes_df[bikes_df["tile_id"] == tid] if not bikes_df.empty else bikes_df
        disabled_free_bikes: list[dict[str, Any]] = []
        for _, b in tile_bikes.iterrows():
            disabled_since_val = b.get("disabled_since")
            if disabled_since_val is None or pd.isna(disabled_since_val):
                disabled_since = pd.Timestamp(b["last_event_at"]).to_pydatetime()
            else:
                disabled_since = pd.Timestamp(disabled_since_val).to_pydatetime()
            dwell_sec = max(0.0, (now - disabled_since).total_seconds())
            disabled_free_bikes.append({
                "bike_id": str(b["bike_id"]),
                "name": (None if pd.isna(b.get("name")) else str(b["name"])),
                "lat": float(b["lat"]),
                "lon": float(b["lon"]),
                "disabled_since": disabled_since,
                "last_event_at": pd.Timestamp(b["last_event_at"]).to_pydatetime(),
                "dwell_seconds_so_far": dwell_sec,
            })

        station_ids_for_tile = station_map.get(tid, [])
        current_disabled_docked_bikes = 0
        current_disabled_docks = 0
        dock_dis_events = 0
        dock_rep_events = 0
        bike_hours_disabled = 0.0
        disabled_stations: list[dict[str, Any]] = []
        for sid in station_ids_for_tile:
            row = station_status_by_id.get(sid)
            if not row:
                continue
            bikes_disabled = int(row.get("num_bikes_disabled") or 0)
            docks_disabled = int(row.get("num_docks_disabled") or 0)
            current_disabled_docked_bikes += bikes_disabled
            current_disabled_docks += docks_disabled
            trans = station_transitions_by_id.get(sid, {})
            station_dis_events = int(trans.get("disability_events") or 0)
            station_rep_events = int(trans.get("repair_events") or 0)
            station_dock_dis = int(trans.get("dock_disability_events") or 0)
            station_dock_rep = int(trans.get("dock_repair_events") or 0)
            station_bike_hours = float(trans.get("bike_hours_disabled") or 0.0)
            dock_dis_events += station_dock_dis
            dock_rep_events += station_dock_rep
            bike_hours_disabled += station_bike_hours
            disabled_stations.append({
                "station_id": sid,
                "name": (None if pd.isna(row.get("name")) else str(row.get("name"))),
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "capacity": (int(row["capacity"]) if not pd.isna(row.get("capacity")) else None),
                "num_bikes_disabled": bikes_disabled,
                "num_docks_disabled": docks_disabled,
                "num_bikes_available": (int(row["num_bikes_available"]) if not pd.isna(row.get("num_bikes_available")) else None),
                "num_ebikes_available": (int(row["num_ebikes_available"]) if not pd.isna(row.get("num_ebikes_available")) else None),
                "disability_events_24h": station_dis_events,
                "repair_events_24h": station_rep_events,
                "bike_hours_disabled_24h": station_bike_hours,
                "last_reported": pd.Timestamp(row["last_reported"]).to_pydatetime(),
            })

        # Free-floating transition counts at the tile level.
        free_ev_row = (
            free_events_df[free_events_df["tile_id"] == tid]
            if not free_events_df.empty
            else free_events_df
        )
        free_dis_events = int(free_ev_row["disability_events"].iloc[0]) if not free_ev_row.empty else 0
        free_rep_events = int(free_ev_row["repair_events"].iloc[0]) if not free_ev_row.empty else 0

        # Combined disability/repair event counts across both data sources.
        station_dis_events_total = sum(
            int((station_transitions_by_id.get(sid, {}).get("disability_events") or 0))
            for sid in station_ids_for_tile
        )
        station_rep_events_total = sum(
            int((station_transitions_by_id.get(sid, {}).get("repair_events") or 0))
            for sid in station_ids_for_tile
        )

        last_event_at: datetime | None = None
        for b in disabled_free_bikes:
            ts = b["disabled_since"]
            if last_event_at is None or ts > last_event_at:
                last_event_at = ts
        for st_row in disabled_stations:
            ts = st_row["last_reported"]
            if last_event_at is None or ts > last_event_at:
                last_event_at = ts

        rows.append({
            "tile_id": tid,
            "current_disabled_free_ebikes": len(disabled_free_bikes),
            "current_disabled_docked_bikes": current_disabled_docked_bikes,
            "current_disabled_docks": current_disabled_docks,
            "disability_events_24h": free_dis_events + station_dis_events_total,
            "repair_events_24h": free_rep_events + station_rep_events_total,
            "free_disability_events_24h": free_dis_events,
            "free_repair_events_24h": free_rep_events,
            "dock_bike_disability_events_24h": station_dis_events_total,
            "dock_bike_repair_events_24h": station_rep_events_total,
            "dock_disability_events_24h": dock_dis_events,
            "dock_repair_events_24h": dock_rep_events,
            "bike_hours_disabled_24h": bike_hours_disabled,
            "last_disability_event_at": last_event_at,
            "disabled_free_bikes": disabled_free_bikes,
            "disabled_stations": disabled_stations,
            "n_stations_in_tile": len(station_ids_for_tile),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Dwell-time summaries.
# ---------------------------------------------------------------------------

def dwell_time_for_disabled(
    conn: duckdb.DuckDBPyConnection,
    tile_ids: Iterable[str],
) -> pd.DataFrame:
    """Return per-tile dwell-time summaries for currently-disabled bikes.

    Free-floating dwell is derived per-bike from observed disability toggles:
    we report median and p90 of (now - disabled_since) in minutes across the
    currently-disabled free bikes in the tile.

    Docked dwell is the accumulated bike-hours of disability in the last 24h
    (integral of num_bikes_disabled over time, summed across stations in the
    tile). Divvy doesn't expose individual bike_ids for docked bikes, so this
    is the only meaningful aggregation.

    Columns:
      tile_id, n_disabled_free, median_free_dwell_minutes, p90_free_dwell_minutes,
      bike_hours_disabled_24h.
    """
    state_df = current_tile_disability_state(conn, tile_ids)
    if state_df.empty:
        return pd.DataFrame(
            columns=[
                "tile_id",
                "n_disabled_free",
                "median_free_dwell_minutes",
                "p90_free_dwell_minutes",
                "bike_hours_disabled_24h",
            ]
        )

    rows = []
    for _, r in state_df.iterrows():
        dwells = [
            b["dwell_seconds_so_far"] / 60.0 for b in r["disabled_free_bikes"]
        ]
        median_dwell = float(pd.Series(dwells).median()) if dwells else 0.0
        p90_dwell = float(pd.Series(dwells).quantile(0.9)) if dwells else 0.0
        rows.append({
            "tile_id": r["tile_id"],
            "n_disabled_free": int(r["current_disabled_free_ebikes"]),
            "median_free_dwell_minutes": median_dwell,
            "p90_free_dwell_minutes": p90_dwell,
            "bike_hours_disabled_24h": float(r["bike_hours_disabled_24h"]),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Empirical-Bayes repair-rate prior at the anchor's (hour, dow).
# ---------------------------------------------------------------------------

def _repair_rate_history(
    conn: duckdb.DuckDBPyConnection,
    tile_ids: list[str],
    anchor_ts: datetime,
    lookback_days: int,
) -> tuple[pd.DataFrame, float, float]:
    """Per-(tile) repair-event rate per disabled-bike-hour at the anchor's hour/dow.

    Returns (per_tile_df, global_rate_per_hour, n_tiles_observed). The per-tile
    rate is computed by summing observed repair events (free + dock) and
    dividing by accumulated disabled-bike-hours over the lookback period at
    the matching (hour, dow). Tiles with no observed disability events at the
    matching hour return 0 for both numerator and denominator and rely on the
    EB shrinkage toward the global rate.
    """
    placeholders = _ids_placeholder(tile_ids)
    start_ts = anchor_ts - pd.Timedelta(days=lookback_days)
    anchor_hour = int(anchor_ts.hour)
    anchor_dow = int(anchor_ts.weekday())

    station_map = tile_predictor.stations_in_tile_map(conn)
    station_to_tile: dict[str, str] = {}
    for tid, sids in station_map.items():
        for sid in sids:
            station_to_tile[str(sid)] = tid

    # Per-tile dock-side repair events + accumulated disabled-bike-hours.
    if station_to_tile:
        station_ids_in_scope = [
            sid for sid, tid in station_to_tile.items() if tid in tile_ids
        ]
    else:
        station_ids_in_scope = []

    if station_ids_in_scope:
        sids_ph = _ids_placeholder(station_ids_in_scope)
        dock_df = conn.execute(
            f"""
            WITH per_station AS (
              SELECT
                station_id, last_reported,
                num_bikes_disabled,
                LAG(num_bikes_disabled) OVER (PARTITION BY station_id ORDER BY last_reported) AS prev_bikes_disabled,
                LAG(last_reported)      OVER (PARTITION BY station_id ORDER BY last_reported) AS prev_reported
              FROM station_status
              WHERE station_id IN ({sids_ph})
                AND last_reported >= ?
            )
            SELECT
              station_id,
              SUM(CASE
                WHEN prev_bikes_disabled IS NOT NULL
                 AND num_bikes_disabled < prev_bikes_disabled
                 AND EXTRACT(HOUR FROM last_reported) = ?
                 AND (EXTRACT(ISODOW FROM last_reported) - 1) = ?
                  THEN prev_bikes_disabled - num_bikes_disabled
                ELSE 0 END
              )::DOUBLE AS repair_events,
              SUM(CASE
                WHEN prev_reported IS NULL THEN 0.0
                WHEN EXTRACT(HOUR FROM last_reported) = ?
                 AND (EXTRACT(ISODOW FROM last_reported) - 1) = ?
                  THEN ((COALESCE(num_bikes_disabled,0) + COALESCE(prev_bikes_disabled,0)) / 2.0)
                     * EXTRACT(EPOCH FROM (last_reported - prev_reported)) / 3600.0
                ELSE 0.0 END
              ) AS disabled_bike_hours
            FROM per_station
            GROUP BY station_id
            """,
            [*station_ids_in_scope, start_ts, anchor_hour, anchor_dow, anchor_hour, anchor_dow],
        ).df()
    else:
        dock_df = pd.DataFrame(columns=["station_id", "repair_events", "disabled_bike_hours"])

    # Map per-station to per-tile.
    if not dock_df.empty:
        dock_df["tile_id"] = dock_df["station_id"].astype(str).map(station_to_tile)
        dock_per_tile = dock_df.groupby("tile_id", as_index=False).agg(
            dock_repair_events=("repair_events", "sum"),
            dock_disabled_bike_hours=("disabled_bike_hours", "sum"),
        )
    else:
        dock_per_tile = pd.DataFrame(columns=["tile_id", "dock_repair_events", "dock_disabled_bike_hours"])

    # Per-tile free-floating repair events + disabled-bike-hours over time.
    free_df = conn.execute(
        f"""
        WITH per_bike AS (
          SELECT
            bike_id, fetched_at, tile_id, is_disabled,
            LAG(is_disabled) OVER (PARTITION BY bike_id ORDER BY fetched_at) AS prev_disabled,
            LAG(fetched_at)  OVER (PARTITION BY bike_id ORDER BY fetched_at) AS prev_fetched_at
          FROM free_bike_status
          WHERE tile_id IS NOT NULL
            AND fetched_at >= ?
        )
        SELECT
          tile_id,
          SUM(CASE
            WHEN prev_disabled = true
             AND COALESCE(is_disabled,false) = false
             AND EXTRACT(HOUR FROM fetched_at) = ?
             AND (EXTRACT(ISODOW FROM fetched_at) - 1) = ?
              THEN 1 ELSE 0 END
          )::DOUBLE AS free_repair_events,
          SUM(CASE
            WHEN prev_fetched_at IS NULL OR COALESCE(prev_disabled, false) = false THEN 0.0
            WHEN EXTRACT(HOUR FROM fetched_at) = ?
             AND (EXTRACT(ISODOW FROM fetched_at) - 1) = ?
              THEN EXTRACT(EPOCH FROM (fetched_at - prev_fetched_at)) / 3600.0
            ELSE 0.0 END
          ) AS free_disabled_bike_hours
        FROM per_bike
        WHERE tile_id IN ({placeholders})
        GROUP BY tile_id
        """,
        [start_ts, anchor_hour, anchor_dow, anchor_hour, anchor_dow, *tile_ids],
    ).df()

    merged = pd.merge(dock_per_tile, free_df, on="tile_id", how="outer").fillna(0.0)
    if merged.empty:
        merged = pd.DataFrame(columns=[
            "tile_id", "dock_repair_events", "dock_disabled_bike_hours",
            "free_repair_events", "free_disabled_bike_hours",
        ])

    merged["repair_events"] = merged.get("dock_repair_events", 0.0) + merged.get("free_repair_events", 0.0)
    merged["disabled_bike_hours"] = (
        merged.get("dock_disabled_bike_hours", 0.0) + merged.get("free_disabled_bike_hours", 0.0)
    )

    global_total_events = float(merged["repair_events"].sum())
    global_total_hours = float(merged["disabled_bike_hours"].sum())
    global_rate_per_hour = (global_total_events / global_total_hours) if global_total_hours > 0 else 0.0
    n_tiles = int(len(merged))

    return merged[["tile_id", "repair_events", "disabled_bike_hours"]], global_rate_per_hour, n_tiles


def repair_rate_priors(
    conn: duckdb.DuckDBPyConnection,
    tile_ids: Iterable[str],
    anchor_ts: datetime,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> pd.DataFrame:
    """Per-tile repair hazard rate (events per disabled-bike-hour), EB-shrunk.

    Columns: tile_id, repair_rate_per_hour, global_rate_per_hour,
             observed_events, observed_disabled_bike_hours.
    """
    tile_ids_list = list(dict.fromkeys(tile_ids))
    if not tile_ids_list:
        return pd.DataFrame(
            columns=[
                "tile_id",
                "repair_rate_per_hour",
                "global_rate_per_hour",
                "observed_events",
                "observed_disabled_bike_hours",
            ]
        )

    per_tile, global_rate, _ = _repair_rate_history(
        conn, tile_ids_list, anchor_ts, lookback_days
    )
    counts_by_tile = {
        str(r["tile_id"]): (float(r["repair_events"]), float(r["disabled_bike_hours"]))
        for _, r in per_tile.iterrows()
    }

    rows = []
    for tid in tile_ids_list:
        events, hours = counts_by_tile.get(tid, (0.0, 0.0))
        # EB shrinkage: combined = (alpha * global_rate + events) / (alpha + hours).
        # alpha plays the role of "synthetic disabled-bike-hours" for the prior.
        rate = (EB_ALPHA * global_rate + events) / (EB_ALPHA + hours)
        rows.append({
            "tile_id": tid,
            "repair_rate_per_hour": rate,
            "global_rate_per_hour": global_rate,
            "observed_events": events,
            "observed_disabled_bike_hours": hours,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Repair-time prediction (survival curve from constant-hazard model).
# ---------------------------------------------------------------------------

def predict_repair_time(
    conn: duckdb.DuckDBPyConnection,
    tile_id: str,
    current_disabled_count: int,
    *,
    anchor_ts: datetime | None = None,
    horizons_hours: tuple[float, ...] = HORIZONS_HOURS,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> dict[float, dict[str, float]]:
    """Return per-horizon repair-event probabilities for one tile.

    Models repair as Poisson(rate * n_disabled) per unit time, where ``rate``
    is the EB-shrunk per-disabled-bike-hour hazard at the anchor's hour-of-day
    and ``n_disabled = current_disabled_count``. The closed-form survival is
    ``P(no repair in t hours) = exp(-rate * n_disabled * t)``.

    A purely non-parametric Kaplan-Meier estimate from observed durations is
    feasible for free-floating bikes (we see per-bike disabled_since), but for
    docked bikes we only see aggregate counts, so we can't recover individual
    durations. The constant-hazard model gives one consistent estimator across
    both sources; the trade-off is that it can't capture early-vs-late
    differences in repair propensity (e.g. truck routes that prefer
    just-broken bikes).

    Returns ``{horizon_hours: {p_any_repair, p_all_repaired, expected_repairs,
    rate_per_hour}}``.
    """
    anchor_ts = anchor_ts or _utc_now()
    n = max(0, int(current_disabled_count))
    priors = repair_rate_priors(conn, [tile_id], anchor_ts, lookback_days)
    if priors.empty:
        return {
            float(h): {
                "p_any_repair": 0.0,
                "p_all_repaired": 1.0 if n == 0 else 0.0,
                "expected_repairs": 0.0,
                "rate_per_hour": 0.0,
            }
            for h in horizons_hours
        }

    rate = float(priors["repair_rate_per_hour"].iloc[0])
    out: dict[float, dict[str, float]] = {}
    for h in horizons_hours:
        h = float(h)
        # Per-tile event-process rate scales linearly with the number of
        # at-risk disabled bikes.
        process_rate = rate * max(0, n) * h
        p_no_event = math.exp(-process_rate)
        p_any_event = 1.0 - p_no_event
        # P(all n repaired in horizon) — each bike independently has a constant
        # hazard ``rate`` per hour, so P(bike still disabled after h) = exp(-rate*h)
        # and P(bike repaired) = 1 - exp(-rate*h).
        p_one_repaired = 1.0 - math.exp(-rate * h)
        p_all_repaired = p_one_repaired ** max(0, n) if n > 0 else 1.0
        out[h] = {
            "p_any_repair": p_any_event,
            "p_all_repaired": p_all_repaired,
            "expected_repairs": process_rate,
            "rate_per_hour": rate,
        }
    return out


# ---------------------------------------------------------------------------
# Combined scoring entry point.
# ---------------------------------------------------------------------------

def score_tiles_disability(
    conn: duckdb.DuckDBPyConnection,
    tile_ids: Iterable[str],
    *,
    anchor_ts: datetime | None = None,
    horizons_hours: tuple[float, ...] = HORIZONS_HOURS,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (per-tile state with dwell-time fields, per-(tile, horizon) repair scores)."""
    anchor_ts = anchor_ts or _utc_now()
    state_df = current_tile_disability_state(conn, tile_ids)
    if state_df.empty:
        return state_df, pd.DataFrame()

    tile_ids_list = state_df["tile_id"].tolist()
    dwell_df = dwell_time_for_disabled(conn, tile_ids_list)
    dwell_by_tile = {str(r["tile_id"]): r.to_dict() for _, r in dwell_df.iterrows()}
    priors = repair_rate_priors(conn, tile_ids_list, anchor_ts, lookback_days)
    rate_by_tile = {
        str(r["tile_id"]): float(r["repair_rate_per_hour"]) for _, r in priors.iterrows()
    }

    state_df = state_df.copy()
    state_df["median_free_dwell_minutes"] = state_df["tile_id"].map(
        lambda t: float(dwell_by_tile.get(str(t), {}).get("median_free_dwell_minutes", 0.0))
    )
    state_df["p90_free_dwell_minutes"] = state_df["tile_id"].map(
        lambda t: float(dwell_by_tile.get(str(t), {}).get("p90_free_dwell_minutes", 0.0))
    )

    score_rows: list[dict[str, Any]] = []
    for _, row in state_df.iterrows():
        tid = row["tile_id"]
        n = int(row["current_disabled_free_ebikes"]) + int(row["current_disabled_docked_bikes"])
        rate = rate_by_tile.get(tid, 0.0)
        for h in horizons_hours:
            h = float(h)
            process_rate = rate * max(0, n) * h
            p_no_event = math.exp(-process_rate)
            p_any_event = 1.0 - p_no_event
            p_one_repaired = 1.0 - math.exp(-rate * h)
            p_all_repaired = p_one_repaired ** max(0, n) if n > 0 else 1.0
            score_rows.append({
                "tile_id": tid,
                "horizon_hours": h,
                "n_disabled": n,
                "rate_per_hour": rate,
                "p_any_repair": p_any_event,
                "p_all_repaired": p_all_repaired,
                "expected_repairs": process_rate,
            })
    return state_df, pd.DataFrame(score_rows)
