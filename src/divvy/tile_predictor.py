"""Tile-level state + free-ebike persistence predictor.

Three jobs:

1. Snapshot the current state of each tile (free-bike count, churn, list of
   bikes presently in the tile, list of stations whose lat/lon falls in the
   tile and their latest GBFS status).

2. Estimate the tile's free-ebike departure and arrival intensities over the
   next ``horizon`` minutes by EB-shrinking observed (tile, hour, dow) flow
   rates against (hour, dow) globals.

3. For each tile and each horizon, run ``tile_dp.tile_rollout`` for the
   free-ebike side, aggregate per-station horizon predictions from
   ``live_station_predictions`` for the dock side, and compose the combined
   "any ebike anywhere in this tile" probability.

Dock-side predictions are *read*, not recomputed: the collector's existing
scoring loop already populates ``live_station_predictions`` for every active
model + horizon, and the tile aggregator just JOINs against it. Independence
is assumed between stations (and between docked / free) when forming
``p_any_ebike``; this is the standard "at least one" rollup and produces an
upper bound when correlations are high.
"""
from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from typing import Any, Iterable

import duckdb
import pandas as pd

from . import live_cache, model_selection, predictor, tile, tile_dp


HORIZONS: tuple[int, ...] = predictor.HORIZONS
EB_ALPHA = 50.0  # empirical-Bayes shrinkage prior weight; matches dg_nissm_features.
DEFAULT_LOOKBACK_DAYS = 28
MAX_BIKE_AGE_HOURS = 72  # ignore bikes whose latest event is older than this
DOCK_PRED_MAX_AGE_MIN = 30

_STATIONS_CACHE_TTL_SEC = 6 * 3600
_stations_cache: dict[str, tuple[float, dict[str, list[str]]]] = {}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _ids_placeholder(items: list[str]) -> str:
    return ",".join(["?"] * len(items))


# ---------------------------------------------------------------------------
# Static reverse-index: station → tile.
# Station positions almost never change; rebuild every six hours.
# ---------------------------------------------------------------------------

def stations_in_tile_map(conn: duckdb.DuckDBPyConnection) -> dict[str, list[str]]:
    cached = _stations_cache.get("v")
    if cached is not None and (time.monotonic() - cached[0]) < _STATIONS_CACHE_TTL_SEC:
        return cached[1]
    rows = conn.execute(
        "SELECT station_id, lat, lon FROM stations WHERE lat IS NOT NULL AND lon IS NOT NULL"
    ).fetchall()
    mapping: dict[str, list[str]] = {}
    for station_id, lat, lon in rows:
        tid = tile.tile_id_for(lat, lon)
        if tid is None:
            continue
        mapping.setdefault(tid, []).append(str(station_id))
    _stations_cache["v"] = (time.monotonic(), mapping)
    return mapping


# ---------------------------------------------------------------------------
# Current state per tile (free bikes + stations + churn).
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
          l.num_docks_available, l.last_reported
        FROM stations s
        JOIN latest l USING (station_id)
        WHERE l.rn = 1
        """,
        station_ids,
    ).df()


def current_tile_state(
    conn: duckdb.DuckDBPyConnection,
    tile_ids: Iterable[str],
    *,
    live_bike_ids: set[str] | None = None,
) -> pd.DataFrame:
    """Return one row per tile_id with current free-bike state and station list.

    Columns:
      tile_id, current_free_ebikes, churn_rate_5m, churn_rate_30m,
      last_change_at, bikes (list[dict]), stations (list[dict]),
      current_docked_ebikes, n_stations_in_tile.

    Important: ``free_bike_status`` rows are written only when a bike's state
    changes, and there is no removal event when a bike leaves the GBFS feed
    (operator pickup, docking, etc). The "latest event" query alone therefore
    over-counts retired bikes. Pass ``live_bike_ids`` (the set of bike_ids in
    the most recent GBFS poll) to filter to actually-present bikes. When None,
    we fall back to the ``MAX_BIKE_AGE_HOURS`` cutoff, which is a coarse
    approximation: bikes that have sat motionless longer than that get pruned
    along with truly retired ones.
    """
    tile_ids_list = list(dict.fromkeys(tile_ids))
    if not tile_ids_list:
        return pd.DataFrame()
    now = _utc_now()
    placeholders = _ids_placeholder(tile_ids_list)

    bikes_df = conn.execute(
        f"""
        WITH per_bike AS (
          SELECT
            bike_id, fetched_at, name, lat, lon, tile_id, is_reserved, is_disabled,
            LAG(tile_id) OVER (PARTITION BY bike_id ORDER BY fetched_at) AS prev_tile,
            ROW_NUMBER() OVER (PARTITION BY bike_id ORDER BY fetched_at DESC) AS rn_desc
          FROM free_bike_status
          WHERE tile_id IS NOT NULL
            AND fetched_at >= ? - INTERVAL ({MAX_BIKE_AGE_HOURS}) HOUR
        ),
        latest_per_bike AS (
          SELECT * FROM per_bike WHERE rn_desc = 1
        ),
        entries_per_bike AS (
          SELECT bike_id, tile_id, MAX(fetched_at) AS entered_at
          FROM per_bike
          WHERE prev_tile IS NULL OR prev_tile != tile_id
          GROUP BY bike_id, tile_id
        )
        SELECT
          lp.tile_id, lp.bike_id, lp.name, lp.lat, lp.lon,
          lp.is_reserved,
          lp.fetched_at AS last_event_at, ep.entered_at
        FROM latest_per_bike lp
        JOIN entries_per_bike ep
          ON ep.bike_id = lp.bike_id AND ep.tile_id = lp.tile_id
        WHERE lp.tile_id IN ({placeholders})
          AND COALESCE(lp.is_disabled, false) = false
        """,
        [now, *tile_ids_list],
    ).df()

    if live_bike_ids is not None and not bikes_df.empty:
        # Apply live filter only to NON-reserved bikes. Reserved bikes drop out of
        # Divvy's GBFS free_bike_status feed for the duration of the hold (~5 min),
        # so a bike whose latest stored state is is_reserved=true won't appear in
        # live_bike_ids — but it's still legitimately "reserved here right now."
        # We additionally require the reserved row to be fresh (≤10 min) so we
        # don't surface bikes that were briefly reserved hours ago and have since
        # vanished without a follow-up row.
        is_reserved_mask = bikes_df["is_reserved"].fillna(False).astype(bool)
        last_event_recent_mask = bikes_df["last_event_at"] >= pd.Timestamp(now) - pd.Timedelta(minutes=10)
        keep_free = (~is_reserved_mask) & bikes_df["bike_id"].astype(str).isin(live_bike_ids)
        keep_reserved = is_reserved_mask & last_event_recent_mask
        bikes_df = bikes_df[keep_free | keep_reserved]

    # Count fresh reservation events (false → true transitions) per tile in two windows.
    reservation_events_df = conn.execute(
        f"""
        WITH per_bike AS (
          SELECT
            bike_id, fetched_at, tile_id, is_reserved,
            LAG(is_reserved) OVER (PARTITION BY bike_id ORDER BY fetched_at) AS prev_reserved
          FROM free_bike_status
          WHERE tile_id IS NOT NULL
            AND fetched_at >= ? - INTERVAL 30 MINUTE
        ),
        fresh_reservations AS (
          SELECT tile_id, fetched_at
          FROM per_bike
          WHERE COALESCE(is_reserved, false) = true
            AND prev_reserved IS NOT NULL
            AND prev_reserved = false
            AND tile_id IN ({placeholders})
        )
        SELECT
          tile_id,
          COUNT(*) FILTER (WHERE fetched_at >= ? - INTERVAL 5 MINUTE) AS reservation_events_5m,
          COUNT(*) AS reservation_events_30m
        FROM fresh_reservations
        GROUP BY tile_id
        """,
        [now, *tile_ids_list, now],
    ).df()

    churn_df = conn.execute(
        f"""
        SELECT
          tile_id,
          COUNT(*) FILTER (WHERE fetched_at >= ? - INTERVAL 5 MINUTE) AS churn_5m,
          COUNT(*) FILTER (WHERE fetched_at >= ? - INTERVAL 30 MINUTE) AS churn_30m,
          MAX(fetched_at) AS last_change_at
        FROM free_bike_status
        WHERE tile_id IN ({placeholders})
          AND fetched_at >= ? - INTERVAL 30 MINUTE
        GROUP BY tile_id
        """,
        [now, now, *tile_ids_list, now],
    ).df()

    station_map = stations_in_tile_map(conn)
    all_station_ids = sorted({
        sid
        for tid in tile_ids_list
        for sid in station_map.get(tid, [])
    })
    station_status_df = _latest_station_status(conn, all_station_ids)
    by_station = {
        str(row["station_id"]): row.to_dict() for _, row in station_status_df.iterrows()
    }

    rows = []
    for tid in tile_ids_list:
        tile_bikes = bikes_df[bikes_df["tile_id"] == tid] if not bikes_df.empty else bikes_df
        bikes_list: list[dict[str, Any]] = []
        reserved_bikes_list: list[dict[str, Any]] = []
        for _, b in tile_bikes.iterrows():
            entered_at = pd.Timestamp(b["entered_at"]).to_pydatetime()
            dwell_sec = max(0.0, (now - entered_at).total_seconds())
            bike_dict = {
                "bike_id": str(b["bike_id"]),
                "name": (None if pd.isna(b.get("name")) else str(b["name"])),
                "lat": float(b["lat"]),
                "lon": float(b["lon"]),
                "entered_at": entered_at,
                "last_event_at": pd.Timestamp(b["last_event_at"]).to_pydatetime(),
                "dwell_seconds_so_far": dwell_sec,
            }
            if bool(b.get("is_reserved")):
                reserved_bikes_list.append(bike_dict)
            else:
                bikes_list.append(bike_dict)

        station_ids_for_tile = station_map.get(tid, [])
        stations_list: list[dict[str, Any]] = []
        current_docked_ebikes = 0
        for sid in station_ids_for_tile:
            row = by_station.get(sid)
            if not row:
                continue
            ebikes = int(row.get("num_ebikes_available") or 0)
            current_docked_ebikes += ebikes
            stations_list.append({
                "station_id": sid,
                "name": (None if pd.isna(row.get("name")) else str(row.get("name"))),
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "capacity": (int(row["capacity"]) if not pd.isna(row.get("capacity")) else None),
                "num_ebikes_available": ebikes,
                "num_bikes_available": (int(row["num_bikes_available"]) if not pd.isna(row.get("num_bikes_available")) else None),
                "num_docks_available": (int(row["num_docks_available"]) if not pd.isna(row.get("num_docks_available")) else None),
                "last_reported": pd.Timestamp(row["last_reported"]).to_pydatetime(),
            })

        churn_row = churn_df[churn_df["tile_id"] == tid] if not churn_df.empty else churn_df
        churn_5m = int(churn_row["churn_5m"].iloc[0]) if not churn_row.empty else 0
        churn_30m = int(churn_row["churn_30m"].iloc[0]) if not churn_row.empty else 0
        last_change = (
            pd.Timestamp(churn_row["last_change_at"].iloc[0]).to_pydatetime()
            if not churn_row.empty and not pd.isna(churn_row["last_change_at"].iloc[0])
            else None
        )

        res_row = (
            reservation_events_df[reservation_events_df["tile_id"] == tid]
            if not reservation_events_df.empty
            else reservation_events_df
        )
        reservation_events_5m = int(res_row["reservation_events_5m"].iloc[0]) if not res_row.empty else 0
        reservation_events_30m = int(res_row["reservation_events_30m"].iloc[0]) if not res_row.empty else 0

        rows.append({
            "tile_id": tid,
            "current_free_ebikes": len(bikes_list),
            "current_reserved_free_ebikes": len(reserved_bikes_list),
            "churn_rate_5m": churn_5m,
            "churn_rate_30m": churn_30m,
            "reservation_events_5m": reservation_events_5m,
            "reservation_events_30m": reservation_events_30m,
            "last_change_at": last_change,
            "bikes": bikes_list,
            "reserved_bikes": reserved_bikes_list,
            "stations": stations_list,
            "current_docked_ebikes": current_docked_ebikes,
            "n_stations_in_tile": len(stations_list),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Empirical-Bayes flow priors per tile, at the anchor's (hour, dow).
# ---------------------------------------------------------------------------

def _flow_history(
    conn: duckdb.DuckDBPyConnection,
    tile_ids: list[str],
    anchor_ts: datetime,
    lookback_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Return (per-tile counts at anchor's hour/dow, global counts, total minutes-per-hour-dow)."""
    placeholders = _ids_placeholder(tile_ids)
    start_ts = anchor_ts - pd.Timedelta(days=lookback_days)
    anchor_hour = int(anchor_ts.hour)
    anchor_dow = int(anchor_ts.weekday())

    # DuckDB EXTRACT(DOW) returns 0=Sunday..6=Saturday; Python weekday() is 0=Monday.
    # We don't need them to match — we just need consistency between filter and counts.
    # Using EXTRACT(ISODOW)-1 lines up with Python's weekday() (0=Mon).
    per_tile = conn.execute(
        f"""
        WITH per_bike AS (
          SELECT
            bike_id, fetched_at, tile_id,
            LAG(tile_id) OVER (PARTITION BY bike_id ORDER BY fetched_at) AS prev_tile,
            LEAD(tile_id) OVER (PARTITION BY bike_id ORDER BY fetched_at) AS next_tile,
            LEAD(fetched_at) OVER (PARTITION BY bike_id ORDER BY fetched_at) AS next_fetched_at
          FROM free_bike_status
          WHERE tile_id IS NOT NULL
            AND fetched_at >= ?
        ),
        arrivals AS (
          SELECT tile_id AS event_tile, fetched_at AS event_at
          FROM per_bike
          WHERE (prev_tile IS NULL OR prev_tile != tile_id)
            AND tile_id IN ({placeholders})
        ),
        departures AS (
          SELECT tile_id AS event_tile, next_fetched_at AS event_at
          FROM per_bike
          WHERE next_tile IS NOT NULL
            AND next_tile != tile_id
            AND tile_id IN ({placeholders})
        )
        SELECT
          event_tile AS tile_id,
          SUM(CASE WHEN kind = 'arrival'   THEN 1 ELSE 0 END) AS n_arrivals,
          SUM(CASE WHEN kind = 'departure' THEN 1 ELSE 0 END) AS n_departures
        FROM (
          SELECT event_tile, event_at, 'arrival'   AS kind FROM arrivals
          UNION ALL
          SELECT event_tile, event_at, 'departure' AS kind FROM departures
        )
        WHERE event_at IS NOT NULL
          AND EXTRACT(HOUR    FROM event_at) = ?
          AND (EXTRACT(ISODOW FROM event_at) - 1) = ?
        GROUP BY event_tile
        """,
        [start_ts, *tile_ids, *tile_ids, anchor_hour, anchor_dow],
    ).df()

    global_row = conn.execute(
        """
        WITH per_bike AS (
          SELECT
            bike_id, fetched_at, tile_id,
            LAG(tile_id) OVER (PARTITION BY bike_id ORDER BY fetched_at) AS prev_tile,
            LEAD(tile_id) OVER (PARTITION BY bike_id ORDER BY fetched_at) AS next_tile,
            LEAD(fetched_at) OVER (PARTITION BY bike_id ORDER BY fetched_at) AS next_fetched_at
          FROM free_bike_status
          WHERE tile_id IS NOT NULL
            AND fetched_at >= ?
        ),
        all_events AS (
          SELECT tile_id, fetched_at AS event_at, 'arrival' AS kind FROM per_bike
            WHERE prev_tile IS NULL OR prev_tile != tile_id
          UNION ALL
          SELECT tile_id, next_fetched_at AS event_at, 'departure' AS kind FROM per_bike
            WHERE next_tile IS NOT NULL AND next_tile != tile_id
        ),
        per_tile_counts AS (
          SELECT
            tile_id,
            SUM(CASE WHEN kind = 'arrival'   THEN 1 ELSE 0 END) AS n_a,
            SUM(CASE WHEN kind = 'departure' THEN 1 ELSE 0 END) AS n_d,
            COUNT(*) AS n_total
          FROM all_events
          WHERE event_at IS NOT NULL
            AND EXTRACT(HOUR    FROM event_at) = ?
            AND (EXTRACT(ISODOW FROM event_at) - 1) = ?
          GROUP BY tile_id
        )
        SELECT
          COALESCE(AVG(n_a), 0.0)::DOUBLE AS mean_arrivals,
          COALESCE(AVG(n_d), 0.0)::DOUBLE AS mean_departures,
          COALESCE(COUNT(*), 0)::BIGINT  AS n_tiles
        FROM per_tile_counts
        """,
        [start_ts, anchor_hour, anchor_dow],
    ).df()

    # Window of relevant minutes for this (hour, dow): roughly lookback_days/7 weekly
    # instances × 60 minutes. We use this denominator to convert counts into
    # rate-per-minute.
    window_minutes = (lookback_days / 7.0) * 60.0
    return per_tile, global_row, max(1.0, window_minutes)


def tile_flow_priors(
    conn: duckdb.DuckDBPyConnection,
    tile_ids: Iterable[str],
    anchor_ts: datetime,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> pd.DataFrame:
    """Return per-tile depart_mean_per_min and arrive_mean_per_min, EB-shrunk."""
    tile_ids_list = list(dict.fromkeys(tile_ids))
    if not tile_ids_list:
        return pd.DataFrame(columns=["tile_id", "depart_mean_per_min", "arrive_mean_per_min"])

    per_tile, global_row, window_minutes = _flow_history(
        conn, tile_ids_list, anchor_ts, lookback_days
    )

    global_arr_rate = float(global_row["mean_arrivals"].iloc[0]) / window_minutes
    global_dep_rate = float(global_row["mean_departures"].iloc[0]) / window_minutes

    counts_by_tile = {
        str(r["tile_id"]): (int(r["n_arrivals"]), int(r["n_departures"]))
        for _, r in per_tile.iterrows()
    }

    rows = []
    for tid in tile_ids_list:
        n_a, n_d = counts_by_tile.get(tid, (0, 0))
        # EB shrinkage: combined_rate = (alpha * global_rate + n_events) / (alpha + n_minutes)
        arr_rate = (EB_ALPHA * global_arr_rate + n_a) / (EB_ALPHA + window_minutes)
        dep_rate = (EB_ALPHA * global_dep_rate + n_d) / (EB_ALPHA + window_minutes)
        rows.append({
            "tile_id": tid,
            "arrive_mean_per_min": arr_rate,
            "depart_mean_per_min": dep_rate,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Dock-side aggregation from live_station_predictions.
# ---------------------------------------------------------------------------

def _dock_predictions(
    conn: duckdb.DuckDBPyConnection,
    station_ids: list[str],
    model_key: str,
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    if not station_ids:
        return pd.DataFrame()
    return live_cache.latest_prediction_cache(
        conn,
        model_key,
        station_ids,
        horizons=horizons,
        max_age_minutes=DOCK_PRED_MAX_AGE_MIN,
    )


def _active_model_key(conn: duckdb.DuckDBPyConnection) -> str:
    state = model_selection.latest_selection_state(conn)
    return state.get("active_model_key") or "logistic"


# ---------------------------------------------------------------------------
# Main scoring entry points.
# ---------------------------------------------------------------------------

def score_tiles(
    conn: duckdb.DuckDBPyConnection,
    tile_ids: Iterable[str],
    horizons: tuple[int, ...] = HORIZONS,
    *,
    anchor_ts: datetime | None = None,
    live_bike_ids: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (per-tile current state, per-(tile, horizon) score frame).

    Score columns per (tile_id, horizon_minutes):
      free_p_has_bike, free_p_survives, free_p_appears, free_expected_count,
      free_expected_arrivals, free_expected_departures, free_p_count_json,
      dock_p_any_has_ebike, dock_expected_count, dock_per_station_json,
      combined_p_any_ebike, combined_total_expected_ebikes,
      dock_predictions_as_of (or null if no dock data)
    """
    anchor_ts = anchor_ts or _utc_now()
    state_df = current_tile_state(conn, tile_ids, live_bike_ids=live_bike_ids)
    if state_df.empty:
        return state_df, pd.DataFrame()

    tile_ids_list = state_df["tile_id"].tolist()
    flow_df = tile_flow_priors(conn, tile_ids_list, anchor_ts)
    flow_by_tile = {
        str(r["tile_id"]): (float(r["depart_mean_per_min"]), float(r["arrive_mean_per_min"]))
        for _, r in flow_df.iterrows()
    }

    # Lazy import to avoid a circular dependency — disabled_predictor uses
    # tile_predictor.stations_in_tile_map for the docked side.
    from . import disabled_predictor
    dis_df = disabled_predictor.free_disability_rate_priors(conn, tile_ids_list, anchor_ts)
    dis_by_tile = {
        str(r["tile_id"]): float(r["disability_rate_per_min"])
        for _, r in dis_df.iterrows()
    }

    # Pull dock predictions once for all unique stations in scope.
    model_key = _active_model_key(conn)
    all_station_ids = sorted({
        s["station_id"]
        for _, row in state_df.iterrows()
        for s in row["stations"]
    })
    dock_cache = _dock_predictions(conn, all_station_ids, model_key, horizons)

    # Attach per-bike survival probabilities to each bike dict, using the same
    # flow priors we just computed (avoids the API needing a second query).
    # Disability hazard is added to the departure rate: a free bike that runs
    # out of battery leaves the rider-available pool the same way a ride does.
    augmented_bikes_by_tile: dict[str, list[dict[str, Any]]] = {}
    for _, row in state_df.iterrows():
        tid = row["tile_id"]
        dep_rate, _ = flow_by_tile.get(tid, (0.0, 0.0))
        dis_rate = dis_by_tile.get(tid, 0.0)
        effective_rate = dep_rate + dis_rate
        current_free = int(row["current_free_ebikes"])
        new_bikes = []
        for bike in row["bikes"]:
            p_stays = {
                int(h): tile_dp.per_bike_survival(
                    depart_mean_per_horizon=effective_rate * int(h),
                    current_free_ebikes=current_free,
                    horizon=int(h),
                )
                for h in horizons
            }
            new_bikes.append({**bike, "p_stays": p_stays})
        augmented_bikes_by_tile[tid] = new_bikes
    state_df = state_df.copy()
    state_df["bikes"] = state_df["tile_id"].map(augmented_bikes_by_tile)

    score_rows: list[dict[str, Any]] = []
    for _, row in state_df.iterrows():
        tid = row["tile_id"]
        current_free = int(row["current_free_ebikes"])
        dep_rate, arr_rate = flow_by_tile.get(tid, (0.0, 0.0))
        dis_rate = dis_by_tile.get(tid, 0.0)
        effective_rate = dep_rate + dis_rate
        station_ids = [s["station_id"] for s in row["stations"]]
        dock_rows = (
            dock_cache[dock_cache["station_id"].astype(str).isin(station_ids)]
            if not dock_cache.empty
            else dock_cache
        )
        dock_as_of = (
            pd.to_datetime(dock_rows["as_of"]).max().to_pydatetime()
            if not dock_rows.empty
            else None
        )

        for horizon in horizons:
            free_result = tile_dp.tile_rollout(
                current_free_ebikes=current_free,
                depart_mean_per_horizon=effective_rate * horizon,
                arrive_mean_per_horizon=arr_rate * horizon,
                horizon=int(horizon),
            )
            # tile_rollout reports total drain events under the effective rate;
            # split them proportionally so free_expected_departures keeps the
            # original "rider-driven" semantics. With dis_rate=0 (the dominant
            # case for Divvy today) this collapses to the previous behavior.
            total_drain = float(free_result["expected_departures"])
            dep_share = dep_rate / effective_rate if effective_rate > 0 else 1.0
            expected_departures = total_drain * dep_share
            expected_disabilities = total_drain - expected_departures

            per_station: list[dict[str, Any]] = []
            dock_no_ebike_product = 1.0
            dock_expected_count = 0.0
            if not dock_rows.empty:
                horizon_rows = dock_rows[dock_rows["horizon_minutes"].astype(int) == int(horizon)]
                station_meta = {s["station_id"]: s for s in row["stations"]}
                for _, dr in horizon_rows.iterrows():
                    sid = str(dr["station_id"])
                    p = float(dr.get("p_has_ebike") or 0.0)
                    expected = float(dr.get("expected_ebikes") or 0.0)
                    dock_no_ebike_product *= max(0.0, min(1.0, 1.0 - p))
                    dock_expected_count += expected
                    meta = station_meta.get(sid, {})
                    per_station.append({
                        "station_id": sid,
                        "name": meta.get("name"),
                        "p_has_ebike": p,
                        "expected_ebikes": expected,
                    })
            dock_p_any = 1.0 - dock_no_ebike_product if per_station else 0.0

            combined_p_any = 1.0 - (1.0 - free_result["p_has_bike"]) * (1.0 - dock_p_any)
            combined_total_expected = free_result["expected_count"] + dock_expected_count

            score_rows.append({
                "tile_id": tid,
                "horizon_minutes": int(horizon),
                "free_p_has_bike": free_result["p_has_bike"],
                "free_p_survives": free_result["p_survives"],
                "free_p_appears": free_result["p_appears"],
                "free_expected_count": free_result["expected_count"],
                "free_expected_arrivals": free_result["expected_arrivals"],
                "free_expected_departures": expected_departures,
                "free_expected_disabilities": expected_disabilities,
                "free_disability_rate_per_min": dis_rate,
                "free_p_count": free_result["p_count"],
                "dock_p_any_has_ebike": dock_p_any,
                "dock_expected_count": dock_expected_count,
                "dock_per_station": per_station,
                "combined_p_any_ebike": combined_p_any,
                "combined_total_expected_ebikes": combined_total_expected,
                "dock_predictions_as_of": dock_as_of,
            })

    return state_df, pd.DataFrame(score_rows)


def score_single_bike(
    conn: duckdb.DuckDBPyConnection,
    bike_id: str,
    tile_id: str,
    horizons: tuple[int, ...] = HORIZONS,
    *,
    anchor_ts: datetime | None = None,
) -> dict[int, float]:
    """Return {horizon_minutes: p_stays} for one specific free ebike in a tile.

    ``p_stays`` accounts for both ride-departures and the per-bike hazard of
    becoming disabled (running out of battery / breaking) while in this tile.
    """
    anchor_ts = anchor_ts or _utc_now()
    state_df = current_tile_state(conn, [tile_id])
    if state_df.empty:
        return {int(h): 1.0 for h in horizons}
    row = state_df.iloc[0]
    current_free = int(row["current_free_ebikes"])
    flow_df = tile_flow_priors(conn, [tile_id], anchor_ts)
    if flow_df.empty:
        depart_rate = 0.0
    else:
        depart_rate = float(flow_df["depart_mean_per_min"].iloc[0])
    from . import disabled_predictor
    dis_df = disabled_predictor.free_disability_rate_priors(conn, [tile_id], anchor_ts)
    disability_rate = (
        float(dis_df["disability_rate_per_min"].iloc[0]) if not dis_df.empty else 0.0
    )
    effective_rate = depart_rate + disability_rate
    return {
        int(h): tile_dp.per_bike_survival(
            depart_mean_per_horizon=effective_rate * int(h),
            current_free_ebikes=current_free,
            horizon=int(h),
        )
        for h in horizons
    }
