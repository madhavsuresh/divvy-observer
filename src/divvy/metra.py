"""Metra commuter-rail schedule features for Divvy demand prediction.

Empirically (`tools/metra_event_study`), trip-start rates at Divvy stations
near Metra stops spike measurably right after weekday train arrivals (and
trip-end rates spike right before train departures, when commuters drop off
bikes to catch a train). The signal is station-specific — Ravenswood +26%,
Irving Park +105%, Clybourn +46% (and 76% drop-off lift), with downtown
terminals like Union Station and LaSalle showing 20-30% pickup lifts.

This module loads three pre-built data artifacts under ``data/metra/``:

- ``stops.parquet``           - Metra stop_id, name, lat, lon
- ``weekday_schedule.parquet`` - per (stop_id, minute_of_day) arrival /
                                 departure event counts on a typical weekday
- ``pair_lifts.parquet``       - per (Divvy station name, Metra stop) the
                                 measured pickup-lift and drop-off-lift
                                 coefficients with weekend control

and exposes:

- ``get_metra_features(divvy_station_name, ts) -> dict``: returns a small
  feature dict suitable to merge into a per-station candidate row at
  prediction or training time. Stations not near any Metra stop return an
  all-zero / `is_near_metra=False` dict so downstream models can ignore.

- ``find_pair(divvy_station_name)``: lookup the pre-computed pair record
  for a station, or None if it has no Metra neighbor within 400 m.

The data files are refreshed by ``python -m divvy.metra sync``, which
re-downloads the upstream Metra GTFS feed and re-runs the event study on
the latest divvy_trips. The ``sync-metra`` automation job runs this
quarterly (Metra's GTFS schedule changes infrequently, and the per-pair
event study is expensive — no value in daily refreshes).
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import math
import sys
import zipfile
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import duckdb
import pandas as pd
import requests

from . import config, db


log = logging.getLogger("divvy.metra")

METRA_GTFS_URL = "https://schedules.metrarail.com/gtfs/schedule.zip"
METRA_DATA_DIR = config.DATA_DIR / "metra"
PAIR_DISTANCE_THRESHOLD_M = 400.0
EVENT_STUDY_WINDOW_MIN = 10
MIN_TRIPS_FOR_LIFT_ESTIMATE = 500


def _strip_strings(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object or str(df[c].dtype) == "str":
            df[c] = df[c].astype(str).str.strip()
    return df


def _to_minute(t: str) -> int | None:
    try:
        h, m, _s = t.split(":")
        return (int(h) * 60 + int(m)) % 1440
    except (AttributeError, ValueError):
        return None


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# Lazy-loaded data accessors. Modules that consume the features import these
# helpers; data files are only opened on first call.
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _load_stops() -> pd.DataFrame:
    path = METRA_DATA_DIR / "stops.parquet"
    if not path.exists():
        log.warning("metra stops parquet missing at %s", path)
        return pd.DataFrame(columns=["stop_id", "stop_name", "stop_lat", "stop_lon"])
    return pd.read_parquet(path)


@lru_cache(maxsize=1)
def _load_schedule() -> pd.DataFrame:
    path = METRA_DATA_DIR / "weekday_schedule.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["stop_id", "minute_of_day", "n_arrivals", "n_departures"])
    return pd.read_parquet(path)


@lru_cache(maxsize=1)
def _load_pair_lifts() -> pd.DataFrame:
    path = METRA_DATA_DIR / "pair_lifts.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["divvy_name"] = df["divvy_name"].astype(str).str.strip()
    return df


@lru_cache(maxsize=1)
def _pair_lookup_by_divvy_name() -> dict[str, dict]:
    """Closest-Metra-pair record keyed by Divvy station name.

    A Divvy station can be near multiple Metra stops; we keep the one with
    the strongest pickup effect (largest ``metra_effect_pickup``) so the
    feature lookup is unambiguous.
    """
    df = _load_pair_lifts()
    if df.empty:
        return {}
    best = df.sort_values("metra_effect_pickup", ascending=False).drop_duplicates("divvy_name", keep="first")
    return {row["divvy_name"]: row.to_dict() for _, row in best.iterrows()}


def _schedule_lookup(stop_id: str) -> pd.DataFrame:
    """Slice the weekday schedule down to one Metra stop, indexed by minute_of_day."""
    sched = _load_schedule()
    sub = sched[sched["stop_id"] == stop_id].set_index("minute_of_day")
    return sub


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

ZERO_FEATURES = {
    "is_near_metra": 0.0,
    "metra_distance_m": 0.0,
    "metra_arr_in_5m": 0.0,
    "metra_arr_in_10m": 0.0,
    "metra_arr_in_30m": 0.0,
    "metra_dep_in_5m": 0.0,
    "metra_dep_in_10m": 0.0,
    "metra_dep_in_30m": 0.0,
    "metra_arr_in_last_5m": 0.0,
    "metra_arr_in_last_10m": 0.0,
    "metra_dep_in_last_5m": 0.0,
    "metra_pickup_lift": 0.0,
    "metra_dropoff_lift": 0.0,
    "metra_pickup_lift_x_arr_5m": 0.0,
    "metra_dropoff_lift_x_dep_5m": 0.0,
}


def get_metra_features(divvy_station_name: str | None, ts: datetime | pd.Timestamp | None) -> dict[str, float]:
    """Return Metra-event features for one Divvy station at one timestamp.

    The output is a flat dict of floats safe to merge into a feature row.
    Stations with no nearby Metra stop or with insufficient lift sample size
    return all zeros and ``is_near_metra=0.0``, so downstream models can
    treat the features uniformly.

    The schedule is "current weekday" (single repeating Mon-Fri pattern),
    not historical — see module docstring for the staleness caveat at
    training time.
    """
    if divvy_station_name is None or ts is None:
        return dict(ZERO_FEATURES)
    pair = _pair_lookup_by_divvy_name().get(str(divvy_station_name).strip())
    if pair is None:
        return dict(ZERO_FEATURES)

    # We only emit Metra features Mon-Fri (weekday schedule is what we have).
    # Saturday/Sunday rows fall through to zeros — Metra runs different
    # services and our event-study lifts were measured weekday-only.
    if hasattr(ts, "weekday"):
        if ts.weekday() >= 5:
            return dict(ZERO_FEATURES)
    else:
        return dict(ZERO_FEATURES)

    sched = _schedule_lookup(pair["metra_id"])
    if sched.empty:
        return dict(ZERO_FEATURES)

    minute = ts.hour * 60 + ts.minute

    def window_count(col: str, lo_offset: int, hi_offset: int) -> int:
        # Sum events in [minute + lo_offset, minute + hi_offset] mod 1440
        idx = [(minute + k) % 1440 for k in range(lo_offset, hi_offset + 1)]
        existing = [i for i in idx if i in sched.index]
        if not existing:
            return 0
        return int(sched.loc[existing, col].sum())

    arr_5 = window_count("n_arrivals", 1, 5)
    arr_10 = window_count("n_arrivals", 1, 10)
    arr_30 = window_count("n_arrivals", 1, 30)
    dep_5 = window_count("n_departures", 1, 5)
    dep_10 = window_count("n_departures", 1, 10)
    dep_30 = window_count("n_departures", 1, 30)
    arr_last_5 = window_count("n_arrivals", -5, -1)
    arr_last_10 = window_count("n_arrivals", -10, -1)
    dep_last_5 = window_count("n_departures", -5, -1)

    pickup_lift = float(pair.get("metra_effect_pickup") or 0.0)
    dropoff_lift = float(pair.get("lift_dropoff_wkd") or 0.0)

    return {
        "is_near_metra": 1.0,
        "metra_distance_m": float(pair.get("distance_m") or 0.0),
        "metra_arr_in_5m": float(arr_5),
        "metra_arr_in_10m": float(arr_10),
        "metra_arr_in_30m": float(arr_30),
        "metra_dep_in_5m": float(dep_5),
        "metra_dep_in_10m": float(dep_10),
        "metra_dep_in_30m": float(dep_30),
        "metra_arr_in_last_5m": float(arr_last_5),
        "metra_arr_in_last_10m": float(arr_last_10),
        "metra_dep_in_last_5m": float(dep_last_5),
        "metra_pickup_lift": pickup_lift,
        "metra_dropoff_lift": dropoff_lift,
        # Interaction features so a linear-ish model can use them directly.
        # Big positive when a station with a big lift coefficient also has
        # a train arriving soon.
        "metra_pickup_lift_x_arr_5m": pickup_lift * float(arr_last_5),
        "metra_dropoff_lift_x_dep_5m": dropoff_lift * float(dep_5),
    }


def attach_metra_features(rows: pd.DataFrame, ts_col: str = "as_of") -> pd.DataFrame:
    """Vectorized version of get_metra_features for batch feature engineering.

    Adds one column per key in ZERO_FEATURES to ``rows`` based on
    ``rows['name']`` (the Divvy station name) and ``rows[ts_col]``.
    """
    out = rows.copy()
    feat_keys = list(ZERO_FEATURES.keys())
    if rows.empty:
        for k in feat_keys:
            out[k] = pd.Series(dtype=float)
        return out
    if "name" not in rows.columns:
        log.warning("attach_metra_features: rows missing 'name' column; skipping")
        for k in feat_keys:
            out[k] = 0.0
        return out
    if ts_col not in rows.columns:
        log.warning("attach_metra_features: rows missing %r column; using current time", ts_col)
        ts_series = pd.Series([datetime.now(timezone.utc)] * len(rows))
    else:
        ts_series = pd.to_datetime(rows[ts_col], errors="coerce")

    feats = []
    for name, ts in zip(rows["name"], ts_series):
        feats.append(get_metra_features(name, ts if pd.notna(ts) else None))
    feats_df = pd.DataFrame(feats, index=rows.index)
    for k in feat_keys:
        out[k] = feats_df[k].values
    return out


# ---------------------------------------------------------------------------
# Sync: refresh Metra GTFS + per-pair lift coefficients
# ---------------------------------------------------------------------------


def _download_gtfs(timeout: int = 60) -> bytes:
    log.info("fetching Metra GTFS from %s", METRA_GTFS_URL)
    response = requests.get(METRA_GTFS_URL, timeout=timeout)
    response.raise_for_status()
    return response.content


def _gtfs_to_artifacts(gtfs_bytes: bytes, conn: duckdb.DuckDBPyConnection) -> dict:
    """Parse GTFS, build stops + weekday_schedule + pair_lifts artifacts.

    The pair_lifts step queries divvy_trips on ``conn``; passing a read-only
    connection is fine. Writes parquet files into ``METRA_DATA_DIR``.
    """
    METRA_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(gtfs_bytes)) as zf:
        def _read(name: str) -> pd.DataFrame:
            with zf.open(name) as fh:
                return _strip_strings(pd.read_csv(fh))
        stops = _read("stops.txt")
        trips_g = _read("trips.txt")
        st = _read("stop_times.txt")
        cal = _read("calendar.txt")

    # Active weekday services (every weekday flag = 1)
    wkd_svc = cal[
        (cal.monday == 1) & (cal.tuesday == 1) & (cal.wednesday == 1)
        & (cal.thursday == 1) & (cal.friday == 1)
    ].service_id.tolist()
    wkd_trip_ids = set(trips_g[trips_g.service_id.isin(wkd_svc)].trip_id.tolist())
    wkd_st = st[st.trip_id.isin(wkd_trip_ids)].copy()
    wkd_st["arr_min"] = wkd_st.arrival_time.apply(_to_minute)
    wkd_st["dep_min"] = wkd_st.departure_time.apply(_to_minute)

    # stops + schedule artifacts
    stops_out = stops[["stop_id", "stop_name", "stop_lat", "stop_lon"]].copy()
    arrivals = wkd_st.groupby(["stop_id", "arr_min"]).size().reset_index(name="n_arrivals")
    departures = wkd_st.groupby(["stop_id", "dep_min"]).size().reset_index(name="n_departures")
    arrivals.columns = ["stop_id", "minute_of_day", "n_arrivals"]
    departures.columns = ["stop_id", "minute_of_day", "n_departures"]
    sched = pd.merge(arrivals, departures, on=["stop_id", "minute_of_day"], how="outer").fillna(0)
    sched[["n_arrivals", "n_departures"]] = sched[["n_arrivals", "n_departures"]].astype("int32")
    sched["minute_of_day"] = sched["minute_of_day"].astype("int32")

    stops_out.to_parquet(METRA_DATA_DIR / "stops.parquet", index=False)
    sched.to_parquet(METRA_DATA_DIR / "weekday_schedule.parquet", index=False)

    # Build pair_lifts via event study against current divvy_trips
    log.info("computing pair lifts via event study on divvy_trips")
    pair_lifts_df = _compute_pair_lifts(conn, stops_out, wkd_st)
    pair_lifts_df.to_parquet(METRA_DATA_DIR / "pair_lifts.parquet", index=False)

    meta = {
        "gtfs_fetched_at": datetime.now(timezone.utc).isoformat(),
        "gtfs_source": METRA_GTFS_URL,
        "stops_count": int(len(stops_out)),
        "schedule_rows": int(len(sched)),
        "pair_lifts_count": int(len(pair_lifts_df)),
        "pair_lifts_meaningful_pickup_10pct": int((pair_lifts_df["metra_effect_pickup"] >= 0.10).sum()),
        "pair_lifts_meaningful_dropoff_10pct": int((pair_lifts_df["lift_dropoff_wkd"] >= 0.10).sum()),
    }
    (METRA_DATA_DIR / "meta.json").write_text(json.dumps(meta, indent=2))

    # Bust caches so the next get_metra_features call sees fresh data.
    _load_stops.cache_clear()
    _load_schedule.cache_clear()
    _load_pair_lifts.cache_clear()
    _pair_lookup_by_divvy_name.cache_clear()
    return meta


def _compute_pair_lifts(
    conn: duckdb.DuckDBPyConnection,
    stops: pd.DataFrame,
    wkd_st: pd.DataFrame,
) -> pd.DataFrame:
    # 1. Spatial join: Divvy stations within PAIR_DISTANCE_THRESHOLD_M of any Metra stop.
    divvy_stations = conn.execute(
        "SELECT station_id, name, lat, lon FROM stations WHERE lat IS NOT NULL AND lon IS NOT NULL"
    ).df()
    metra_chi = stops[
        (stops.stop_lat.between(41.6, 42.05))
        & (stops.stop_lon.between(-87.95, -87.5))
    ]
    pairs = []
    for _, m in metra_chi.iterrows():
        for _, d in divvy_stations.iterrows():
            dist = _haversine_m(m.stop_lat, m.stop_lon, d.lat, d.lon)
            if dist <= PAIR_DISTANCE_THRESHOLD_M:
                pairs.append({
                    "metra_id": m.stop_id, "metra_name": m.stop_name,
                    "divvy_id": d.station_id, "divvy_name": d["name"],
                    "distance_m": round(dist, 1),
                })
    pairs_df = pd.DataFrame(pairs)
    if pairs_df.empty:
        return pd.DataFrame(columns=[
            "metra_name", "divvy_name", "distance_m", "n_arrivals",
            "wkd_starts_12mo", "lift_pickup_wkd", "metra_effect_pickup", "lift_dropoff_wkd",
        ])

    # 2. Per-stop arrival/departure minute lists.
    metra_arr_minutes: dict[str, list[int]] = {}
    metra_dep_minutes: dict[str, list[int]] = {}
    for stop_id, sub in wkd_st.groupby("stop_id"):
        if stop_id not in pairs_df.metra_id.values:
            continue
        metra_arr_minutes[stop_id] = sub.arr_min.dropna().astype(int).tolist()
        metra_dep_minutes[stop_id] = sub.dep_min.dropna().astype(int).tolist()

    # 3. Event study per pair.
    results = []
    for p in pairs_df.itertuples():
        arr = metra_arr_minutes.get(p.metra_id, [])
        dep = metra_dep_minutes.get(p.metra_id, [])
        if not arr:
            continue
        wkd_starts_df = conn.execute(
            """
            SELECT EXTRACT(HOUR FROM started_at)::INT*60 + EXTRACT(MINUTE FROM started_at)::INT AS m,
                   COUNT(*) AS n
            FROM divvy_trips
            WHERE start_station_name = ?
              AND started_at > now() - INTERVAL 12 MONTH
              AND EXTRACT(DOW FROM started_at) BETWEEN 1 AND 5
            GROUP BY m
            """,
            [p.divvy_name],
        ).df()
        wkd_ends_df = conn.execute(
            """
            SELECT EXTRACT(HOUR FROM ended_at)::INT*60 + EXTRACT(MINUTE FROM ended_at)::INT AS m,
                   COUNT(*) AS n
            FROM divvy_trips
            WHERE end_station_name = ?
              AND ended_at > now() - INTERVAL 12 MONTH
              AND EXTRACT(DOW FROM ended_at) BETWEEN 1 AND 5
            GROUP BY m
            """,
            [p.divvy_name],
        ).df()
        wke_starts_df = conn.execute(
            """
            SELECT EXTRACT(HOUR FROM started_at)::INT*60 + EXTRACT(MINUTE FROM started_at)::INT AS m,
                   COUNT(*) AS n
            FROM divvy_trips
            WHERE start_station_name = ?
              AND started_at > now() - INTERVAL 12 MONTH
              AND EXTRACT(DOW FROM started_at) IN (0, 6)
            GROUP BY m
            """,
            [p.divvy_name],
        ).df()
        wkd_starts = wkd_starts_df.set_index("m").n.reindex(range(1440), fill_value=0).values
        wkd_ends = wkd_ends_df.set_index("m").n.reindex(range(1440), fill_value=0).values
        wke_starts = wke_starts_df.set_index("m").n.reindex(range(1440), fill_value=0).values
        n_wkd_starts = int(wkd_starts.sum())
        if n_wkd_starts < MIN_TRIPS_FOR_LIFT_ESTIMATE:
            continue

        def window_lift(series, events, post: bool, window: int = EVENT_STUDY_WINDOW_MIN):
            pre_total = post_total = 0
            for m in events:
                for k in range(1, window + 1):
                    pre_total += series[(m - k) % 1440]
                    post_total += series[(m + k) % 1440]
            n = window * len(events)
            pre_rate = pre_total / n if n else 0
            post_rate = post_total / n if n else 0
            if not post:
                # for drop-off: pre>post means people drop off BEFORE the event
                return (pre_rate - post_rate) / post_rate if post_rate > 0 else 0
            return (post_rate - pre_rate) / pre_rate if pre_rate > 0 else 0

        lift_pickup_wkd = window_lift(wkd_starts, arr, post=True)
        lift_pickup_wke = window_lift(wke_starts, arr, post=True)
        lift_dropoff_wkd = window_lift(wkd_ends, dep, post=False)
        results.append({
            "metra_name": p.metra_name,
            "divvy_name": p.divvy_name,
            "distance_m": p.distance_m,
            "n_arrivals": len(arr),
            "n_departures": len(dep),
            "wkd_starts_12mo": n_wkd_starts,
            "lift_pickup_wkd": round(lift_pickup_wkd, 3),
            "lift_pickup_wke": round(lift_pickup_wke, 3),
            "metra_effect_pickup": round(lift_pickup_wkd - lift_pickup_wke, 3),
            "lift_dropoff_wkd": round(lift_dropoff_wkd, 3),
        })
    return pd.DataFrame(results)


def sync(*, force: bool = False) -> dict:
    """Refresh the Metra artifacts from upstream.

    Returns the meta dict written to ``data/metra/meta.json``.
    """
    gtfs_bytes = _download_gtfs()
    with db.session(read_only=False) as conn:
        return _gtfs_to_artifacts(gtfs_bytes, conn)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Metra GTFS sync + per-pair Divvy lift study")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("sync", help="Re-download Metra GTFS and rebuild lift coefficients.")
    show = sub.add_parser("show", help="Inspect features for one Divvy station.")
    show.add_argument("station_name")
    show.add_argument("--at", help="ISO timestamp; defaults to now (Chicago time).")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s", stream=sys.stdout)
    if args.command == "sync":
        meta = sync()
        print(json.dumps(meta, indent=2))
    elif args.command == "show":
        ts = pd.Timestamp(args.at) if args.at else pd.Timestamp.now()
        feats = get_metra_features(args.station_name, ts)
        print(json.dumps(feats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
