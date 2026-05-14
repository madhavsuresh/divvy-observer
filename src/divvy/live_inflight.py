from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Iterable

import duckdb
import pandas as pd

from . import db, predictor


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _ids_clause(ids: Iterable[str]) -> tuple[str, list[str]]:
    values = [str(value) for value in ids if value is not None]
    if not values:
        return "", []
    return ",".join(["?"] * len(values)), values


def _local_hour_dow(ts: datetime) -> tuple[int, int]:
    local = pd.Timestamp(ts, tz="UTC").tz_convert(predictor.LOCAL_TZ)
    return int(local.hour), int(local.dayofweek)


def update_live_inflight_arrivals(
    conn: duckdb.DuckDBPyConnection,
    lookback_minutes: int = 30,
    horizons: tuple[int, ...] = predictor.HORIZONS,
) -> dict:
    """Infer near-term in-flight bikes from live station-status deltas.

    Departures are allocated to historical OD route priors. Rows are cumulative
    by horizon: a row at 10 minutes means expected arrivals due by 10 minutes.
    """
    db.init_schema(conn)
    now = _utc_now()
    horizon_values = tuple(int(h) for h in horizons)
    if not horizon_values:
        return {"status": "skipped", "rows_inserted": 0, "departures": 0}
    max_horizon = max(horizon_values)
    conn.execute("DELETE FROM live_inflight_arrivals WHERE expires_at < ?", [now])
    deltas = conn.execute(
        """
        WITH recent AS (
          SELECT
            station_id,
            last_reported,
            COALESCE(num_ebikes_available, 0) AS ebikes,
            GREATEST(COALESCE(num_bikes_available, 0) - COALESCE(num_ebikes_available, 0), 0) AS classic_bikes,
            LAG(COALESCE(num_ebikes_available, 0)) OVER (
              PARTITION BY station_id ORDER BY last_reported
            ) AS prev_ebikes,
            LAG(GREATEST(COALESCE(num_bikes_available, 0) - COALESCE(num_ebikes_available, 0), 0)) OVER (
              PARTITION BY station_id ORDER BY last_reported
            ) AS prev_classic_bikes
          FROM station_status
          WHERE last_reported >= ? - (? * INTERVAL '1 minute')
        )
        SELECT station_id, last_reported, ebikes, classic_bikes, prev_ebikes, prev_classic_bikes
        FROM recent
        WHERE prev_ebikes IS NOT NULL
          AND (ebikes < prev_ebikes OR classic_bikes < prev_classic_bikes)
        ORDER BY last_reported
        """,
        [now, int(lookback_minutes)],
    ).df()
    if deltas.empty:
        return {"status": "ok", "rows_inserted": 0, "departures": 0}

    insert_rows: list[tuple] = []
    departures = 0.0
    for row in deltas.itertuples(index=False):
        source_id = str(row.station_id)
        ebike_departures = max(0.0, float(row.prev_ebikes or 0) - float(row.ebikes or 0))
        classic_departures = max(0.0, float(row.prev_classic_bikes or 0) - float(row.classic_bikes or 0))
        if ebike_departures <= 0 and classic_departures <= 0:
            continue
        departures += ebike_departures + classic_departures
        hour, dow = _local_hour_dow(pd.Timestamp(row.last_reported).to_pydatetime())
        routes = conn.execute(
            """
            SELECT
              end_station_id,
              trips,
              ebike_trips,
              COALESCE(median_duration_minutes, avg_duration_minutes, 12.0) AS duration
            FROM station_trip_routes
            WHERE start_station_id = ?
              AND local_hour = ?
              AND dow = ?
              AND end_station_id IS NOT NULL
              AND end_station_id <> ?
            ORDER BY trips DESC
            LIMIT 40
            """,
            [source_id, hour, dow, source_id],
        ).df()
        if routes.empty:
            routes = conn.execute(
                """
                SELECT
                  end_station_id,
                  trips,
                  ebike_trips,
                  COALESCE(median_duration_minutes, avg_duration_minutes, 12.0) AS duration
                FROM station_trip_routes
                WHERE start_station_id = ?
                  AND end_station_id IS NOT NULL
                  AND end_station_id <> ?
                ORDER BY trips DESC
                LIMIT 40
                """,
                [source_id, source_id],
            ).df()
        if routes.empty:
            continue
        routes["trips"] = pd.to_numeric(routes["trips"], errors="coerce").fillna(0.0)
        total_trips = float(routes["trips"].sum())
        if total_trips <= 0:
            continue
        routes["route_probability"] = routes["trips"] / total_trips
        routes["duration"] = pd.to_numeric(routes["duration"], errors="coerce").fillna(12.0).clip(lower=1.0)
        for r in routes.itertuples(index=False):
            p_dst = float(r.route_probability)
            duration = float(r.duration)
            for horizon in horizon_values:
                if duration > horizon:
                    due_fraction = max(0.0, 1.0 - (duration - horizon) / max(duration, 1.0)) * 0.25
                else:
                    due_fraction = min(1.0, 0.65 + 0.35 * (horizon - duration) / max(max_horizon, 1))
                if due_fraction <= 0:
                    continue
                insert_rows.append((
                    now,
                    source_id,
                    str(r.end_station_id),
                    int(horizon),
                    float(ebike_departures * p_dst * due_fraction),
                    float(classic_departures * p_dst * due_fraction),
                    now + timedelta(minutes=max_horizon + 5),
                    pd.Timestamp(row.last_reported).to_pydatetime(),
                    now,
                ))

    if insert_rows:
        conn.executemany(
            """
            INSERT INTO live_inflight_arrivals (
              updated_at, source_station_id, dst_station_id, horizon_minutes,
              ebike_mass, classic_mass, expires_at, source_status_at, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            insert_rows,
        )
    return {
        "status": "ok",
        "rows_inserted": len(insert_rows),
        "departures": departures,
    }


def get_live_inflight_features(
    conn: duckdb.DuckDBPyConnection,
    station_ids: Iterable[str],
    now: datetime | None = None,
    horizons: tuple[int, ...] = predictor.HORIZONS,
) -> pd.DataFrame:
    station_ids_list = [str(station_id) for station_id in station_ids]
    columns = ["station_id"]
    for horizon in horizons:
        columns.extend([
            f"live_inflight_ebike_due_{int(horizon)}m",
            f"live_inflight_classic_due_{int(horizon)}m",
        ])
    base = pd.DataFrame({"station_id": station_ids_list})
    for column in columns:
        if column != "station_id":
            base[column] = 0.0
    if not station_ids_list:
        return base[columns]
    now = now or _utc_now()
    placeholders, params = _ids_clause(station_ids_list)
    rows = conn.execute(
        f"""
        SELECT
          dst_station_id AS station_id,
          horizon_minutes,
          SUM(ebike_mass) AS ebike_mass,
          SUM(classic_mass) AS classic_mass
        FROM live_inflight_arrivals
        WHERE dst_station_id IN ({placeholders})
          AND expires_at >= ?
        GROUP BY dst_station_id, horizon_minutes
        """,
        [*params, now],
    ).df()
    if rows.empty:
        return base[columns]
    for horizon in horizons:
        h = rows[rows["horizon_minutes"].astype(int) == int(horizon)]
        if h.empty:
            continue
        h = h[["station_id", "ebike_mass", "classic_mass"]].rename(
            columns={
                "ebike_mass": f"live_inflight_ebike_due_{int(horizon)}m",
                "classic_mass": f"live_inflight_classic_due_{int(horizon)}m",
            }
        )
        base = base.merge(h, on="station_id", how="left", suffixes=("", "_new"))
        for prefix in ["ebike", "classic"]:
            col = f"live_inflight_{prefix}_due_{int(horizon)}m"
            new_col = f"{col}_new"
            if new_col in base.columns:
                base[col] = pd.to_numeric(base[new_col], errors="coerce").fillna(base[col]).fillna(0.0)
                base = base.drop(columns=[new_col])
    for column in columns:
        if column != "station_id":
            base[column] = pd.to_numeric(base[column], errors="coerce").fillna(0.0)
    return base[columns]
