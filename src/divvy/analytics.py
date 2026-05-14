from __future__ import annotations

from datetime import datetime, timedelta, timezone

import duckdb
import numpy as np
import pandas as pd

LOCAL_TZ = "America/Chicago"


def _local_expr(col: str) -> str:
    return f"(CAST({col} AS TIMESTAMP) AT TIME ZONE 'UTC' AT TIME ZONE '{LOCAL_TZ}')"


def _table_has_rows(conn: duckdb.DuckDBPyConnection, table_name: str) -> bool:
    try:
        row = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
        return bool(row and row[0])
    except duckdb.Error:
        return False


def _station_trip_ids(conn: duckdb.DuckDBPyConnection, station_id: str) -> list[str]:
    rows = conn.execute(
        """
        SELECT station_id, legacy_id, short_name
        FROM stations
        WHERE station_id = ?
        """,
        [station_id],
    ).fetchone()
    ids = [station_id]
    if rows:
        ids.extend([value for value in rows if value])
    return sorted(set(str(value) for value in ids if value))


def station_options(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return conn.execute(
        """
        SELECT s.station_id, s.short_name, s.name, s.capacity, s.lat, s.lon,
               COUNT(ss.station_id) AS n_obs,
               MAX(ss.last_reported) AS last_obs
        FROM stations s
        LEFT JOIN station_status ss USING (station_id)
        GROUP BY s.station_id, s.short_name, s.name, s.capacity, s.lat, s.lon
        ORDER BY s.name
        """
    ).df()


def nearest_stations(
    conn: duckdb.DuckDBPyConnection,
    lat: float,
    lon: float,
    k: int = 3,
) -> pd.DataFrame:
    """Return the k stations nearest to (lat, lon), with haversine distance in km.

    Uses the haversine formula directly in SQL so we don't have to pull every
    station into Python. Earth radius = 6371 km.
    """
    return conn.execute(
        """
        SELECT
          station_id,
          short_name,
          name,
          capacity,
          lat,
          lon,
          6371.0 * 2.0 * ASIN(
            SQRT(
              POWER(SIN(RADIANS(lat - ?) / 2.0), 2)
              + COS(RADIANS(?)) * COS(RADIANS(lat))
                * POWER(SIN(RADIANS(lon - ?) / 2.0), 2)
            )
          ) AS distance_km
        FROM stations
        WHERE lat IS NOT NULL AND lon IS NOT NULL
        ORDER BY distance_km
        LIMIT ?
        """,
        [lat, lat, lon, k],
    ).df()


def stations_with_ebikes_nearby(
    conn: duckdb.DuckDBPyConnection,
    lat: float,
    lon: float,
    radius_km: float = 2.0,
    limit: int = 30,
) -> pd.DataFrame:
    """Stations within `radius_km` that currently have ≥1 ebike available.

    Joins each station's latest `station_status` row, filters to ebikes ≥ 1,
    bounds by distance, returns at most `limit` rows sorted by distance.

    Columns: station_id, name, lat, lon, capacity, ebikes_available,
             classic_bikes, last_reported, distance_km.
    """
    return conn.execute(
        """
        WITH latest AS (
          SELECT station_id, num_bikes_available, num_ebikes_available,
                 (num_bikes_available - num_ebikes_available) AS classic_bikes,
                 last_reported
          FROM (
            SELECT *, ROW_NUMBER() OVER (
              PARTITION BY station_id ORDER BY last_reported DESC
            ) AS rn
            FROM station_status
          )
          WHERE rn = 1
        ),
        joined AS (
          SELECT
            s.station_id, s.name, s.lat, s.lon, s.capacity,
            l.num_ebikes_available AS ebikes_available,
            l.classic_bikes,
            l.last_reported,
            6371.0 * 2.0 * ASIN(
              SQRT(
                POWER(SIN(RADIANS(s.lat - ?) / 2.0), 2)
                + COS(RADIANS(?)) * COS(RADIANS(s.lat))
                  * POWER(SIN(RADIANS(s.lon - ?) / 2.0), 2)
              )
            ) AS distance_km
          FROM stations s
          JOIN latest l USING (station_id)
          WHERE s.lat IS NOT NULL AND s.lon IS NOT NULL
            AND l.num_ebikes_available >= 1
        )
        SELECT * FROM joined
        WHERE distance_km <= ?
        ORDER BY distance_km
        LIMIT ?
        """,
        [lat, lat, lon, radius_km, limit],
    ).df()


def free_bike_density(
    conn: duckdb.DuckDBPyConnection,
    lat: float,
    lon: float,
    radius_km: float = 2.0,
    hours: int = 168,
) -> pd.DataFrame:
    """Recent free-bike positions near (lat, lon) for density visualization.

    Returns one row per stored position event in the window, scoped to a
    bounding box around (lat, lon) — pydeck's HexagonLayer will bin these
    into hex cells for the heatmap. We use a rough lat/lon box (no haversine)
    for speed; the slight over-fetch beyond `radius_km` is fine.
    """
    # 1° lat ≈ 111 km; cos(lat) ≈ 0.74 in Chicago, so 1° lon ≈ 82 km.
    lat_delta = radius_km / 110.574
    lon_delta = radius_km / 85.0
    return conn.execute(
        f"""
        SELECT lat, lon, fetched_at, bike_id
        FROM free_bike_status
        WHERE fetched_at > now() - INTERVAL '{int(hours)} hours'
          AND NOT is_disabled
          AND lat BETWEEN ? AND ?
          AND lon BETWEEN ? AND ?
        """,
        [lat - lat_delta, lat + lat_delta, lon - lon_delta, lon + lon_delta],
    ).df()


def free_bikes_in_box(
    conn: duckdb.DuckDBPyConnection,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    hours: int = 168,
) -> pd.DataFrame:
    """Free-bike events recorded in a lat/lon bounding box over the last N hours.

    Used to drill into a clicked hex cell. Returns one row per *position event*
    (each time a bike's lat/lon changed). Sorted newest first.
    """
    return conn.execute(
        f"""
        SELECT bike_id, name, lat, lon, fetched_at
        FROM free_bike_status
        WHERE fetched_at > now() - INTERVAL '{int(hours)} hours'
          AND NOT is_disabled
          AND lat BETWEEN ? AND ?
          AND lon BETWEEN ? AND ?
        ORDER BY fetched_at DESC
        LIMIT 200
        """,
        [lat_min, lat_max, lon_min, lon_max],
    ).df()


def free_bike_count_by_hour(
    conn: duckdb.DuckDBPyConnection,
    hours: int = 168,
) -> pd.DataFrame:
    """Citywide free-bike count by local hour of day, last N hours.

    For each hour-of-day, averages the count of distinct bikes seen in
    that hour across days in the window. Useful for "expected number of
    free bikes available at 8 AM."
    """
    local = _local_expr("fetched_at")
    return conn.execute(
        f"""
        WITH per_hour AS (
          SELECT
            DATE_TRUNC('hour', {local}) AS hour_bucket,
            CAST(EXTRACT(HOUR FROM {local}) AS INTEGER) AS hour_of_day,
            COUNT(DISTINCT bike_id) AS n_bikes
          FROM free_bike_status
          WHERE fetched_at > now() - INTERVAL '{int(hours)} hours'
            AND NOT is_disabled
          GROUP BY hour_bucket, hour_of_day
        )
        SELECT
          hour_of_day,
          AVG(n_bikes) AS avg_bikes,
          MIN(n_bikes) AS min_bikes,
          MAX(n_bikes) AS max_bikes,
          COUNT(*) AS n_hours_observed
        FROM per_hour
        GROUP BY hour_of_day
        ORDER BY hour_of_day
        """,
    ).df()


def station_trip_demand_profile(
    conn: duckdb.DuckDBPyConnection,
    station_id: str,
    *,
    anchor: datetime | None = None,
    lookback_days: int = 365,
    window_minutes: int = 120,
    slot_minutes: int = 10,
) -> pd.DataFrame:
    """Weighted trip-demand profile around the anchor time.

    Uses completed Divvy trip history rather than live station_status. The
    returned rows are expected departures/arrivals per `slot_minutes`, centered
    on the anchor local time. Seasonal weights emphasize a one-week band around
    the anchor day-of-year; calendar weights heavily favor the same weekday and
    then the same weekday/weekend class.
    """
    has_csv = _table_has_rows(conn, "station_trip_flows")
    has_inferred = _table_has_rows(conn, "station_inferred_flows")
    if not (has_csv or has_inferred):
        return pd.DataFrame()

    anchor = anchor or datetime.now(timezone.utc)
    if anchor.tzinfo is None:
        anchor_local = pd.Timestamp(anchor, tz="UTC").tz_convert(LOCAL_TZ)
    else:
        anchor_local = pd.Timestamp(anchor).tz_convert(LOCAL_TZ)
    anchor_date = anchor_local.date()
    anchor_dow = int(anchor_local.weekday())
    anchor_is_weekend = anchor_dow >= 5
    anchor_minute = int(anchor_local.hour * 60 + (anchor_local.minute // slot_minutes) * slot_minutes)
    offsets = list(range(-window_minutes, window_minutes + 1, slot_minutes))
    slot_df = pd.DataFrame([
        {
            "offset_minutes": offset,
            "minute_of_day": (anchor_minute + offset) % (24 * 60),
            "display_ts": (anchor_local.floor(f"{slot_minutes}min") + pd.Timedelta(minutes=offset)).to_pydatetime(),
        }
        for offset in offsets
    ])

    trip_ids = _station_trip_ids(conn, station_id)
    placeholders = ",".join(["?"] * len(trip_ids))
    start_local = anchor_local - pd.Timedelta(days=lookback_days + 2)
    start_utc = start_local.tz_convert("UTC").tz_localize(None).to_pydatetime()

    frames: list[pd.DataFrame] = []
    if has_csv:
        csv_rows = conn.execute(
            f"""
            SELECT bucket_start, departures, arrivals, ebike_departures, ebike_arrivals
            FROM station_trip_flows
            WHERE station_id IN ({placeholders})
              AND bucket_start >= ?
            """,
            [*trip_ids, start_utc],
        ).df()
        if not csv_rows.empty:
            csv_rows["source"] = "csv"
            frames.append(csv_rows)
    if has_inferred:
        inferred_rows = conn.execute(
            f"""
            SELECT bucket_start, departures, arrivals, ebike_departures, ebike_arrivals
            FROM station_inferred_flows
            WHERE station_id IN ({placeholders})
              AND bucket_start >= ?
            """,
            [*trip_ids, start_utc],
        ).df()
        if not inferred_rows.empty:
            inferred_rows["source"] = "inferred"
            frames.append(inferred_rows)

    if not frames:
        return pd.DataFrame()

    rows = pd.concat(frames, ignore_index=True)
    # Deduplicate: when both sources cover the same bucket, prefer CSV (authoritative).
    rows = rows.sort_values(
        ["bucket_start", "source"],
        key=lambda col: col.map({"csv": 0, "inferred": 1}) if col.name == "source" else col,
    ).drop_duplicates(subset=["bucket_start"], keep="first").reset_index(drop=True)
    if rows.empty:
        return pd.DataFrame()

    local_ts = pd.to_datetime(rows["bucket_start"], utc=True).dt.tz_convert(LOCAL_TZ)
    rows["local_date"] = local_ts.dt.date
    rows["minute_of_day"] = (local_ts.dt.hour * 60 + (local_ts.dt.minute // slot_minutes) * slot_minutes).astype(int)

    def _dominant_source(values: pd.Series) -> str:
        s = set(values.dropna().astype(str))
        if "csv" in s and "inferred" in s:
            return "mixed"
        if "csv" in s:
            return "csv"
        if "inferred" in s:
            return "inferred"
        return ""

    grouped = (
        rows.groupby(["local_date", "minute_of_day"], as_index=False)
        .agg(
            departures=("departures", "sum"),
            arrivals=("arrivals", "sum"),
            ebike_departures=("ebike_departures", "sum"),
            ebike_arrivals=("ebike_arrivals", "sum"),
            source=("source", _dominant_source),
        )
    )

    min_date = max(
        pd.Timestamp(grouped["local_date"].min()).date(),
        (anchor_local - pd.Timedelta(days=lookback_days)).date(),
    )
    max_date = min(pd.Timestamp(grouped["local_date"].max()).date(), anchor_date)
    if min_date > max_date:
        return pd.DataFrame()

    date_df = pd.DataFrame({"local_date": pd.date_range(min_date, max_date, freq="D").date})
    date_df["dow"] = pd.to_datetime(date_df["local_date"]).dt.weekday
    date_df["is_weekend"] = date_df["dow"] >= 5
    date_df["same_weekday"] = date_df["dow"] == anchor_dow
    date_df["same_day_type"] = date_df["is_weekend"] == anchor_is_weekend
    day_of_year = pd.to_datetime(date_df["local_date"]).dt.dayofyear.astype(int)
    anchor_doy = int(anchor_local.dayofyear)
    seasonal_delta = np.minimum((day_of_year - anchor_doy).abs(), 366 - (day_of_year - anchor_doy).abs())
    date_df["season_weight"] = np.exp(-0.5 * (seasonal_delta / 7.0) ** 2)
    date_df["calendar_weight"] = np.where(
        date_df["same_weekday"],
        4.0,
        np.where(date_df["same_day_type"], 1.6, 0.45),
    )
    date_df["weight"] = date_df["season_weight"] * date_df["calendar_weight"]

    grid = date_df.merge(slot_df[["offset_minutes", "minute_of_day", "display_ts"]], how="cross")
    grid = grid.merge(grouped, on=["local_date", "minute_of_day"], how="left")
    for column in ["departures", "arrivals", "ebike_departures", "ebike_arrivals"]:
        grid[column] = pd.to_numeric(grid[column], errors="coerce").fillna(0.0)
    if "source" not in grid.columns:
        grid["source"] = ""
    grid["source"] = grid["source"].fillna("")

    def weighted_average(group: pd.DataFrame, value: str, mask=None) -> float:
        if mask is not None:
            group = group[mask(group)]
        if group.empty:
            return 0.0
        denom = float(group["weight"].sum())
        if denom <= 0:
            return 0.0
        return float((group[value] * group["weight"]).sum() / denom)

    out_rows = []
    for (offset, minute, display_ts), group in grid.groupby(["offset_minutes", "minute_of_day", "display_ts"]):
        csv_mask = group["source"].isin(["csv", "mixed"])
        inferred_mask = group["source"].isin(["inferred", "mixed"])
        out_rows.append({
            "offset_minutes": int(offset),
            "minute_of_day": int(minute),
            "display_ts": display_ts,
            "time_label": f"{int(minute) // 60:02d}:{int(minute) % 60:02d}",
            "weighted_departures_10m": weighted_average(group, "departures"),
            "weighted_arrivals_10m": weighted_average(group, "arrivals"),
            "same_weekday_departures_10m": weighted_average(group, "departures", lambda g: g["same_weekday"]),
            "same_weekday_arrivals_10m": weighted_average(group, "arrivals", lambda g: g["same_weekday"]),
            "same_day_type_departures_10m": weighted_average(group, "departures", lambda g: g["same_day_type"]),
            "same_day_type_arrivals_10m": weighted_average(group, "arrivals", lambda g: g["same_day_type"]),
            "weighted_ebike_departures_10m": weighted_average(group, "ebike_departures"),
            "weighted_ebike_arrivals_10m": weighted_average(group, "ebike_arrivals"),
            "sample_days": int(group["local_date"].nunique()),
            "same_weekday_days": int(group.loc[group["same_weekday"], "local_date"].nunique()),
            "same_day_type_days": int(group.loc[group["same_day_type"], "local_date"].nunique()),
            "effective_weight_days": float(group["weight"].sum()),
            "csv_days": int(group.loc[csv_mask, "local_date"].nunique()),
            "inferred_days": int(group.loc[inferred_mask, "local_date"].nunique()),
        })
    out = pd.DataFrame(out_rows).sort_values("offset_minutes")
    out["anchor_time_label"] = anchor_local.strftime("%a %H:%M")
    out["anchor_day_type"] = "weekend" if anchor_is_weekend else "weekday"
    return out


def latest_status(
    conn: duckdb.DuckDBPyConnection,
    station_ids: list[str],
) -> pd.DataFrame:
    """Most recent station_status row per station_id.

    `num_classic_bikes` = total bikes available − ebikes (so empty docks aren't
    counted as classic). Returns empty frame if `station_ids` is empty.
    """
    if not station_ids:
        return pd.DataFrame()
    placeholders = ",".join(["?"] * len(station_ids))
    return conn.execute(
        f"""
        SELECT station_id,
               num_bikes_available,
               num_ebikes_available,
               (num_bikes_available - num_ebikes_available) AS num_classic_bikes,
               last_reported,
               is_renting
        FROM (
          SELECT *,
                 ROW_NUMBER() OVER (
                   PARTITION BY station_id ORDER BY last_reported DESC
                 ) AS rn
          FROM station_status
          WHERE station_id IN ({placeholders})
        )
        WHERE rn = 1
        """,
        station_ids,
    ).df()


def station_meta(conn: duckdb.DuckDBPyConnection, station_id: str) -> dict:
    row = conn.execute(
        """
        SELECT s.station_id, s.short_name, s.name, s.capacity, s.lat, s.lon,
               MIN(ss.last_reported) AS first_obs,
               MAX(ss.last_reported) AS last_obs,
               COUNT(ss.station_id) AS n_obs
        FROM stations s
        LEFT JOIN station_status ss USING (station_id)
        WHERE s.station_id = ?
        GROUP BY s.station_id, s.short_name, s.name, s.capacity, s.lat, s.lon
        """,
        [station_id],
    ).df()
    return row.iloc[0].to_dict() if len(row) else {}


def _window_filter(start: datetime | None, end: datetime | None) -> tuple[str, list]:
    clauses, params = [], []
    if start is not None:
        clauses.append("last_reported >= ?")
        params.append(start)
    if end is not None:
        clauses.append("last_reported < ?")
        params.append(end)
    where = (" AND " + " AND ".join(clauses)) if clauses else ""
    return where, params


def availability_heatmap(
    conn: duckdb.DuckDBPyConnection,
    station_id: str,
    start: datetime | None = None,
    end: datetime | None = None,
    require_renting: bool = True,
) -> pd.DataFrame:
    window, params = _window_filter(start, end)
    renting_clause = " AND is_renting" if require_renting else ""
    local = _local_expr("last_reported")
    sql = f"""
        SELECT
          CAST(EXTRACT(DOW FROM {local}) AS INTEGER) AS dow,
          CAST(EXTRACT(HOUR FROM {local}) AS INTEGER) AS hour,
          AVG(CASE WHEN num_bikes_available = 0 THEN 1.0 ELSE 0.0 END) AS p_empty,
          AVG(CASE WHEN num_docks_available = 0 THEN 1.0 ELSE 0.0 END) AS p_full,
          AVG(CASE WHEN num_ebikes_available >= 1 THEN 1.0 ELSE 0.0 END) AS p_ebike_available,
          AVG(num_bikes_available) AS mean_bikes,
          AVG(num_ebikes_available) AS mean_ebikes,
          AVG(num_bikes_available - num_ebikes_available) AS mean_classic,
          AVG(num_docks_available) AS mean_docks,
          COUNT(*) AS n
        FROM station_status
        WHERE station_id = ? {renting_clause} {window}
        GROUP BY dow, hour
        ORDER BY dow, hour
    """
    return conn.execute(sql, [station_id, *params]).df()


def range_by_hour(
    conn: duckdb.DuckDBPyConnection,
    station_id: str,
    start: datetime | None = None,
    end: datetime | None = None,
) -> pd.DataFrame:
    window, params = _window_filter(start, end)
    local = _local_expr("last_reported")
    sql = f"""
        SELECT
          CAST(EXTRACT(HOUR FROM {local}) AS INTEGER) AS hour,
          QUANTILE_CONT(num_bikes_available, 0.10) AS p10_bikes,
          QUANTILE_CONT(num_bikes_available, 0.25) AS p25_bikes,
          QUANTILE_CONT(num_bikes_available, 0.50) AS p50_bikes,
          QUANTILE_CONT(num_bikes_available, 0.75) AS p75_bikes,
          QUANTILE_CONT(num_bikes_available, 0.90) AS p90_bikes,
          QUANTILE_CONT(num_ebikes_available, 0.50) AS p50_ebikes,
          QUANTILE_CONT(num_bikes_available - num_ebikes_available, 0.50) AS p50_classic,
          AVG(num_bikes_available) AS mean_bikes,
          COUNT(*) AS n
        FROM station_status
        WHERE station_id = ? {window}
        GROUP BY hour
        ORDER BY hour
    """
    return conn.execute(sql, [station_id, *params]).df()


def churn_by_hour(
    conn: duckdb.DuckDBPyConnection,
    station_id: str,
    start: datetime | None = None,
    end: datetime | None = None,
) -> pd.DataFrame:
    window, params = _window_filter(start, end)
    local = _local_expr("last_reported")
    sql = f"""
        WITH ordered AS (
          SELECT
            station_id,
            last_reported,
            num_bikes_available,
            LAG(num_bikes_available) OVER (
              PARTITION BY station_id ORDER BY last_reported
            ) AS prev_bikes
          FROM station_status
          WHERE station_id = ? {window}
        ),
        diffs AS (
          SELECT
            CAST(EXTRACT(DOW FROM {_local_expr("last_reported")}) AS INTEGER) AS dow,
            CAST(EXTRACT(HOUR FROM {_local_expr("last_reported")}) AS INTEGER) AS hour,
            DATE_TRUNC('day', {_local_expr("last_reported")}) AS local_day,
            ABS(num_bikes_available - prev_bikes) AS delta
          FROM ordered
          WHERE prev_bikes IS NOT NULL
        )
        SELECT
          dow,
          hour,
          SUM(delta) AS total_delta,
          COUNT(DISTINCT local_day) AS n_days,
          SUM(delta) * 1.0 / NULLIF(COUNT(DISTINCT local_day), 0) AS rides_per_hour_est
        FROM diffs
        GROUP BY dow, hour
        ORDER BY dow, hour
    """
    return conn.execute(sql, [station_id, *params]).df()


def time_series(
    conn: duckdb.DuckDBPyConnection,
    station_id: str,
    start: datetime | None = None,
    end: datetime | None = None,
    resample_minutes: int | None = None,
) -> pd.DataFrame:
    window, params = _window_filter(start, end)
    if resample_minutes:
        sql = f"""
            SELECT
              TIME_BUCKET(INTERVAL '{resample_minutes} minutes', last_reported) AS bucket,
              AVG(num_bikes_available) AS num_bikes_available,
              AVG(num_ebikes_available) AS num_ebikes_available,
              AVG(num_bikes_available - num_ebikes_available) AS num_classic_bikes,
              AVG(num_docks_available) AS num_docks_available
            FROM station_status
            WHERE station_id = ? {window}
            GROUP BY bucket
            ORDER BY bucket
        """
        df = conn.execute(sql, [station_id, *params]).df()
        df = df.rename(columns={"bucket": "ts"})
    else:
        sql = f"""
            SELECT
              last_reported AS ts,
              num_bikes_available,
              num_ebikes_available,
              num_bikes_available - num_ebikes_available AS num_classic_bikes,
              num_docks_available
            FROM station_status
            WHERE station_id = ? {window}
            ORDER BY ts
        """
        df = conn.execute(sql, [station_id, *params]).df()
    return df


def ebike_summary(
    conn: duckdb.DuckDBPyConnection,
    station_id: str,
    start: datetime | None = None,
    end: datetime | None = None,
) -> dict:
    """Latest ebike count + min/median/mean/max + per-day-range stats over the window."""
    window, params = _window_filter(start, end)
    sql = f"""
        SELECT
          MIN(num_ebikes_available)                    AS min_eb,
          QUANTILE_CONT(num_ebikes_available, 0.5)     AS median_eb,
          AVG(num_ebikes_available)                    AS mean_eb,
          MAX(num_ebikes_available)                    AS max_eb,
          COUNT(*)                                     AS n_obs,
          ARG_MAX(num_ebikes_available, last_reported) AS current_eb,
          MAX(last_reported)                           AS as_of
        FROM station_status
        WHERE station_id = ? {window}
    """
    row = conn.execute(sql, [station_id, *params]).df()
    if row.empty or row.iloc[0]["n_obs"] == 0:
        return {}

    daily_sql = f"""
        SELECT AVG(daily_range) AS avg_daily_range,
               MIN(daily_range) AS min_daily_range,
               MAX(daily_range) AS max_daily_range,
               COUNT(*)         AS n_days
        FROM (
          SELECT MAX(num_ebikes_available) - MIN(num_ebikes_available) AS daily_range
          FROM station_status
          WHERE station_id = ? {window}
          GROUP BY DATE_TRUNC('day', {_local_expr("last_reported")})
          HAVING COUNT(*) >= 2
        )
    """
    dr = conn.execute(daily_sql, [station_id, *params]).df().iloc[0].to_dict()
    out = row.iloc[0].to_dict()
    out.update(dr)
    return out


def ebike_forecast(
    conn: duckdb.DuckDBPyConnection,
    station_id: str,
    current_value: int,
    start: datetime | None = None,
    end: datetime | None = None,
    lags: tuple[int, ...] = (5, 10, 15, 20),
    tolerance_min: int = 1,
) -> pd.DataFrame:
    """For each lag k in `lags` minutes, return empirical conditional and unconditional
    expectations of ebike count using historical (X(t), X(t+k)) pairs from this station.

    The conditional curve answers "given the station has `current_value` ebikes right now,
    how many do I expect in k minutes?" It decays toward the unconditional mean as k grows.

    Columns: lag_minutes, e_conditional, p25_conditional, p75_conditional, n_conditional,
             e_unconditional, n_total
    """
    window, params = _window_filter(start, end)
    max_lag = max(lags)
    when_clauses = " ".join(
        f"WHEN lag_min BETWEEN {lag - tolerance_min} AND {lag + tolerance_min} THEN {lag}"
        for lag in lags
    )
    sql = f"""
        WITH base AS (
          SELECT last_reported, num_ebikes_available
          FROM station_status
          WHERE station_id = ? {window}
        ),
        pairs AS (
          SELECT
            a.num_ebikes_available AS x_now,
            b.num_ebikes_available AS x_future,
            EXTRACT(EPOCH FROM (b.last_reported - a.last_reported)) / 60.0 AS lag_min
          FROM base a, base b
          WHERE b.last_reported > a.last_reported
            AND b.last_reported <= a.last_reported + INTERVAL '{max_lag + tolerance_min} minutes'
        )
        SELECT
          CAST((CASE {when_clauses} END) AS INTEGER) AS lag_minutes,
          x_now,
          x_future
        FROM pairs
        WHERE (CASE {when_clauses} END) IS NOT NULL
    """
    pairs = conn.execute(sql, [station_id, *params]).df()

    rows = []
    for lag in lags:
        sub = pairs[pairs["lag_minutes"] == lag]
        cond = sub[sub["x_now"] == current_value]
        rows.append({
            "lag_minutes": lag,
            "e_conditional": float(cond["x_future"].mean()) if len(cond) else None,
            "p25_conditional": float(cond["x_future"].quantile(0.25)) if len(cond) else None,
            "p75_conditional": float(cond["x_future"].quantile(0.75)) if len(cond) else None,
            "n_conditional": int(len(cond)),
            "e_unconditional": float(sub["x_future"].mean()) if len(sub) else None,
            "n_total": int(len(sub)),
        })
    return pd.DataFrame(rows)


def coverage_health(conn: duckdb.DuckDBPyConnection, hours: int = 24) -> pd.DataFrame:
    return conn.execute(
        f"""
        SELECT
          DATE_TRUNC('minute', fetched_at) AS minute,
          COUNT(*) AS rows_inserted
        FROM station_status
        WHERE fetched_at > now() - INTERVAL '{hours} hours'
        GROUP BY minute
        ORDER BY minute
        """
    ).df()


def default_window(days: int = 30) -> tuple[datetime, datetime]:
    end = datetime.now(timezone.utc).replace(tzinfo=None)
    start = end - timedelta(days=days)
    return start, end
