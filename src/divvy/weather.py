from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import duckdb
import pandas as pd
import requests

from . import db

CHICAGO_LAT = 41.8781
CHICAGO_LON = -87.6298
LOCAL_TZ = "America/Chicago"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "apparent_temperature",
    "precipitation",
    "rain",
    "snowfall",
    "snow_depth",
    "cloud_cover",
    "wind_speed_10m",
    "wind_gusts_10m",
    "wind_direction_10m",
    "weather_code",
]


def _to_utc_naive(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, errors="coerce")
    if getattr(ts.dt, "tz", None) is None:
        localized = ts.dt.tz_localize(LOCAL_TZ, nonexistent="shift_forward", ambiguous="NaT")
    else:
        localized = ts.dt.tz_convert(LOCAL_TZ)
    return localized.dt.tz_convert("UTC").dt.tz_localize(None)


def normalize_weather_payload(payload: dict, source: str = "open-meteo") -> pd.DataFrame:
    hourly = payload.get("hourly") or {}
    if "time" not in hourly:
        return pd.DataFrame(columns=["observed_at", *HOURLY_VARIABLES, "source", "fetched_at"])

    out = pd.DataFrame({"observed_at": _to_utc_naive(pd.Series(hourly["time"]))})
    for variable in HOURLY_VARIABLES:
        out[variable] = pd.to_numeric(pd.Series(hourly.get(variable, [pd.NA] * len(out))), errors="coerce")
    out["weather_code"] = out["weather_code"].astype("Int64")
    out["source"] = source
    out["fetched_at"] = datetime.now(timezone.utc).replace(tzinfo=None)
    return dedupe_weather_rows(out.dropna(subset=["observed_at"]))


def dedupe_weather_rows(weather_rows: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicate hours caused by DST/local-time conversion."""
    if weather_rows.empty:
        return weather_rows
    rows = weather_rows.copy()
    rows["observed_at"] = pd.to_datetime(rows["observed_at"], errors="coerce")
    rows = rows.dropna(subset=["observed_at"])
    for variable in HOURLY_VARIABLES:
        if variable not in rows.columns:
            rows[variable] = pd.NA
    if "source" not in rows.columns:
        rows["source"] = "unknown"
    if "fetched_at" not in rows.columns:
        rows["fetched_at"] = datetime.now(timezone.utc).replace(tzinfo=None)

    agg = {variable: "mean" for variable in HOURLY_VARIABLES if variable != "weather_code"}
    agg["weather_code"] = "last"
    agg["source"] = "last"
    agg["fetched_at"] = "max"
    out = rows.groupby("observed_at", as_index=False).agg(agg)
    out["weather_code"] = pd.to_numeric(out["weather_code"], errors="coerce").astype("Int64")
    return out


def upsert_weather(conn: duckdb.DuckDBPyConnection, weather_rows: pd.DataFrame) -> int:
    weather_rows = dedupe_weather_rows(weather_rows)
    if weather_rows.empty:
        return 0
    before = conn.execute("SELECT COUNT(*) FROM weather_hourly").fetchone()[0]
    conn.register("_weather_ingest", weather_rows)
    try:
        conn.execute(
            """
            DELETE FROM weather_hourly
            WHERE observed_at IN (SELECT observed_at FROM _weather_ingest)
            """
        )
        conn.execute(
            """
            INSERT INTO weather_hourly (
              observed_at, source, temperature_2m, relative_humidity_2m,
              apparent_temperature, precipitation, rain, snowfall, snow_depth,
              cloud_cover, wind_speed_10m, wind_gusts_10m, wind_direction_10m,
              weather_code, fetched_at
            )
            SELECT
              observed_at, source, temperature_2m, relative_humidity_2m,
              apparent_temperature, precipitation, rain, snowfall, snow_depth,
              cloud_cover, wind_speed_10m, wind_gusts_10m, wind_direction_10m,
              weather_code, fetched_at
            FROM _weather_ingest
            """
        )
    finally:
        conn.unregister("_weather_ingest")
    after = conn.execute("SELECT COUNT(*) FROM weather_hourly").fetchone()[0]
    return int(max(0, after - before))


def _request_open_meteo(url: str, params: dict, timeout_seconds: int = 45) -> pd.DataFrame:
    response = requests.get(url, params=params, timeout=timeout_seconds)
    response.raise_for_status()
    return normalize_weather_payload(response.json())


def fetch_historical_weather(
    start: date,
    end: date,
    lat: float = CHICAGO_LAT,
    lon: float = CHICAGO_LON,
) -> pd.DataFrame:
    return _request_open_meteo(
        OPEN_METEO_ARCHIVE_URL,
        {
            "latitude": lat,
            "longitude": lon,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "hourly": ",".join(HOURLY_VARIABLES),
            "timezone": LOCAL_TZ,
            "wind_speed_unit": "kmh",
            "precipitation_unit": "mm",
        },
    )


def fetch_forecast_weather(
    days: int = 3,
    lat: float = CHICAGO_LAT,
    lon: float = CHICAGO_LON,
) -> pd.DataFrame:
    return _request_open_meteo(
        OPEN_METEO_FORECAST_URL,
        {
            "latitude": lat,
            "longitude": lon,
            "forecast_days": max(1, min(days, 16)),
            "hourly": ",".join(HOURLY_VARIABLES),
            "timezone": LOCAL_TZ,
            "wind_speed_unit": "kmh",
            "precipitation_unit": "mm",
        },
    )


def sync_recent_history(conn: duckdb.DuckDBPyConnection, days: int = 90) -> int:
    yesterday = date.today() - timedelta(days=1)
    start = yesterday - timedelta(days=max(1, days - 1))
    rows = fetch_historical_weather(start, yesterday)
    return upsert_weather(conn, rows)


def sync_forecast(conn: duckdb.DuckDBPyConnection, days: int = 3) -> int:
    rows = fetch_forecast_weather(days=days)
    return upsert_weather(conn, rows)


# ---------------------------------------------------------------------------
# Forecast snapshots — capture what the forecast SAID at each tick so model
# replay doesn't get to peek at the actual outcome via weather_hourly.
# ---------------------------------------------------------------------------

FORECAST_HOURLY_VARIABLES = [
    *HOURLY_VARIABLES,
    "precipitation_probability",
]

NOWCAST_MINUTELY_VARIABLES = [
    "precipitation",
    "rain",
    "snowfall",
]


def _open_meteo_get(url: str, params: dict, timeout_seconds: int = 30) -> dict:
    resp = requests.get(url, params=params, timeout=timeout_seconds)
    resp.raise_for_status()
    return resp.json()


def poll_forecast_snapshot(
    conn: duckdb.DuckDBPyConnection,
    hours_ahead: int | None = None,
    lat: float = CHICAGO_LAT,
    lon: float = CHICAGO_LON,
) -> int:
    """Capture the current forecast as a snapshot, keyed by (snapshot_at, for_at).

    Required for honest replay: weather_hourly stores observed values, which
    leak future state into a model that uses 'next-hour weather as feature.'
    Each row here is 'as of snapshot_at, the model expected for_at to look
    like this' — exactly what a live model would have seen.
    """
    from . import config as _config
    if hours_ahead is None:
        hours_ahead = _config.WEATHER_FORECAST_HOURS_AHEAD
    forecast_days = max(1, (hours_ahead + 23) // 24)
    payload = _open_meteo_get(
        OPEN_METEO_FORECAST_URL,
        {
            "latitude": lat,
            "longitude": lon,
            "forecast_days": min(forecast_days, 16),
            "hourly": ",".join(FORECAST_HOURLY_VARIABLES),
            "timezone": LOCAL_TZ,
            "wind_speed_unit": "kmh",
            "precipitation_unit": "mm",
        },
    )
    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []
    if not times:
        return 0

    snapshot_at = datetime.now(timezone.utc).replace(tzinfo=None)
    forecast_for = _to_utc_naive(pd.Series(times))

    frame = pd.DataFrame({"forecast_for_at": forecast_for})
    for var in FORECAST_HOURLY_VARIABLES:
        frame[var] = pd.to_numeric(
            pd.Series(hourly.get(var, [pd.NA] * len(frame))),
            errors="coerce",
        )
    frame = frame.dropna(subset=["forecast_for_at"])
    # Keep only forecasts inside our intended horizon (avoid storing the next
    # 16 days when the user asked for 24h).
    cutoff = snapshot_at + pd.Timedelta(hours=hours_ahead)
    frame = frame[(frame["forecast_for_at"] >= snapshot_at - pd.Timedelta(hours=1)) & (frame["forecast_for_at"] <= cutoff)]
    if frame.empty:
        return 0
    frame["horizon_minutes"] = (
        (frame["forecast_for_at"] - snapshot_at).dt.total_seconds() / 60.0
    ).round().astype("Int64")
    frame["snapshot_at"] = snapshot_at
    frame["source"] = "open-meteo"
    frame["weather_code"] = pd.to_numeric(frame["weather_code"], errors="coerce").astype("Int64")

    conn.register("_forecast_snap", frame)
    try:
        conn.execute(
            """
            INSERT INTO weather_forecast_snapshots
              (snapshot_at, forecast_for_at, horizon_minutes,
               temperature_2m, apparent_temperature, relative_humidity_2m,
               precipitation, precipitation_probability, rain, snowfall,
               cloud_cover, wind_speed_10m, wind_gusts_10m,
               wind_direction_10m, weather_code, source)
            SELECT
              snapshot_at, forecast_for_at, horizon_minutes,
              temperature_2m, apparent_temperature, relative_humidity_2m,
              precipitation, precipitation_probability, rain, snowfall,
              cloud_cover, wind_speed_10m, wind_gusts_10m,
              wind_direction_10m, weather_code, source
            FROM _forecast_snap
            ON CONFLICT (snapshot_at, forecast_for_at) DO NOTHING
            """
        )
    finally:
        conn.unregister("_forecast_snap")
    return int(len(frame))


def poll_nowcast(
    conn: duckdb.DuckDBPyConnection,
    lat: float = CHICAGO_LAT,
    lon: float = CHICAGO_LON,
) -> int:
    """Pull Open-Meteo minutely_15 precipitation forecast — next ~1 hour.

    This is the only signal that catches 'will it rain in the next 20 minutes',
    which the hourly grid blurs out. Stored separately so a re-poll just
    overwrites the same observed_at rows with the most recent estimate.
    """
    payload = _open_meteo_get(
        OPEN_METEO_FORECAST_URL,
        {
            "latitude": lat,
            "longitude": lon,
            "forecast_days": 1,
            "minutely_15": ",".join(NOWCAST_MINUTELY_VARIABLES),
            "timezone": LOCAL_TZ,
            "precipitation_unit": "mm",
        },
    )
    minutely = payload.get("minutely_15") or {}
    times = minutely.get("time") or []
    if not times:
        return 0

    fetched_at = datetime.now(timezone.utc).replace(tzinfo=None)
    observed_at = _to_utc_naive(pd.Series(times))

    frame = pd.DataFrame({"observed_at": observed_at})
    for var in NOWCAST_MINUTELY_VARIABLES:
        frame[var] = pd.to_numeric(
            pd.Series(minutely.get(var, [pd.NA] * len(frame))),
            errors="coerce",
        )
    frame = frame.dropna(subset=["observed_at"])
    if frame.empty:
        return 0
    # Keep only the next ~75 minutes; the rest is the hourly grid in disguise.
    cutoff = fetched_at + pd.Timedelta(minutes=75)
    frame = frame[frame["observed_at"] <= cutoff]
    if frame.empty:
        return 0
    frame["source"] = "open-meteo-minutely_15"
    frame["fetched_at"] = fetched_at

    conn.register("_nowcast", frame)
    try:
        conn.execute(
            """
            DELETE FROM weather_nowcast
            WHERE observed_at IN (SELECT observed_at FROM _nowcast)
            """
        )
        conn.execute(
            """
            INSERT INTO weather_nowcast (observed_at, source, precipitation, rain, snowfall, fetched_at)
            SELECT observed_at, source, precipitation, rain, snowfall, fetched_at FROM _nowcast
            """
        )
    finally:
        conn.unregister("_nowcast")
    return int(len(frame))


def ingest_csv(conn: duckdb.DuckDBPyConnection, path: Path) -> int:
    frame = pd.read_csv(path)
    if "observed_at" in frame.columns:
        frame["observed_at"] = _to_utc_naive(frame["observed_at"])
    elif "time" in frame.columns:
        frame["observed_at"] = _to_utc_naive(frame["time"])
    else:
        raise ValueError("Weather CSV needs an observed_at or time column")
    for variable in HOURLY_VARIABLES:
        if variable not in frame.columns:
            frame[variable] = pd.NA
    frame["source"] = frame.get("source", "csv")
    frame["fetched_at"] = datetime.now(timezone.utc).replace(tzinfo=None)
    return upsert_weather(
        conn,
        dedupe_weather_rows(frame[["observed_at", *HOURLY_VARIABLES, "source", "fetched_at"]]),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest Chicago weather features for Divvy models.")
    sub = parser.add_subparsers(dest="command", required=True)

    recent = sub.add_parser("sync-recent", help="Fetch recent historical hourly weather.")
    recent.add_argument("--days", type=int, default=90)

    history = sub.add_parser("sync-history", help="Fetch a specific historical range.")
    history.add_argument("--start", required=True)
    history.add_argument("--end", required=True)

    forecast = sub.add_parser("sync-forecast", help="Fetch near-future hourly weather forecasts.")
    forecast.add_argument("--days", type=int, default=3)

    csv = sub.add_parser("ingest-csv", help="Load hourly weather features from CSV.")
    csv.add_argument("path", type=Path)

    args = parser.parse_args(argv)
    with db.session() as conn:
        db.init_schema(conn)
        if args.command == "sync-recent":
            inserted = sync_recent_history(conn, days=args.days)
        elif args.command == "sync-history":
            rows = fetch_historical_weather(
                date.fromisoformat(args.start),
                date.fromisoformat(args.end),
            )
            inserted = upsert_weather(conn, rows)
        elif args.command == "sync-forecast":
            inserted = sync_forecast(conn, days=args.days)
        elif args.command == "ingest-csv":
            inserted = ingest_csv(conn, args.path)
        print(f"weather rows inserted or refreshed: {inserted:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
