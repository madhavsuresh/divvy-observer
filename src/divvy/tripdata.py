from __future__ import annotations

import argparse
import io
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import duckdb
import pandas as pd
import requests

from . import db

DIVVY_TRIPDATA_BASE_URL = "https://divvy-tripdata.s3.amazonaws.com"
LOCAL_TZ = "America/Chicago"
CHUNK_SIZE = 150_000
# Earliest year/month that uses the modern monthly ZIP naming + schema.
# Files before April 2020 use a different naming scheme (Divvy_Trips_YYYY_QX.zip)
# AND a different column schema (trip_id, starttime, ...) — those would need a
# separate normalize path. The modern schema is what every model in this repo
# expects, so we backfill from the first available month onward.
S3_MONTHLY_FORMAT_FROM = (2020, 4)
# Treat a month as "already loaded" if we have at least this many trips for it.
# Real months are >100k trips, so 1k is a generous floor that still catches
# mostly-empty partial loads from interrupted past runs.
MIN_TRIPS_FOR_LOADED_MONTH = 1000


@dataclass(frozen=True)
class TripIngestResult:
    month: str | None
    rows_inserted: int
    rows_seen: int
    source: str


@dataclass(frozen=True)
class FlowBuildResult:
    flow_rows: int
    route_rows: int


def month_url(year: int, month: int) -> str:
    return f"{DIVVY_TRIPDATA_BASE_URL}/{year}{month:02d}-divvy-tripdata.zip"


def completed_months(count: int, end: date | None = None) -> list[tuple[int, int]]:
    if count <= 0:
        return []
    anchor = end or date.today()
    first_of_month = date(anchor.year, anchor.month, 1)
    current = first_of_month - timedelta(days=1)
    months: list[tuple[int, int]] = []
    for _ in range(count):
        months.append((current.year, current.month))
        current = date(current.year, current.month, 1) - timedelta(days=1)
    return list(reversed(months))


def _to_utc_naive(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, errors="coerce")
    if getattr(ts.dt, "tz", None) is None:
        localized = ts.dt.tz_localize(LOCAL_TZ, nonexistent="shift_forward", ambiguous="NaT")
    else:
        localized = ts.dt.tz_convert(LOCAL_TZ)
    return localized.dt.tz_convert("UTC").dt.tz_localize(None)


def _clean_station_id(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.strip()
    return cleaned.mask(cleaned.isin(["", "nan", "None", "<NA>"]))


def normalize_trip_frame(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = frame.rename(
        columns={
            "start_lng": "start_lon",
            "end_lng": "end_lon",
        }
    ).copy()
    required = [
        "ride_id",
        "rideable_type",
        "started_at",
        "ended_at",
        "start_station_id",
        "start_station_name",
        "end_station_id",
        "end_station_name",
        "start_lat",
        "start_lon",
        "end_lat",
        "end_lon",
        "member_casual",
    ]
    for column in required:
        if column not in renamed.columns:
            renamed[column] = pd.NA

    out = renamed[required].copy()
    out["ride_id"] = out["ride_id"].astype("string").str.strip()
    out["rideable_type"] = out["rideable_type"].astype("string")
    out["started_at"] = _to_utc_naive(out["started_at"])
    out["ended_at"] = _to_utc_naive(out["ended_at"])
    out["start_station_id"] = _clean_station_id(out["start_station_id"])
    out["end_station_id"] = _clean_station_id(out["end_station_id"])
    out["start_station_name"] = out["start_station_name"].astype("string")
    out["end_station_name"] = out["end_station_name"].astype("string")
    out["member_casual"] = out["member_casual"].astype("string")
    for column in ["start_lat", "start_lon", "end_lat", "end_lon"]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["duration_minutes"] = (
        (out["ended_at"] - out["started_at"]).dt.total_seconds() / 60.0
    )
    out = out.dropna(subset=["ride_id", "started_at", "ended_at"])
    out = out[(out["duration_minutes"] >= 1.0) & (out["duration_minutes"] <= 240.0)]
    return out


def insert_trips(conn: duckdb.DuckDBPyConnection, trips: pd.DataFrame) -> int:
    if trips.empty:
        return 0
    before = conn.execute("SELECT COUNT(*) FROM divvy_trips").fetchone()[0]
    conn.register("_trip_ingest", trips)
    try:
        conn.execute(
            """
            INSERT OR IGNORE INTO divvy_trips (
              ride_id, rideable_type, started_at, ended_at,
              start_station_id, start_station_name, end_station_id, end_station_name,
              start_lat, start_lon, end_lat, end_lon, member_casual, duration_minutes
            )
            SELECT
              ride_id, rideable_type, started_at, ended_at,
              start_station_id, start_station_name, end_station_id, end_station_name,
              start_lat, start_lon, end_lat, end_lon, member_casual, duration_minutes
            FROM _trip_ingest
            """
        )
    finally:
        conn.unregister("_trip_ingest")
    after = conn.execute("SELECT COUNT(*) FROM divvy_trips").fetchone()[0]
    return int(after - before)


def _csv_members(zf: zipfile.ZipFile) -> list[str]:
    return [name for name in zf.namelist() if name.lower().endswith(".csv") and not name.startswith("__MACOSX/")]


def ingest_zip(
    conn: duckdb.DuckDBPyConnection,
    source: str | Path | bytes,
    month: str | None = None,
) -> TripIngestResult:
    if isinstance(source, bytes):
        payload = io.BytesIO(source)
        source_label = month or "downloaded-zip"
    else:
        payload = Path(source)
        source_label = str(payload)

    inserted = 0
    seen = 0
    with zipfile.ZipFile(payload) as zf:
        members = _csv_members(zf)
        if not members:
            raise ValueError(f"No CSV files found in {source_label}")
        for member in members:
            with zf.open(member) as handle:
                for chunk in pd.read_csv(handle, chunksize=CHUNK_SIZE, low_memory=False):
                    normalized = normalize_trip_frame(chunk)
                    seen += len(chunk)
                    inserted += insert_trips(conn, normalized)
    return TripIngestResult(month=month, rows_inserted=inserted, rows_seen=seen, source=source_label)


def download_month(
    conn: duckdb.DuckDBPyConnection,
    year: int,
    month: int,
    timeout_seconds: int = 90,
) -> TripIngestResult:
    label = f"{year}-{month:02d}"
    response = requests.get(month_url(year, month), timeout=timeout_seconds)
    response.raise_for_status()
    return ingest_zip(conn, response.content, month=label)


def rebuild_flow_tables(conn: duckdb.DuckDBPyConnection) -> FlowBuildResult:
    conn.execute("DELETE FROM station_trip_flows")
    conn.execute("DELETE FROM station_trip_routes")
    conn.execute(
        """
        INSERT INTO station_trip_flows (
          station_id, bucket_start, departures, arrivals, ebike_departures, ebike_arrivals
        )
        SELECT
          station_id,
          bucket_start,
          CAST(SUM(departures) AS INTEGER) AS departures,
          CAST(SUM(arrivals) AS INTEGER) AS arrivals,
          CAST(SUM(ebike_departures) AS INTEGER) AS ebike_departures,
          CAST(SUM(ebike_arrivals) AS INTEGER) AS ebike_arrivals
        FROM (
          SELECT
            start_station_id AS station_id,
            DATE_TRUNC('minute', started_at) AS bucket_start,
            1 AS departures,
            0 AS arrivals,
            CASE WHEN LOWER(COALESCE(rideable_type, '')) LIKE '%electric%' THEN 1 ELSE 0 END
              AS ebike_departures,
            0 AS ebike_arrivals
          FROM divvy_trips
          WHERE start_station_id IS NOT NULL
          UNION ALL
          SELECT
            end_station_id AS station_id,
            DATE_TRUNC('minute', ended_at) AS bucket_start,
            0 AS departures,
            1 AS arrivals,
            0 AS ebike_departures,
            CASE WHEN LOWER(COALESCE(rideable_type, '')) LIKE '%electric%' THEN 1 ELSE 0 END
              AS ebike_arrivals
          FROM divvy_trips
          WHERE end_station_id IS NOT NULL
        )
        GROUP BY station_id, bucket_start
        """
    )
    conn.execute(
        f"""
        INSERT INTO station_trip_routes (
          start_station_id, end_station_id, local_hour, dow, trips, ebike_trips,
          avg_duration_minutes, median_duration_minutes
        )
        WITH routed AS (
          SELECT
            start_station_id,
            end_station_id,
            CAST(EXTRACT(HOUR FROM (
              CAST(ended_at AS TIMESTAMP) AT TIME ZONE 'UTC' AT TIME ZONE '{LOCAL_TZ}'
            )) AS INTEGER) AS local_hour,
            CAST((
              CAST(EXTRACT(DOW FROM (
                CAST(ended_at AS TIMESTAMP) AT TIME ZONE 'UTC' AT TIME ZONE '{LOCAL_TZ}'
              )) AS INTEGER) + 6
            ) % 7 AS INTEGER) AS dow,
            CASE WHEN LOWER(COALESCE(rideable_type, '')) LIKE '%electric%' THEN 1 ELSE 0 END
              AS is_ebike,
            duration_minutes
          FROM divvy_trips
          WHERE start_station_id IS NOT NULL
            AND end_station_id IS NOT NULL
            AND start_station_id <> end_station_id
            AND duration_minutes BETWEEN 1.0 AND 240.0
        )
        SELECT
          start_station_id,
          end_station_id,
          local_hour,
          dow,
          CAST(COUNT(*) AS INTEGER) AS trips,
          CAST(SUM(is_ebike) AS INTEGER) AS ebike_trips,
          AVG(duration_minutes) AS avg_duration_minutes,
          MEDIAN(duration_minutes) AS median_duration_minutes
        FROM routed
        GROUP BY start_station_id, end_station_id, local_hour, dow
        """
    )
    flow_rows = conn.execute("SELECT COUNT(*) FROM station_trip_flows").fetchone()[0]
    route_rows = conn.execute("SELECT COUNT(*) FROM station_trip_routes").fetchone()[0]
    return FlowBuildResult(flow_rows=int(flow_rows), route_rows=int(route_rows))


def sync_recent_months(conn: duckdb.DuckDBPyConnection, months: int = 3) -> list[TripIngestResult]:
    results = [download_month(conn, year, month) for year, month in completed_months(months)]
    rebuild_flow_tables(conn)
    return results


def all_completed_months_since(
    start: tuple[int, int] = S3_MONTHLY_FORMAT_FROM,
    end: date | None = None,
) -> list[tuple[int, int]]:
    """Every (year, month) from ``start`` through the most recently completed month."""
    anchor = end or date.today()
    end_first = date(anchor.year, anchor.month, 1) - timedelta(days=1)
    months: list[tuple[int, int]] = []
    cursor = date(start[0], start[1], 1)
    while cursor <= end_first:
        months.append((cursor.year, cursor.month))
        if cursor.month == 12:
            cursor = date(cursor.year + 1, 1, 1)
        else:
            cursor = date(cursor.year, cursor.month + 1, 1)
    return months


def months_already_loaded(
    conn: duckdb.DuckDBPyConnection, *, min_trips: int = MIN_TRIPS_FOR_LOADED_MONTH
) -> set[tuple[int, int]]:
    """Months in ``divvy_trips`` with at least ``min_trips`` rows.

    Used to skip re-downloading months we've already ingested. We use the
    started_at field (UTC); month boundaries can shift a few hours of trips
    into the neighboring month vs. local time, but for a "have we loaded
    this?" gate that's negligible.
    """
    rows = conn.execute(
        """
        SELECT
          CAST(EXTRACT(YEAR FROM started_at) AS INTEGER) AS y,
          CAST(EXTRACT(MONTH FROM started_at) AS INTEGER) AS m,
          COUNT(*) AS n
        FROM divvy_trips
        WHERE started_at IS NOT NULL
        GROUP BY y, m
        HAVING COUNT(*) >= ?
        """,
        [int(min_trips)],
    ).fetchall()
    return {(int(y), int(m)) for y, m, _ in rows}


def sync_range(
    conn: duckdb.DuckDBPyConnection,
    *,
    start: tuple[int, int] = S3_MONTHLY_FORMAT_FROM,
    end: date | None = None,
    skip_existing: bool = True,
    rebuild_flows: bool = True,
    on_progress=None,
) -> list[TripIngestResult]:
    """Backfill all monthly archives in [start, latest completed month].

    - Skips months that already have >= MIN_TRIPS_FOR_LOADED_MONTH trips loaded
      (so re-running is cheap and idempotent — only new months hit S3).
    - Skips months whose ZIP returns 404 (S3 hasn't published it yet).
    - Defers ``rebuild_flow_tables`` until the end so we don't pay the
      O(divvy_trips) cost N times during a backfill.

    Returns the per-month TripIngestResult for every month we actually
    attempted to download (skipped months are not in the list).
    """
    months = all_completed_months_since(start, end=end)
    if skip_existing:
        loaded = months_already_loaded(conn)
        months = [m for m in months if m not in loaded]
    results: list[TripIngestResult] = []
    for year, month in months:
        try:
            result = download_month(conn, year, month)
        except requests.HTTPError as exc:
            status = getattr(exc.response, "status_code", None)
            if status == 404:
                # S3 doesn't have this month yet (recent month not published,
                # or older month that pre-dates the modern naming scheme).
                if on_progress is not None:
                    on_progress(year, month, None, "missing")
                continue
            raise
        results.append(result)
        if on_progress is not None:
            on_progress(year, month, result, "ok")
    if rebuild_flows and results:
        rebuild_flow_tables(conn)
    return results


def _format_results(results: Iterable[TripIngestResult]) -> str:
    lines = []
    for result in results:
        label = result.month or result.source
        lines.append(f"{label}: inserted {result.rows_inserted:,} of {result.rows_seen:,} rows")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest Divvy historical trip data.")
    sub = parser.add_subparsers(dest="command", required=True)

    sync = sub.add_parser("sync", help="Download and ingest recent completed monthly trip archives.")
    sync.add_argument("--months", type=int, default=3)

    sync_all = sub.add_parser(
        "sync-all",
        help="Backfill every monthly archive from start month through the latest completed month.",
    )
    sync_all.add_argument("--start-year", type=int, default=S3_MONTHLY_FORMAT_FROM[0])
    sync_all.add_argument("--start-month", type=int, default=S3_MONTHLY_FORMAT_FROM[1])
    sync_all.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        default=True,
        help="Re-download months that are already loaded (default: skip).",
    )
    sync_all.add_argument(
        "--no-rebuild-flows",
        dest="rebuild_flows",
        action="store_false",
        default=True,
        help="Skip the post-download rebuild of station_trip_flows / station_trip_routes.",
    )

    ingest = sub.add_parser("ingest-file", help="Ingest a local Divvy trip-data ZIP file.")
    ingest.add_argument("path", type=Path)
    ingest.add_argument("--month")

    sub.add_parser("rebuild-features", help="Rebuild station flow and route aggregate tables.")

    args = parser.parse_args(argv)
    with db.session() as conn:
        db.init_schema(conn)
        if args.command == "sync":
            results = sync_recent_months(conn, months=args.months)
            print(_format_results(results))
        elif args.command == "sync-all":
            def _progress(year, month, result, status):
                if status == "ok" and result is not None:
                    print(
                        f"{year}-{month:02d}: inserted {result.rows_inserted:,} of "
                        f"{result.rows_seen:,} rows",
                        flush=True,
                    )
                elif status == "missing":
                    print(f"{year}-{month:02d}: not on S3 (skipped)", flush=True)
            results = sync_range(
                conn,
                start=(int(args.start_year), int(args.start_month)),
                skip_existing=bool(args.skip_existing),
                rebuild_flows=bool(args.rebuild_flows),
                on_progress=_progress,
            )
            total_inserted = sum(r.rows_inserted for r in results)
            total_seen = sum(r.rows_seen for r in results)
            print(
                f"\nDONE: {len(results)} months ingested, "
                f"{total_inserted:,} rows inserted of {total_seen:,} seen."
            )
        elif args.command == "ingest-file":
            result = ingest_zip(conn, args.path, month=args.month)
            flow = rebuild_flow_tables(conn)
            print(_format_results([result]))
            print(f"flow rows: {flow.flow_rows:,}; route rows: {flow.route_rows:,}")
        elif args.command == "rebuild-features":
            flow = rebuild_flow_tables(conn)
            print(f"flow rows: {flow.flow_rows:,}; route rows: {flow.route_rows:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
