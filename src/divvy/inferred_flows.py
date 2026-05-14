"""Inferred trip flows from ``station_status`` count deltas.

Each 60-second poll of the GBFS ``station_status`` feed gives aggregate counts
per station (ebikes available, classic bikes available). Differences between
consecutive snapshots are interpreted as departures (count drop) or arrivals
(count rise). Aggregated into 10-minute UTC buckets, they fill the gap between
the most recent monthly trip CSV and "now" -- the public CSVs typically lag
1-3 months.

Large single-tick swings (>= REBALANCE_THRESHOLD net bikes moved at one
station in one tick) are treated as operator rebalancing/repair moves and
counted in ``rebalancing_events`` instead of flow counts.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone

import duckdb

from . import db, service_state


log = logging.getLogger("divvy.inferred_flows")


WATERMARK_KEY = "inferred_flows_processed_through"
BUCKET_MINUTES = 10
REBALANCE_THRESHOLD = 4
CHUNK_DAYS = 1
# Pull this far before the window for LAG context so the first delta in the
# window has a previous row to compare against. Plenty of buffer for sparse
# stations that don't report every minute.
LAG_LOOKBACK_HOURS = 2


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _floor_to_bucket(ts: datetime) -> datetime:
    minutes = (ts.minute // BUCKET_MINUTES) * BUCKET_MINUTES
    return ts.replace(minute=minutes, second=0, microsecond=0)


def compute_window(
    conn: duckdb.DuckDBPyConnection,
    since: datetime,
    until: datetime,
    *,
    rebalance_threshold: int = REBALANCE_THRESHOLD,
) -> int:
    """Compute inferred flows for ``station_status`` rows in ``[since, until)``.

    Returns the number of (station_id, bucket_start) rows upserted into
    ``station_inferred_flows``.
    """
    if since >= until:
        return 0

    # Always rewrite full buckets: align the start to the bucket containing
    # `since`. Pull station_status from further back (LAG_LOOKBACK_HOURS) so
    # the first delta in the bucket has a previous row to compare against,
    # but only emit deltas at or after the aligned bucket boundary.
    aligned_since = _floor_to_bucket(since)
    fetch_from = aligned_since - timedelta(hours=LAG_LOOKBACK_HOURS)

    rows = conn.execute(
        """
        WITH ordered AS (
          SELECT
            station_id,
            last_reported,
            COALESCE(num_ebikes_available, 0) AS ebikes,
            GREATEST(COALESCE(num_bikes_available, 0)
                     - COALESCE(num_ebikes_available, 0), 0) AS classic,
            LAG(COALESCE(num_ebikes_available, 0)) OVER w AS prev_ebikes,
            LAG(GREATEST(COALESCE(num_bikes_available, 0)
                         - COALESCE(num_ebikes_available, 0), 0)) OVER w AS prev_classic
          FROM station_status
          WHERE last_reported >= ? AND last_reported < ?
          WINDOW w AS (PARTITION BY station_id ORDER BY last_reported)
        )
        SELECT station_id, last_reported,
               (prev_ebikes - ebikes)   AS ebike_change,
               (prev_classic - classic) AS classic_change
        FROM ordered
        WHERE prev_ebikes IS NOT NULL
          AND last_reported >= ?
          AND (ebikes <> prev_ebikes OR classic <> prev_classic)
        """,
        [fetch_from, until, aligned_since],
    ).df()

    if rows.empty:
        return 0

    buckets: dict[tuple[str, datetime], dict[str, int]] = {}

    def _slot(station_id: str, bucket_start: datetime) -> dict[str, int]:
        key = (station_id, bucket_start)
        if key not in buckets:
            buckets[key] = {
                "ebike_dep": 0,
                "ebike_arr": 0,
                "classic_dep": 0,
                "classic_arr": 0,
                "observations": 0,
                "rebalancing": 0,
            }
        return buckets[key]

    for row in rows.itertuples(index=False):
        station_id = str(row.station_id)
        ts = row.last_reported
        if not isinstance(ts, datetime):
            # DuckDB may return a pandas.Timestamp -- coerce
            ts = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else datetime.fromisoformat(str(ts))
        bucket_start = _floor_to_bucket(ts)

        ebike_change = int(row.ebike_change)
        classic_change = int(row.classic_change)

        slot = _slot(station_id, bucket_start)
        slot["observations"] += 1

        if abs(ebike_change) + abs(classic_change) >= rebalance_threshold:
            slot["rebalancing"] += 1
            continue

        if ebike_change > 0:
            slot["ebike_dep"] += ebike_change
        elif ebike_change < 0:
            slot["ebike_arr"] += -ebike_change

        if classic_change > 0:
            slot["classic_dep"] += classic_change
        elif classic_change < 0:
            slot["classic_arr"] += -classic_change

    if not buckets:
        return 0

    payload = []
    computed_at = _utc_now()
    for (station_id, bucket_start), slot in buckets.items():
        departures = slot["ebike_dep"] + slot["classic_dep"]
        arrivals = slot["ebike_arr"] + slot["classic_arr"]
        payload.append(
            (
                station_id,
                bucket_start,
                departures,
                arrivals,
                slot["ebike_dep"],
                slot["ebike_arr"],
                slot["classic_dep"],
                slot["classic_arr"],
                slot["observations"],
                slot["rebalancing"],
                computed_at,
            )
        )

    conn.executemany(
        """
        INSERT INTO station_inferred_flows (
          station_id, bucket_start,
          departures, arrivals,
          ebike_departures, ebike_arrivals,
          classic_departures, classic_arrivals,
          observations, rebalancing_events, computed_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (station_id, bucket_start) DO UPDATE SET
          departures = excluded.departures,
          arrivals = excluded.arrivals,
          ebike_departures = excluded.ebike_departures,
          ebike_arrivals = excluded.ebike_arrivals,
          classic_departures = excluded.classic_departures,
          classic_arrivals = excluded.classic_arrivals,
          observations = excluded.observations,
          rebalancing_events = excluded.rebalancing_events,
          computed_at = excluded.computed_at
        """,
        payload,
    )

    return len(payload)


def _earliest_station_status(conn: duckdb.DuckDBPyConnection) -> datetime | None:
    row = conn.execute("SELECT MIN(last_reported) FROM station_status").fetchone()
    if not row or row[0] is None:
        return None
    ts = row[0]
    if not isinstance(ts, datetime):
        ts = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else datetime.fromisoformat(str(ts))
    return ts


def _latest_station_status(conn: duckdb.DuckDBPyConnection) -> datetime | None:
    row = conn.execute("SELECT MAX(last_reported) FROM station_status").fetchone()
    if not row or row[0] is None:
        return None
    ts = row[0]
    if not isinstance(ts, datetime):
        ts = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else datetime.fromisoformat(str(ts))
    return ts


def run(
    conn: duckdb.DuckDBPyConnection,
    *,
    mode: str = "incremental",
    chunk_days: int = CHUNK_DAYS,
) -> dict:
    """Backfill or incrementally extend ``station_inferred_flows``.

    ``mode='backfill'`` ignores the watermark and processes from the earliest
    ``station_status`` row. ``mode='incremental'`` resumes from the watermark.
    """
    db.init_schema(conn)
    latest = _latest_station_status(conn)
    if latest is None:
        return {"status": "no_data", "rows_upserted": 0, "windows": 0}

    if mode == "backfill":
        cursor = _earliest_station_status(conn)
    else:
        watermark = service_state.get_watermark(conn, WATERMARK_KEY)
        if watermark is None:
            cursor = _earliest_station_status(conn)
        else:
            # compute_window aligns its start to the bucket containing `since`
            # and pulls LAG context from further back, so re-processing from
            # the prior watermark cleanly rewrites the trailing partial bucket.
            cursor = watermark

    if cursor is None:
        return {"status": "no_data", "rows_upserted": 0, "windows": 0}

    total_rows = 0
    windows = 0
    chunk = timedelta(days=int(chunk_days))
    while cursor < latest:
        window_end = min(cursor + chunk, latest + timedelta(seconds=1))
        upserted = compute_window(conn, cursor, window_end)
        total_rows += upserted
        windows += 1
        log.info(
            "inferred_flows window %s -> %s upserted=%d",
            cursor.isoformat(),
            window_end.isoformat(),
            upserted,
        )
        cursor = window_end

    service_state.set_watermark(conn, WATERMARK_KEY, latest)
    return {
        "status": "ok",
        "rows_upserted": int(total_rows),
        "windows": int(windows),
        "mode": mode,
        "watermark": latest.isoformat(),
    }


def run_incremental(conn: duckdb.DuckDBPyConnection) -> dict:
    return run(conn, mode="incremental")


def run_backfill(conn: duckdb.DuckDBPyConnection, *, chunk_days: int = CHUNK_DAYS) -> dict:
    # Start clean -- backfill wipes prior state for that bike-status history
    conn.execute("DELETE FROM station_inferred_flows")
    conn.execute("DELETE FROM flow_processing_state WHERE key = ?", [WATERMARK_KEY])
    return run(conn, mode="backfill", chunk_days=chunk_days)


def _cli(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    parser = argparse.ArgumentParser(prog="python -m divvy.inferred_flows")
    parser.add_argument("mode", choices=["backfill", "incremental"], nargs="?", default="incremental")
    parser.add_argument("--chunk-days", type=int, default=CHUNK_DAYS)
    args = parser.parse_args(argv)

    with db.session(read_only=False) as conn:
        if args.mode == "backfill":
            result = run_backfill(conn, chunk_days=args.chunk_days)
        else:
            result = run_incremental(conn)
    print(result)
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
