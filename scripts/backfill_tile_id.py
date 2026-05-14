"""One-shot backfill of free_bike_status.tile_id for existing rows.

The poller writes tile_id on new inserts; this script fills it in for every
historical row. DuckDB has no H3 UDF so the computation happens in Python.

Approach: open the live DB read-write, process rows in week-sized batches
ordered by fetched_at. For each batch, pull (bike_id, fetched_at, lat, lon),
compute tile_id in Python, then batch-update via a temp table join. CHECKPOINT
after every batch so progress survives interruption. Idempotent — re-running
picks up where it left off via the ``WHERE tile_id IS NULL`` predicate.

Run during a quiet window (the collector tolerates short pauses; stopping it
beforehand is safest):

    uv run divvy stop
    uv run python scripts/backfill_tile_id.py
    uv run divvy start

The script can also be run alongside a live collector — the only risk is
slightly increased lock contention during the UPDATE phase.
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta

import duckdb

from divvy import config, tile


BATCH_DAYS_DEFAULT = 7
SUB_BATCH_ROWS = 50_000


def _backfill_window(conn: duckdb.DuckDBPyConnection, start: datetime, end: datetime) -> int:
    rows = conn.execute(
        """
        SELECT bike_id, fetched_at, lat, lon
        FROM free_bike_status
        WHERE tile_id IS NULL
          AND fetched_at >= ? AND fetched_at < ?
        """,
        [start, end],
    ).fetchall()
    if not rows:
        return 0

    total = 0
    for offset in range(0, len(rows), SUB_BATCH_ROWS):
        chunk = rows[offset : offset + SUB_BATCH_ROWS]
        updates = [
            (bike_id, fetched_at, tile.tile_id_for(lat, lon))
            for bike_id, fetched_at, lat, lon in chunk
        ]
        conn.execute("CREATE TEMP TABLE _tile_backfill (bike_id TEXT, fetched_at TIMESTAMP, tile_id TEXT)")
        conn.executemany("INSERT INTO _tile_backfill VALUES (?, ?, ?)", updates)
        conn.execute(
            """
            UPDATE free_bike_status
            SET tile_id = bt.tile_id
            FROM _tile_backfill bt
            WHERE free_bike_status.bike_id = bt.bike_id
              AND free_bike_status.fetched_at = bt.fetched_at
              AND free_bike_status.tile_id IS NULL
            """
        )
        conn.execute("DROP TABLE _tile_backfill")
        total += len(chunk)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(config.DB_PATH), help="DuckDB path")
    parser.add_argument("--batch-days", type=int, default=BATCH_DAYS_DEFAULT)
    parser.add_argument(
        "--resume-from",
        type=lambda s: datetime.fromisoformat(s),
        default=None,
        help="ISO timestamp to start from (defaults to MIN(fetched_at) of unbackfilled rows)",
    )
    args = parser.parse_args()

    conn = duckdb.connect(args.db, read_only=False)
    try:
        remaining_before = conn.execute(
            "SELECT COUNT(*) FROM free_bike_status WHERE tile_id IS NULL"
        ).fetchone()[0]
        if remaining_before == 0:
            print("Nothing to backfill — tile_id is populated for every row.")
            return

        bounds = conn.execute(
            """
            SELECT MIN(fetched_at), MAX(fetched_at)
            FROM free_bike_status
            WHERE tile_id IS NULL
            """
        ).fetchone()
        start, end_of_data = bounds
        if args.resume_from is not None:
            start = max(start, args.resume_from)
        print(f"Backfilling {remaining_before:,} rows from {start} through {end_of_data}.")

        batch_delta = timedelta(days=args.batch_days)
        cursor = start
        processed = 0
        t0 = time.monotonic()
        while cursor <= end_of_data:
            window_end = cursor + batch_delta
            updated = _backfill_window(conn, cursor, window_end)
            processed += updated
            conn.execute("CHECKPOINT")
            elapsed = time.monotonic() - t0
            rate = processed / elapsed if elapsed > 0 else 0.0
            print(f"  {cursor.date()} → {window_end.date()}: +{updated:,} rows "
                  f"(running total {processed:,}; {rate:,.0f} rows/sec)")
            cursor = window_end

        remaining_after = conn.execute(
            "SELECT COUNT(*) FROM free_bike_status WHERE tile_id IS NULL"
        ).fetchone()[0]
        print(f"Done. Remaining NULL rows: {remaining_after:,}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
