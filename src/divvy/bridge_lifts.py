"""Chicago River drawbridge lift schedule.

There is no clean real-time API for Chicago River bridge openings; the city
publishes Spring and Fall schedules on chicago.gov as static HTML. This
module owns the storage and provides:

  - `add_lift(...)` / `bulk_add_lifts(...)` — programmatic insert
  - CLI: `python -m divvy.bridge_lifts ingest <csv-path>` — load from CSV

The intended workflow is:
  1. At the start of each season (Apr/Oct), pull the schedule from
     https://www.chicago.gov/.../bridges/bridge-lift-schedule.html
  2. Hand-format into the CSV described in the CLI help.
  3. Run `python -m divvy.bridge_lifts ingest schedule.csv`.

If you'd rather scrape: subclass `LiftIngest` and override `parse`. Each
city schedule has been formatted slightly differently year-to-year so a
hand-rolled parser is more brittle than a quarterly manual sync.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import duckdb

from . import db

log = logging.getLogger("divvy.bridge_lifts")


def _parse_dt(value: str) -> datetime | None:
    if not value:
        return None
    s = value.strip()
    for fmt in ("%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def add_lift(
    conn: duckdb.DuckDBPyConnection,
    *,
    scheduled_start: datetime,
    scheduled_end: datetime | None = None,
    direction: str | None = None,
    bridges: list[str] | None = None,
    season: str | None = None,
    source: str = "manual",
    notes: str | None = None,
    lift_id: str | None = None,
) -> str:
    lift_id = lift_id or f"lift_{uuid.uuid4().hex[:12]}"
    conn.execute(
        """
        INSERT INTO chicago_bridge_lifts
          (lift_id, scheduled_start, scheduled_end, direction, bridges_json,
           season, source, notes, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (lift_id) DO UPDATE SET
          scheduled_start = excluded.scheduled_start,
          scheduled_end   = excluded.scheduled_end,
          direction       = excluded.direction,
          bridges_json    = excluded.bridges_json,
          season          = excluded.season,
          source          = excluded.source,
          notes           = excluded.notes,
          updated_at      = excluded.updated_at
        """,
        [
            lift_id,
            scheduled_start,
            scheduled_end,
            direction,
            json.dumps(bridges) if bridges else None,
            season,
            source,
            notes,
            datetime.now(timezone.utc).replace(tzinfo=None),
        ],
    )
    return lift_id


def bulk_add_lifts(conn: duckdb.DuckDBPyConnection, lifts: Iterable[dict]) -> int:
    n = 0
    for lift in lifts:
        add_lift(conn, **lift)
        n += 1
    return n


def ingest_csv(conn: duckdb.DuckDBPyConnection, path: Path) -> int:
    """Ingest a schedule CSV.

    Required columns: scheduled_start (ISO 8601), direction.
    Optional: scheduled_end, bridges (comma- or pipe-separated), season, notes.
    """
    n = 0
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            scheduled_start = _parse_dt(row.get("scheduled_start") or "")
            if scheduled_start is None:
                log.warning("skipping row with unparseable scheduled_start: %s", row)
                continue
            bridges_field = (row.get("bridges") or "").strip()
            bridges = (
                [b.strip() for b in bridges_field.replace("|", ",").split(",") if b.strip()]
                if bridges_field
                else None
            )
            add_lift(
                conn,
                scheduled_start=scheduled_start,
                scheduled_end=_parse_dt(row.get("scheduled_end") or ""),
                direction=(row.get("direction") or "").strip() or None,
                bridges=bridges,
                season=(row.get("season") or "").strip() or None,
                source=f"csv:{path.name}",
                notes=(row.get("notes") or "").strip() or None,
            )
            n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description="Manage Chicago bridge-lift schedule.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    ingest = sub.add_parser("ingest", help="Load a schedule CSV.")
    ingest.add_argument("path", type=Path)
    sub.add_parser("list", help="Show all known lifts.")
    args = parser.parse_args(argv)

    if args.cmd == "ingest":
        with db.session(read_only=False) as conn:
            db.init_schema(conn)
            n = ingest_csv(conn, args.path)
            conn.execute("CHECKPOINT")
        log.info("ingested %d lifts from %s", n, args.path)
        return 0
    if args.cmd == "list":
        with db.session(read_only=True) as conn:
            for row in conn.execute(
                "SELECT scheduled_start, direction, bridges_json, season "
                "FROM chicago_bridge_lifts ORDER BY scheduled_start"
            ).fetchall():
                print(row)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
