"""Chicago Traffic Tracker — region-level congestion estimates.

Pulls the Data Portal dataset 8v9j-bter (Chicago Traffic Tracker - Congestion
Estimates by Regions). No API key required; SODA_APP_TOKEN env var lifts
rate limits.

Each row is the most recent ~10-minute speed estimate for one of ~29 named
regions. The dataset has historically been inconsistent — sometimes stale for
days. We capture whatever we get and flag stale snapshots in the log.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import duckdb
import requests

from . import config

log = logging.getLogger("divvy.traffic")


def _parse_iso(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        # SODA emits "YYYY-MM-DDTHH:MM:SS.000"
        return datetime.fromisoformat(str(value)[:19])
    except (ValueError, TypeError):
        return None


def _maybe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _maybe_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _fetch_regions() -> list[dict[str, Any]]:
    headers = {"Accept": "application/json"}
    if config.SODA_APP_TOKEN:
        headers["X-App-Token"] = config.SODA_APP_TOKEN
    resp = requests.get(
        config.TRAFFIC_REGIONS_URL,
        headers=headers,
        params={"$limit": 5000},
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list):
        return []
    return data


def poll_current(conn: duckdb.DuckDBPyConnection) -> int:
    """Snapshot current region speeds. Dedupes on (region_id, observed_at)."""
    fetched_at = datetime.now(timezone.utc).replace(tzinfo=None)
    try:
        regions = _fetch_regions()
    except requests.RequestException as exc:
        log.warning("Chicago traffic fetch failed: %s", exc)
        return 0

    if not regions:
        return 0

    rows = []
    newest_observed: datetime | None = None
    for r in regions:
        # Field names vary slightly across dataset versions; tolerate both.
        region_id = (
            r.get("region_id")
            or r.get("_region_id")
            or r.get("region")
        )
        if not region_id:
            continue
        observed_at = (
            _parse_iso(r.get("last_updt"))
            or _parse_iso(r.get("time"))
            or _parse_iso(r.get("_last_updt"))
            or fetched_at
        )
        if newest_observed is None or observed_at > newest_observed:
            newest_observed = observed_at
        rows.append(
            (
                str(region_id),
                observed_at,
                r.get("region") or r.get("description"),
                _maybe_float(r.get("current_speed") or r.get("speed")),
                _maybe_int(r.get("bus_count")),
                _maybe_float(r.get("west")),
                _maybe_float(r.get("east")),
                _maybe_float(r.get("north")),
                _maybe_float(r.get("south")),
                r.get("description"),
                fetched_at,
            )
        )
    if not rows:
        return 0

    # Warn loudly if the upstream snapshot is more than an hour old.
    if newest_observed is not None:
        age_seconds = (fetched_at - newest_observed).total_seconds()
        if age_seconds > 3600:
            log.warning(
                "Chicago traffic snapshot is %.0f minutes stale (newest=%s)",
                age_seconds / 60.0,
                newest_observed.isoformat(),
            )

    conn.executemany(
        """
        INSERT INTO chicago_traffic_regions
          (region_id, observed_at, region_name, speed_mph, bus_count,
           west, east, north, south, description, fetched_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (region_id, observed_at) DO UPDATE SET
          speed_mph  = excluded.speed_mph,
          bus_count  = excluded.bus_count,
          fetched_at = excluded.fetched_at
        """,
        rows,
    )
    return len(rows)
