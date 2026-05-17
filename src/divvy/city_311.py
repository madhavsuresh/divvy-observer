"""Chicago 311 service-request incremental sync.

The Data Portal dataset id is env-configurable because Chicago periodically
migrates the 311 dataset to a new resource id. Default is the
"311 Service Requests" current dataset.

We sync only requests created in the last DIVVY_311_LOOKBACK_HOURS, but
INSERT OR REPLACE matches existing rows by sr_number so a long-running
request being updated (status change, close) still gets its fields refreshed.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import duckdb
import requests

from . import config

log = logging.getLogger("divvy.city_311")


def _parse_iso(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value)[:19])
    except (ValueError, TypeError):
        return None


def _maybe_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _maybe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fetch_recent(lookback_hours: int) -> list[dict[str, Any]]:
    cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(
        hours=lookback_hours
    )
    headers = {"Accept": "application/json"}
    if config.SODA_APP_TOKEN:
        headers["X-App-Token"] = config.SODA_APP_TOKEN
    resp = requests.get(
        config.CITY_311_URL,
        headers=headers,
        params={
            "$limit": 50000,
            "$where": f"created_date > '{cutoff.isoformat()}'",
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list):
        return []
    return data


def poll_recent(conn: duckdb.DuckDBPyConnection) -> int:
    """Pull requests created since lookback window, upsert by sr_number."""
    fetched_at = datetime.now(timezone.utc).replace(tzinfo=None)
    try:
        records = _fetch_recent(config.CITY_311_LOOKBACK_HOURS)
    except requests.RequestException as exc:
        log.warning("Chicago 311 fetch failed: %s", exc)
        return 0
    if not records:
        return 0

    rows = []
    for r in records:
        sr_number = r.get("sr_number")
        if not sr_number:
            continue
        rows.append(
            (
                sr_number,
                r.get("sr_type"),
                r.get("sr_short_code"),
                r.get("status"),
                _parse_iso(r.get("created_date")),
                _parse_iso(r.get("closed_date")),
                _maybe_int(r.get("community_area")),
                _maybe_int(r.get("ward")),
                r.get("street_address"),
                r.get("city"),
                r.get("zip_code"),
                _maybe_float(r.get("latitude")),
                _maybe_float(r.get("longitude")),
                json.dumps(r),
                fetched_at,
                fetched_at,
            )
        )
    if not rows:
        return 0

    conn.executemany(
        """
        INSERT INTO chicago_311_requests
          (sr_number, sr_type, sr_short_code, status, created_at, closed_at,
           community_area, ward, street_address, city, zip_code, lat, lon,
           raw_json, fetched_at, last_seen_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (sr_number) DO UPDATE SET
          status       = excluded.status,
          closed_at    = excluded.closed_at,
          raw_json     = excluded.raw_json,
          last_seen_at = excluded.last_seen_at
        """,
        rows,
    )
    return len(rows)
