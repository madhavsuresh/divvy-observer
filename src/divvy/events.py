"""Ticketmaster event sync — venues within Chicago Divvy service area.

Daily poll into the existing `external_events` table (already defined in
db.py — designed for exactly this purpose). Event id is prefixed with
'tm_' to avoid collisions with other sources we may add later (Eventbrite,
PredictHQ, Chicago Park District feeds).

Captures: event name, venue, start/end time, lat/lon, an inferred ~0.5 km
radius for crowd modeling. The raw response is stashed in metadata_json
so you can re-derive any field without re-polling.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

import duckdb
import requests

from . import config

log = logging.getLogger("divvy.events")


def _parse_iso(value: Any) -> datetime | None:
    if not value:
        return None
    s = str(value)
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s).astimezone(timezone.utc).replace(tzinfo=None)
    except (ValueError, TypeError):
        return None


def _maybe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _venue_geo(event: dict[str, Any]) -> tuple[float | None, float | None, str | None]:
    embedded = event.get("_embedded") or {}
    venues = embedded.get("venues") or []
    if not venues:
        return None, None, None
    v = venues[0]
    loc = v.get("location") or {}
    return _maybe_float(loc.get("latitude")), _maybe_float(loc.get("longitude")), v.get("name")


def _fetch_events(page_size: int = 200, max_pages: int = 5) -> list[dict[str, Any]]:
    all_events: list[dict[str, Any]] = []
    for page in range(max_pages):
        resp = requests.get(
            config.TICKETMASTER_URL,
            params={
                "apikey": config.TICKETMASTER_API_KEY,
                "latlong": config.TICKETMASTER_CENTER,
                "radius": config.TICKETMASTER_RADIUS_MILES,
                "unit": "miles",
                "size": page_size,
                "page": page,
                "sort": "date,asc",
            },
            timeout=20,
        )
        resp.raise_for_status()
        payload = resp.json()
        events = (payload.get("_embedded") or {}).get("events") or []
        if not events:
            break
        all_events.extend(events)
        page_info = payload.get("page") or {}
        if page + 1 >= int(page_info.get("totalPages") or 0):
            break
    return all_events


def poll_upcoming(conn: duckdb.DuckDBPyConnection) -> int:
    """Refresh upcoming events into external_events."""
    if not config.TICKETMASTER_API_KEY:
        return 0
    try:
        events = _fetch_events()
    except requests.RequestException as exc:
        log.warning("Ticketmaster fetch failed: %s", exc)
        return 0
    if not events:
        return 0

    rows = []
    for e in events:
        ext_id = e.get("id")
        if not ext_id:
            continue
        dates = e.get("dates") or {}
        start = dates.get("start") or {}
        end = dates.get("end") or {}
        starts_at = _parse_iso(start.get("dateTime"))
        ends_at = _parse_iso(end.get("dateTime"))
        lat, lon, venue_name = _venue_geo(e)
        rows.append(
            (
                f"tm_{ext_id}",
                "ticketmaster",
                e.get("name"),
                starts_at,
                ends_at,
                lat,
                lon,
                0.5,  # default ~0.5 km radius for crowd modeling
                json.dumps({"venue": venue_name, "raw": e}),
            )
        )
    if not rows:
        return 0

    conn.executemany(
        """
        INSERT INTO external_events
          (event_id, event_source, event_name, starts_at, ends_at,
           lat, lon, radius_km, metadata_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (event_id) DO UPDATE SET
          event_name    = excluded.event_name,
          starts_at     = excluded.starts_at,
          ends_at       = excluded.ends_at,
          lat           = excluded.lat,
          lon           = excluded.lon,
          metadata_json = excluded.metadata_json
        """,
        rows,
    )
    return len(rows)
