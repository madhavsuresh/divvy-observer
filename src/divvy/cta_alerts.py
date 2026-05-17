"""CTA Customer Alerts — service disruptions and reroutes.

Polls lapi.transitchicago.com/api/1.0/alerts.aspx every minute. Uses the
same CTA_API_KEY / CTA_TRAIN_API_KEY family as transit-observer, so a single
key works for both repos.

Schema follows the observed-window pattern (alert_id + last_updated_at PK,
first_observed_at / last_observed_at as bookkeeping) so a stable alert that
sits in the feed for hours produces one row, retired alerts have a frozen
last_observed_at, and content updates produce a new row.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

import duckdb
import requests

from . import config

log = logging.getLogger("divvy.cta_alerts")


def _parse_cta_ts(value: Any) -> datetime | None:
    """CTA timestamps are 'YYYYMMDD HH:MM:SS' in local time, no tz."""
    if not value:
        return None
    s = str(value).strip()
    try:
        if "T" in s:
            return datetime.fromisoformat(s[:19])
        # "20260516 14:33:00"
        return datetime.strptime(s, "%Y%m%d %H:%M:%S")
    except ValueError:
        return None


def _as_bool(value: Any) -> bool | None:
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in ("1", "true", "yes"):
        return True
    if s in ("0", "false", "no"):
        return False
    return None


def _impacted_services(alert: dict[str, Any]) -> list[dict[str, Any]]:
    """ImpactedService can be missing, a single dict, or a list of dicts."""
    impacted = alert.get("ImpactedService")
    if impacted is None:
        return []
    inner = impacted.get("Service") if isinstance(impacted, dict) else impacted
    if inner is None:
        return []
    if isinstance(inner, list):
        return [s for s in inner if isinstance(s, dict)]
    if isinstance(inner, dict):
        return [inner]
    return []


def _fetch_alerts() -> list[dict[str, Any]]:
    params: dict[str, Any] = {"outputType": "JSON", "activeonly": "true"}
    if config.CTA_API_KEY:
        params["key"] = config.CTA_API_KEY
    resp = requests.get(config.CTA_ALERTS_URL, params=params, timeout=15)
    resp.raise_for_status()
    payload = resp.json()
    root = payload.get("CTAAlerts") or {}
    alerts = root.get("Alert") or []
    if isinstance(alerts, dict):
        alerts = [alerts]
    return alerts if isinstance(alerts, list) else []


def poll_current(conn: duckdb.DuckDBPyConnection) -> int:
    """Snapshot active CTA alerts, bump last_observed_at on repeat sightings."""
    fetched_at = datetime.now(timezone.utc).replace(tzinfo=None)
    try:
        alerts = _fetch_alerts()
    except requests.RequestException as exc:
        log.warning("CTA alerts fetch failed: %s", exc)
        return 0

    if not alerts:
        return 0

    rows = []
    for a in alerts:
        alert_id = a.get("AlertId") or a.get("alert_id")
        if not alert_id:
            continue
        last_updated = (
            _parse_cta_ts(a.get("EventStart"))
            or _parse_cta_ts(a.get("TBD"))
            or fetched_at
        )
        impacted = _impacted_services(a)
        rows.append(
            (
                str(alert_id),
                last_updated,
                a.get("SeverityScore") or a.get("Severity"),
                a.get("ImpactedService", {}).get("ServiceId")
                if isinstance(a.get("ImpactedService"), dict)
                else None,
                a.get("Headline"),
                a.get("ShortDescription"),
                a.get("FullDescription") or a.get("Description"),
                _parse_cta_ts(a.get("EventStart")),
                _parse_cta_ts(a.get("EventEnd")),
                _as_bool(a.get("MajorAlert")),
                a.get("AlertURL"),
                json.dumps(impacted) if impacted else None,
                json.dumps(a),
                fetched_at,
                fetched_at,
            )
        )
    if not rows:
        return 0

    conn.executemany(
        """
        INSERT INTO cta_alerts
          (alert_id, last_updated_at, severity, service_id, headline,
           short_description, full_description, start_ts, end_ts,
           major_alert, alert_url, impacted_services_json, raw_json,
           first_observed_at, last_observed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (alert_id, last_updated_at) DO UPDATE SET
          last_observed_at = excluded.last_observed_at,
          raw_json         = excluded.raw_json
        """,
        rows,
    )
    return len(rows)
