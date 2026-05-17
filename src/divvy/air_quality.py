"""AirNow (EPA) AQI snapshots for Chicago.

Polls a handful of representative Chicago ZIP codes hourly. Each ZIP returns
one observation per pollutant (PM2.5, O3, etc.). Free key required from
https://docs.airnowapi.org/ — without it this module no-ops.

Backfill is impossible: AirNow's "current observation" endpoint only returns
the latest reading. Historical data requires a separate paid product.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import duckdb
import requests

from . import config

log = logging.getLogger("divvy.air_quality")


def _parse_observed_at(date_str: str, hour: int) -> datetime | None:
    try:
        y, m, d = (int(p) for p in date_str.strip().split("-")[:3])
        return datetime(y, m, d, int(hour), 0, 0)
    except (ValueError, AttributeError):
        return None


def _category_name(payload: Any) -> str | None:
    if isinstance(payload, dict):
        return payload.get("Name")
    return None


def _fetch_zip(zipcode: str) -> list[dict[str, Any]]:
    resp = requests.get(
        config.AIRNOW_URL,
        params={
            "format": "application/json",
            "zipCode": zipcode,
            "distance": config.AIRNOW_DISTANCE_MILES,
            "API_KEY": config.AIRNOW_API_KEY,
        },
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list):
        return []
    return data


def poll_current(conn: duckdb.DuckDBPyConnection) -> int:
    """Pull current AQI for each configured ZIP, upsert into observations table."""
    if not config.AIRNOW_API_KEY:
        return 0
    fetched_at = datetime.now(timezone.utc).replace(tzinfo=None)
    total = 0
    for zipcode in config.AIRNOW_ZIPS:
        try:
            observations = _fetch_zip(zipcode)
        except requests.RequestException as exc:
            log.warning("AirNow fetch failed for %s: %s", zipcode, exc)
            continue

        rows = []
        for obs in observations:
            observed_at = _parse_observed_at(
                obs.get("DateObserved", ""), obs.get("HourObserved", 0)
            )
            if observed_at is None or not obs.get("ParameterName"):
                continue
            aqi = obs.get("AQI")
            rows.append(
                (
                    observed_at,
                    zipcode,
                    obs["ParameterName"],
                    int(aqi) if aqi is not None else None,
                    _category_name(obs.get("Category")),
                    obs.get("ReportingArea"),
                    obs.get("StateCode"),
                    float(obs["Latitude"]) if obs.get("Latitude") is not None else None,
                    float(obs["Longitude"]) if obs.get("Longitude") is not None else None,
                    fetched_at,
                )
            )
        if not rows:
            continue

        conn.executemany(
            """
            INSERT INTO air_quality_observations
              (observed_at, zipcode, parameter, aqi, category,
               reporting_area, state_code, lat, lon, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (observed_at, zipcode, parameter) DO UPDATE SET
              aqi            = excluded.aqi,
              category       = excluded.category,
              reporting_area = excluded.reporting_area,
              fetched_at     = excluded.fetched_at
            """,
            rows,
        )
        total += len(rows)
    return total
