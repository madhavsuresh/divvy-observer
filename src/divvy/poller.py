from __future__ import annotations

import json
import logging
import signal
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

import pandas as pd

import requests

from . import (
    air_quality,
    config,
    cta_alerts,
    city_311,
    db,
    events,
    forecast_queue,
    model_eval,
    service_state,
    tile,
    traffic,
    weather,
)

log = logging.getLogger("divvy.poller")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )


def _fetch(url: str) -> dict[str, Any]:
    headers = {"User-Agent": config.USER_AGENT, "Accept": "application/json"}
    resp = requests.get(url, headers=headers, timeout=config.REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def _fetch_optional(url: str) -> dict[str, Any] | None:
    """Like _fetch, but returns None on 404 / connection errors instead of raising.

    GBFS publishers don't always serve every optional file (system_alerts and
    vehicle_types in particular). We want a missing endpoint to be a silent
    no-op, not a tick failure.
    """
    headers = {"User-Agent": config.USER_AGENT, "Accept": "application/json"}
    try:
        resp = requests.get(url, headers=headers, timeout=config.REQUEST_TIMEOUT)
    except requests.RequestException as exc:
        log.warning("optional GBFS fetch failed for %s: %s", url, exc)
        return None
    if resp.status_code == 404:
        return None
    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        log.warning("optional GBFS fetch HTTP error for %s: %s", url, exc)
        return None
    try:
        return resp.json()
    except ValueError as exc:
        log.warning("optional GBFS fetch returned non-JSON for %s: %s", url, exc)
        return None


def _ts(unix_seconds: int | float) -> datetime:
    return datetime.fromtimestamp(int(unix_seconds), tz=timezone.utc).replace(tzinfo=None)


@dataclass
class PollerState:
    last_info_refresh_unix: float = 0.0
    last_outcome_resolve_unix: float = 0.0
    last_metrics_snapshot_unix: float = 0.0
    last_self_eval_unix: float = 0.0
    self_eval_tick_index: int = 0
    last_read_replica_refresh_unix: float = 0.0
    last_vehicle_types_refresh_unix: float = 0.0
    last_external_poll_unix: dict[str, float] = field(default_factory=dict)
    consecutive_failures: int = 0
    rows_inserted_total: int = 0


# (name, cadence_seconds_callable, poll_fn). The cadence is a callable so the
# value is re-read from config each call — lets you tune via env without
# restart-by-design (config module re-reads on import is enough for now).
def _external_feeds() -> list[tuple[str, int, Callable[[Any], int]]]:
    return [
        ("cta_alerts", config.CTA_ALERTS_POLL_SECONDS, cta_alerts.poll_current),
        ("air_quality", config.AIRNOW_POLL_SECONDS, air_quality.poll_current),
        ("traffic", config.TRAFFIC_POLL_SECONDS, traffic.poll_current),
        ("city_311", config.CITY_311_POLL_SECONDS, city_311.poll_recent),
        ("events", config.TICKETMASTER_POLL_SECONDS, events.poll_upcoming),
        ("weather_forecast", config.WEATHER_FORECAST_POLL_SECONDS, weather.poll_forecast_snapshot),
        ("weather_nowcast", config.WEATHER_NOWCAST_POLL_SECONDS, weather.poll_nowcast),
    ]


def _log_external_poll(
    conn,
    name: str,
    started_at: datetime,
    rows: int,
    status: str,
    error_msg: str | None,
    duration_ms: int,
) -> None:
    try:
        conn.execute(
            """
            INSERT INTO external_data_poll_log
              (source, polled_at, rows_affected, status, error_message, duration_ms)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (source, polled_at) DO NOTHING
            """,
            [name, started_at, rows, status, error_msg, duration_ms],
        )
    except Exception:
        log.exception("failed to write external_data_poll_log entry for %s", name)


def run_due_external_polls(state: PollerState, now: float) -> dict[str, int]:
    """Dispatch all due external-data polls. Each runs in its own write
    session so a slow source doesn't block the GBFS main loop and a failure
    doesn't cascade across sources. The HTTP call sits inside the lock for
    convenience; cadences are long (5 min+) so contention is rare.
    """
    summary: dict[str, int] = {}
    for name, cadence, fn in _external_feeds():
        if cadence <= 0:
            continue
        last = state.last_external_poll_unix.get(name, 0.0)
        if (now - last) < cadence:
            continue
        started_unix = time.time()
        started_at = datetime.fromtimestamp(started_unix, tz=timezone.utc).replace(tzinfo=None)
        rows = 0
        status = "ok"
        error_msg: str | None = None
        try:
            with db.session(read_only=False, retries=10) as conn:
                rows = int(fn(conn) or 0)
                state.last_external_poll_unix[name] = now
                _log_external_poll(
                    conn,
                    name,
                    started_at,
                    rows,
                    status,
                    error_msg,
                    int((time.time() - started_unix) * 1000),
                )
        except Exception as exc:
            log.exception("external poll %s failed: %s", name, exc)
            status = "error"
            error_msg = str(exc)[:500]
            try:
                with db.session(read_only=False, retries=10) as conn:
                    _log_external_poll(
                        conn,
                        name,
                        started_at,
                        rows,
                        status,
                        error_msg,
                        int((time.time() - started_unix) * 1000),
                    )
            except Exception:
                pass
        summary[name] = rows
        if rows or status != "ok":
            log.info(
                "external %s: %d rows in %dms (%s)",
                name,
                rows,
                int((time.time() - started_unix) * 1000),
                status,
            )
    return summary


def process_prediction_writes(
    conn,
    state: PollerState,
    now: float,
    pre_scored_self_eval: pd.DataFrame | None = None,
) -> dict:
    """Drain prediction-service writes while the collector owns DuckDB.

    Pass ``pre_scored_self_eval`` (result of
    :func:`model_eval.score_self_evaluation_candidates` called before the
    write lock was acquired) to skip ML inference inside the lock.
    """
    summary = {
        "forecast_queue_files_processed": 0,
        "forecast_rows_logged": 0,
        "forecast_queue_files_failed": 0,
        "outcomes_resolved": 0,
        "metrics_rows_snapshotted": 0,
        "self_eval_forecasts_logged": 0,
    }
    try:
        drained = forecast_queue.drain_forecast_queue(
            conn,
            limit=config.FORECAST_QUEUE_DRAIN_LIMIT,
        )
        summary["forecast_queue_files_processed"] = int(drained["files_processed"])
        summary["forecast_rows_logged"] = int(drained["forecast_rows_logged"])
        summary["forecast_queue_files_failed"] = int(drained["files_failed"])
        if drained["files_processed"] or drained["files_failed"]:
            log.info(
                "forecast queue: %d files, %d forecast rows, %d failed",
                drained["files_processed"],
                drained["forecast_rows_logged"],
                drained["files_failed"],
            )
    except Exception as exc:
        log.exception("forecast queue drain failed: %s", exc)

    if (now - state.last_outcome_resolve_unix) >= config.MODEL_OUTCOME_RESOLVE_SECONDS:
        try:
            resolved = model_eval.resolve_due_outcomes(conn)
            summary["outcomes_resolved"] = int(resolved)
            state.last_outcome_resolve_unix = now
            if resolved:
                log.info("resolved %d forecast outcomes", resolved)
        except Exception as exc:
            log.exception("forecast outcome resolution failed: %s", exc)

    if config.SELF_EVAL_INTERVAL_SECONDS > 0 and (
        now - state.last_self_eval_unix
    ) >= config.SELF_EVAL_INTERVAL_SECONDS:
        try:
            if pre_scored_self_eval is not None:
                emitted = model_eval.log_self_evaluation_scored(conn, pre_scored_self_eval)
            else:
                emitted = model_eval.emit_self_evaluation_forecasts(
                    conn,
                    station_sample_size=config.SELF_EVAL_STATION_SAMPLE,
                    tick_index=state.self_eval_tick_index,
                )
            summary["self_eval_forecasts_logged"] = int(emitted)
            state.last_self_eval_unix = now
            state.self_eval_tick_index += 1
            if emitted:
                log.info("emitted %d self-eval forecasts", emitted)
        except Exception as exc:
            log.exception("self-eval forecast emission failed: %s", exc)

    if config.MODEL_METRICS_SNAPSHOT_SECONDS > 0 and (
        now - state.last_metrics_snapshot_unix
    ) >= config.MODEL_METRICS_SNAPSHOT_SECONDS:
        try:
            metrics = model_eval.snapshot_metrics(conn, window_hours=24)
            summary["metrics_rows_snapshotted"] = int(metrics)
            state.last_metrics_snapshot_unix = now
            if metrics:
                log.info("snapshotted %d model metric rows", metrics)
        except Exception as exc:
            log.exception("model metric snapshot failed: %s", exc)
    return summary


def insert_collector_tick(
    conn,
    *,
    ticked_at: datetime,
    station_payload_count: int,
    free_bike_payload_count: int,
    station_rows_inserted: int,
    free_bike_events_inserted: int,
    prediction_summary: dict,
) -> None:
    conn.execute(
        """
        INSERT INTO collector_ticks (
          tick_id, ticked_at, station_payload_count, free_bike_payload_count,
          station_rows_inserted, free_bike_events_inserted,
          forecast_queue_files_processed, forecast_rows_logged,
          forecast_queue_files_failed, outcomes_resolved, metrics_rows_snapshotted
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            str(uuid.uuid4()),
            ticked_at,
            station_payload_count,
            free_bike_payload_count,
            station_rows_inserted,
            free_bike_events_inserted,
            int(prediction_summary.get("forecast_queue_files_processed") or 0),
            int(prediction_summary.get("forecast_rows_logged") or 0),
            int(prediction_summary.get("forecast_queue_files_failed") or 0),
            int(prediction_summary.get("outcomes_resolved") or 0),
            int(prediction_summary.get("metrics_rows_snapshotted") or 0),
        ],
    )


def upsert_stations(conn, payload: dict[str, Any]) -> int:
    stations = payload["data"]["stations"]
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    rows = [
        (
            s["station_id"],
            s.get("legacy_id"),
            s.get("short_name"),
            s.get("name"),
            s.get("lat"),
            s.get("lon"),
            s.get("capacity"),
            s.get("station_type"),
            now,
            now,
        )
        for s in stations
    ]
    conn.executemany(
        """
        INSERT INTO stations
          (station_id, legacy_id, short_name, name, lat, lon, capacity, station_type,
           first_seen_at, last_seen_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (station_id) DO UPDATE SET
          legacy_id    = excluded.legacy_id,
          short_name   = excluded.short_name,
          name         = excluded.name,
          lat          = excluded.lat,
          lon          = excluded.lon,
          capacity     = excluded.capacity,
          station_type = excluded.station_type,
          last_seen_at = excluded.last_seen_at
        """,
        rows,
    )
    return len(rows)


def insert_status(conn, payload: dict[str, Any]) -> int:
    stations = payload["data"]["stations"]
    fetched_at = datetime.now(timezone.utc).replace(tzinfo=None)
    rows = [
        (
            s["station_id"],
            _ts(s["last_reported"]),
            fetched_at,
            s.get("num_bikes_available"),
            s.get("num_ebikes_available"),
            s.get("num_bikes_disabled"),
            s.get("num_docks_available"),
            s.get("num_docks_disabled"),
            bool(s.get("is_installed")),
            bool(s.get("is_renting")),
            bool(s.get("is_returning")),
        )
        for s in stations
        if s.get("last_reported") is not None
    ]
    before = conn.execute("SELECT COUNT(*) FROM station_status").fetchone()[0]
    conn.executemany(
        """
        INSERT OR IGNORE INTO station_status
          (station_id, last_reported, fetched_at,
           num_bikes_available, num_ebikes_available, num_bikes_disabled,
           num_docks_available, num_docks_disabled,
           is_installed, is_renting, is_returning)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    after = conn.execute("SELECT COUNT(*) FROM station_status").fetchone()[0]
    return after - before


def _range_bucket(meters: float | None) -> int | None:
    """Round ebike range to the nearest km for diff purposes.

    Range readings flicker by tens of meters between ticks even on a parked
    bike. Bucketing to 1 km means small noise doesn't generate writes, but
    a ride or a real idle drain does.
    """
    if meters is None:
        return None
    return int(round(meters / 1000.0))


def insert_free_bikes(conn, payload: dict[str, Any]) -> int:
    """Persist free-floating bike positions, deduping unchanged rows.

    Each tick, GBFS reports ~2,500 bikes. Most haven't moved since the last
    tick — writing every one would balloon storage. We compare each bike's
    current (lat, lon, is_reserved, is_disabled, range_bucket) against its
    most recent row and only insert when something changed (or the bike is
    new). Range is bucketed to 1 km so flicker doesn't trigger writes.
    """
    fetched_at = datetime.now(timezone.utc).replace(tzinfo=None)
    incoming = [
        {
            "bike_id": b["bike_id"],
            "name": b.get("name"),
            "lat": float(b["lat"]) if b.get("lat") is not None else None,
            "lon": float(b["lon"]) if b.get("lon") is not None else None,
            "is_reserved": bool(b.get("is_reserved")),
            "is_disabled": bool(b.get("is_disabled")),
            "vehicle_type_id": b.get("vehicle_type_id"),
            "current_range_meters": (
                float(b["current_range_meters"])
                if b.get("current_range_meters") is not None
                else None
            ),
            "current_fuel_percent": (
                float(b["current_fuel_percent"])
                if b.get("current_fuel_percent") is not None
                else None
            ),
        }
        for b in payload["data"]["bikes"]
        if b.get("lat") is not None and b.get("lon") is not None
    ]
    if not incoming:
        return 0

    # Pull the most recent row per bike for the bike_ids we just received,
    # so we can diff. Using a CTE keeps this to a single round trip.
    bike_ids = [b["bike_id"] for b in incoming]
    placeholders = ",".join(["?"] * len(bike_ids))
    last_rows = conn.execute(
        f"""
        SELECT bike_id, lat, lon, is_reserved, is_disabled,
               vehicle_type_id, current_range_meters
        FROM (
          SELECT *, ROW_NUMBER() OVER (
            PARTITION BY bike_id ORDER BY fetched_at DESC
          ) AS rn
          FROM free_bike_status
          WHERE bike_id IN ({placeholders})
        )
        WHERE rn = 1
        """,
        bike_ids,
    ).fetchall()
    last_state = {
        bid: (lat, lon, bool(res), bool(dis), vt, _range_bucket(rm))
        for bid, lat, lon, res, dis, vt, rm in last_rows
    }

    rows_to_insert = []
    for b in incoming:
        prev = last_state.get(b["bike_id"])
        curr = (
            b["lat"],
            b["lon"],
            b["is_reserved"],
            b["is_disabled"],
            b["vehicle_type_id"],
            _range_bucket(b["current_range_meters"]),
        )
        if prev == curr:
            continue  # unchanged — skip
        rows_to_insert.append((
            b["bike_id"], fetched_at, b["name"],
            b["lat"], b["lon"], b["is_reserved"], b["is_disabled"],
            tile.tile_id_for(b["lat"], b["lon"]),
            b["vehicle_type_id"],
            b["current_range_meters"],
            b["current_fuel_percent"],
        ))

    if not rows_to_insert:
        return 0

    conn.executemany(
        """
        INSERT OR IGNORE INTO free_bike_status
          (bike_id, fetched_at, name, lat, lon, is_reserved, is_disabled, tile_id,
           vehicle_type_id, current_range_meters, current_fuel_percent)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows_to_insert,
    )
    return len(rows_to_insert)


def _maybe_ts(unix_seconds: Any) -> datetime | None:
    if unix_seconds is None:
        return None
    try:
        return datetime.fromtimestamp(int(unix_seconds), tz=timezone.utc).replace(tzinfo=None)
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def upsert_system_alerts(conn, payload: dict[str, Any]) -> int:
    """Capture GBFS system_alerts.

    Alerts use composite key (alert_id, last_updated). When an alert changes
    content the feed bumps its `last_updated`, producing a new row here.
    `first_observed_at` is frozen on initial sight; `last_observed_at` is
    bumped every tick the alert is still in the feed, so retired alerts have
    a frozen last_observed_at.
    """
    fetched_at = datetime.now(timezone.utc).replace(tzinfo=None)
    alerts = (payload.get("data") or {}).get("alerts") or []
    if not alerts:
        return 0

    rows = []
    for a in alerts:
        alert_id = a.get("alert_id")
        if not alert_id:
            continue
        last_updated = _maybe_ts(a.get("last_updated")) or fetched_at
        times = a.get("times") or []
        start_ts = None
        end_ts = None
        if isinstance(times, list) and times:
            first = times[0] if isinstance(times[0], dict) else {}
            start_ts = _maybe_ts(first.get("start"))
            end_ts = _maybe_ts(first.get("end"))
        rows.append(
            (
                alert_id,
                last_updated,
                a.get("type"),
                json.dumps(a.get("station_ids")) if a.get("station_ids") is not None else None,
                json.dumps(a.get("region_ids")) if a.get("region_ids") is not None else None,
                start_ts,
                end_ts,
                a.get("url"),
                a.get("summary"),
                a.get("description"),
                json.dumps(a),
                fetched_at,
                fetched_at,
            )
        )
    if not rows:
        return 0

    conn.executemany(
        """
        INSERT INTO gbfs_system_alerts
          (alert_id, last_updated_at, alert_type, station_ids_json, region_ids_json,
           start_ts, end_ts, url, summary, description, raw_json,
           first_observed_at, last_observed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (alert_id, last_updated_at) DO UPDATE SET
          last_observed_at = excluded.last_observed_at
        """,
        rows,
    )
    return len(rows)


def upsert_vehicle_types(conn, payload: dict[str, Any]) -> int:
    """Capture GBFS vehicle_types as latest-known reference data.

    Vehicle type definitions rarely change, but when an operator introduces a
    new ebike SKU or changes range claims, that matters for replay. Daily
    poll, keyed on vehicle_type_id with first_seen / last_seen timestamps.
    """
    fetched_at = datetime.now(timezone.utc).replace(tzinfo=None)
    vts = (payload.get("data") or {}).get("vehicle_types") or []
    if not vts:
        return 0

    import json as _json
    rows = []
    for v in vts:
        vt_id = v.get("vehicle_type_id")
        if not vt_id:
            continue
        max_range = v.get("max_range_meters")
        rows.append(
            (
                vt_id,
                v.get("form_factor"),
                v.get("propulsion_type"),
                (v.get("name") or [{}])[0].get("text") if isinstance(v.get("name"), list) else v.get("name"),
                float(max_range) if max_range is not None else None,
                v.get("return_constraint"),
                json.dumps(v),
                fetched_at,
                fetched_at,
            )
        )
    if not rows:
        return 0

    conn.executemany(
        """
        INSERT INTO gbfs_vehicle_types
          (vehicle_type_id, form_factor, propulsion_type, name,
           max_range_meters, return_constraint, raw_json,
           first_seen_at, last_seen_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (vehicle_type_id) DO UPDATE SET
          form_factor       = excluded.form_factor,
          propulsion_type   = excluded.propulsion_type,
          name              = excluded.name,
          max_range_meters  = excluded.max_range_meters,
          return_constraint = excluded.return_constraint,
          raw_json          = excluded.raw_json,
          last_seen_at      = excluded.last_seen_at
        """,
        rows,
    )
    return len(rows)


def _sleep_until_next_tick(interval: int, should_continue) -> None:
    """Sleep to next multiple of `interval`, polling `should_continue()` for SIGTERM."""
    target = (int(time.time()) // interval + 1) * interval
    while should_continue():
        remaining = target - time.time()
        if remaining <= 0:
            return
        time.sleep(min(0.5, remaining))


def poll_once(state: PollerState) -> None:
    now = time.time()
    need_info = (now - state.last_info_refresh_unix) >= config.STATION_INFO_REFRESH_SECONDS
    need_vehicle_types = (
        now - state.last_vehicle_types_refresh_unix
    ) >= config.VEHICLE_TYPES_REFRESH_SECONDS

    info_payload = _fetch(config.STATION_INFO_URL) if need_info else None
    status_payload = _fetch(config.STATION_STATUS_URL)
    free_payload = _fetch(config.FREE_BIKE_STATUS_URL)
    alerts_payload = _fetch_optional(config.SYSTEM_ALERTS_URL)
    vehicle_types_payload = (
        _fetch_optional(config.VEHICLE_TYPES_URL) if need_vehicle_types else None
    )

    # Score self-eval candidates before acquiring the write lock so ML inference
    # doesn't contribute to write-lock hold time (which starves automation jobs).
    pre_scored_self_eval: pd.DataFrame | None = None
    if config.SELF_EVAL_INTERVAL_SECONDS > 0 and (
        now - state.last_self_eval_unix
    ) >= config.SELF_EVAL_INTERVAL_SECONDS:
        try:
            with db.session(read_only=True) as ro_conn:
                pre_scored_self_eval = model_eval.score_self_evaluation_candidates(
                    ro_conn,
                    station_sample_size=config.SELF_EVAL_STATION_SAMPLE,
                    tick_index=state.self_eval_tick_index,
                )
        except Exception as exc:
            log.exception("self-eval pre-scoring failed: %s", exc)

    with db.session(read_only=False) as conn:
        db.init_schema(conn)
        if info_payload is not None:
            n_stations = upsert_stations(conn, info_payload)
            state.last_info_refresh_unix = now
            log.info("refreshed station_information: %d stations", n_stations)
        inserted = insert_status(conn, status_payload)
        free_inserted = insert_free_bikes(conn, free_payload)
        alerts_rows = 0
        if alerts_payload is not None:
            try:
                alerts_rows = upsert_system_alerts(conn, alerts_payload)
            except Exception as exc:
                log.exception("system_alerts upsert failed: %s", exc)
        if vehicle_types_payload is not None:
            try:
                vt_rows = upsert_vehicle_types(conn, vehicle_types_payload)
                state.last_vehicle_types_refresh_unix = now
                if vt_rows:
                    log.info("refreshed gbfs_vehicle_types: %d rows", vt_rows)
            except Exception as exc:
                log.exception("vehicle_types upsert failed: %s", exc)
        prediction_summary = process_prediction_writes(conn, state, now, pre_scored_self_eval)
        service_state.heartbeat(
            conn,
            "divvy.collector",
            {
                "station_payload_count": len(status_payload["data"]["stations"]),
                "free_bike_payload_count": len(free_payload["data"]["bikes"]),
            },
        )
        insert_collector_tick(
            conn,
            ticked_at=datetime.now(timezone.utc).replace(tzinfo=None),
            station_payload_count=len(status_payload["data"]["stations"]),
            free_bike_payload_count=len(free_payload["data"]["bikes"]),
            station_rows_inserted=inserted,
            free_bike_events_inserted=free_inserted,
            prediction_summary=prediction_summary,
        )
        conn.execute("CHECKPOINT")

    # External-data polls (Bucket 2: CTA alerts, traffic, AQ, 311, events,
    # weather forecast snapshots, nowcast). Each runs in its own short-lived
    # write session so a slow upstream doesn't block GBFS polling. Cadences
    # are env-controlled; setting any *_POLL_SECONDS to 0 disables that feed.
    try:
        run_due_external_polls(state, now)
    except Exception as exc:
        log.exception("external poll dispatch failed: %s", exc)

    if (
        config.ENABLE_READ_REPLICA
        and (now - state.last_read_replica_refresh_unix) >= config.READ_REPLICA_REFRESH_SECONDS
    ):
        try:
            if db.refresh_read_replica():
                state.last_read_replica_refresh_unix = now
                log.info("refreshed prediction read replica: %s", config.READ_DB_PATH)
        except Exception as exc:
            log.exception("read replica refresh failed: %s", exc)

    state.rows_inserted_total += inserted
    log.info(
        "status tick: %d stations / %d free bikes in payload, "
        "%d station rows + %d free-bike events + %d alerts (total=%d)",
        len(status_payload["data"]["stations"]),
        len(free_payload["data"]["bikes"]),
        inserted,
        free_inserted,
        alerts_rows,
        state.rows_inserted_total,
    )


def main() -> int:
    _setup_logging()
    config.ensure_dirs()

    running = True

    def _stop(signum, _frame):
        nonlocal running
        log.info("received signal %d, shutting down after current tick", signum)
        running = False

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    log.info("divvy-observer collector starting; db=%s interval=%ds",
             config.DB_PATH, config.POLL_INTERVAL_SECONDS)

    # Bootstrap schema in its own short-lived connection.
    with db.session(read_only=False) as conn:
        db.init_schema(conn)
        conn.execute("CHECKPOINT")
    try:
        db.refresh_read_replica()
    except Exception as exc:
        log.exception("initial read replica refresh failed: %s", exc)

    state = PollerState()

    while running:
        try:
            poll_once(state)
            state.consecutive_failures = 0
        except Exception as exc:
            state.consecutive_failures += 1
            log.exception("poll failed (consecutive=%d): %s",
                          state.consecutive_failures, exc)

        if not running:
            break

        backoff_multiplier = min(state.consecutive_failures, 5) or 1
        interval = config.POLL_INTERVAL_SECONDS * backoff_multiplier
        _sleep_until_next_tick(interval, lambda: running)

    log.info("collector stopped cleanly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
