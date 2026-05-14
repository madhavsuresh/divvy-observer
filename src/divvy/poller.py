from __future__ import annotations

import logging
import signal
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import requests

from . import config, db, forecast_queue, model_eval, service_state, tile

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
    consecutive_failures: int = 0
    rows_inserted_total: int = 0


def process_prediction_writes(conn, state: PollerState, now: float) -> dict:
    """Drain prediction-service writes while the collector owns DuckDB."""
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


def insert_free_bikes(conn, payload: dict[str, Any]) -> int:
    """Persist free-floating bike positions, deduping unchanged rows.

    Each tick, GBFS reports ~2,500 bikes. Most haven't moved since the last
    tick — writing every one would balloon storage. We compare each bike's
    current (lat, lon, is_reserved, is_disabled) against its most recent
    row and only insert when something changed (or the bike is new).
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
        SELECT bike_id, lat, lon, is_reserved, is_disabled
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
        bid: (lat, lon, bool(res), bool(dis))
        for bid, lat, lon, res, dis in last_rows
    }

    rows_to_insert = []
    for b in incoming:
        prev = last_state.get(b["bike_id"])
        curr = (b["lat"], b["lon"], b["is_reserved"], b["is_disabled"])
        if prev == curr:
            continue  # unchanged — skip
        rows_to_insert.append((
            b["bike_id"], fetched_at, b["name"],
            b["lat"], b["lon"], b["is_reserved"], b["is_disabled"],
            tile.tile_id_for(b["lat"], b["lon"]),
        ))

    if not rows_to_insert:
        return 0

    conn.executemany(
        """
        INSERT OR IGNORE INTO free_bike_status
          (bike_id, fetched_at, name, lat, lon, is_reserved, is_disabled, tile_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows_to_insert,
    )
    return len(rows_to_insert)


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

    info_payload = _fetch(config.STATION_INFO_URL) if need_info else None
    status_payload = _fetch(config.STATION_STATUS_URL)
    free_payload = _fetch(config.FREE_BIKE_STATUS_URL)

    with db.session(read_only=False) as conn:
        db.init_schema(conn)
        if info_payload is not None:
            n_stations = upsert_stations(conn, info_payload)
            state.last_info_refresh_unix = now
            log.info("refreshed station_information: %d stations", n_stations)
        inserted = insert_status(conn, status_payload)
        free_inserted = insert_free_bikes(conn, free_payload)
        prediction_summary = process_prediction_writes(conn, state, now)
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
        "%d station rows + %d free-bike events (total=%d)",
        len(status_payload["data"]["stations"]),
        len(free_payload["data"]["bikes"]),
        inserted,
        free_inserted,
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
