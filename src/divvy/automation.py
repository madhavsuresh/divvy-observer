from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Callable

import requests

from . import (
    config,
    db,
    dynamic_graph,
    forecast_queue,
    inferred_flows,
    launchd,
    live_cache,
    live_inflight,
    model_eval,
    model_registry,
    model_selection,
    scheduler,
    service_state,
    train_sota,
    tripdata,
)


log = logging.getLogger("divvy.automation")


WRITE_JOBS = {
    "drain-forecast-queue",
    "resolve-outcomes",
    "snapshot-metrics",
    "refresh-live-predictions",
    "refresh-comparison-predictions",
    "refresh-inflight",
    "refresh-inferred-flows",
    "select-model",
    "cleanup",
    "train-nightly",
    "train-weekly",
    "refresh-graphs",
    "sync-tripdata",
    "retrain-macflow",
}


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _json_print(payload: dict) -> None:
    print(json.dumps(payload, indent=2, default=str))


def _job_drain_forecast_queue() -> dict:
    with db.session(read_only=False) as conn:
        db.init_schema(conn)
        return forecast_queue.drain_forecast_queue(conn, limit=config.FORECAST_QUEUE_DRAIN_LIMIT)


def _job_resolve_outcomes() -> dict:
    with db.session(read_only=False) as conn:
        db.init_schema(conn)
        resolved = model_eval.resolve_due_outcomes(conn)
        return {"outcomes_resolved": int(resolved)}


def _job_snapshot_metrics() -> dict:
    with db.session(read_only=False) as conn:
        db.init_schema(conn)
        rows = model_eval.snapshot_metrics(conn, window_hours=24)
        selection = model_selection.select_primary_driver(conn)
        return {"metrics_rows": int(rows), "selection": selection}


def _job_refresh_live_predictions() -> dict:
    return live_cache.refresh_live_station_predictions_coexisting(active_only=True)


def _job_refresh_comparison_predictions() -> dict:
    return live_cache.refresh_live_station_predictions_coexisting(active_only=False)


def _job_refresh_inflight() -> dict:
    with db.session(read_only=False) as conn:
        db.init_schema(conn)
        return live_inflight.update_live_inflight_arrivals(conn)


def _job_refresh_inferred_flows() -> dict:
    with db.session(read_only=False) as conn:
        db.init_schema(conn)
        return inferred_flows.run_incremental(conn)


def _job_select_model() -> dict:
    with db.session(read_only=False) as conn:
        db.init_schema(conn)
        return model_selection.select_primary_driver(conn)


def _job_cleanup() -> dict:
    with db.session(read_only=False) as conn:
        db.init_schema(conn)
        stale = service_state.mark_stale_locks(conn)
        cutoff = _utc_now() - timedelta(hours=config.LIVE_PREDICTION_RETENTION_HOURS)
        deleted = conn.execute(
            "DELETE FROM live_station_predictions WHERE as_of < ?",
            [cutoff],
        ).rowcount
        return {"stale_locks_recovered": int(stale), "old_cache_rows_deleted": int(deleted or 0)}


def _job_train_nightly() -> dict:
    return train_sota.run_nightly()


def _job_train_weekly() -> dict:
    result = train_sota.run_weekly()
    graph = _job_refresh_graphs()
    return {"training": result, "graph": graph}


def _job_refresh_graphs() -> dict:
    with db.session(read_only=False) as conn:
        db.init_schema(conn)
        return dynamic_graph.refresh_dynamic_graph_cache(conn, lookback_days=30, top_k=16)


def _job_sync_tripdata() -> dict:
    """Pull the most recent few months of Divvy trip data, idempotently.

    Divvy publishes the previous month's data around the 10th. Trying the
    last 3 completed months on a daily cadence guarantees we pick up new
    months within ~24h of release without re-downloading anything we
    already have.

    Months that 404 (S3 hasn't published them yet) are skipped silently.
    The flow-tables rebuild is deferred to ``rebuild_flow_tables`` only when
    we actually inserted new months, to avoid the O(divvy_trips) cost on
    no-op days.
    """
    with db.session(read_only=False) as conn:
        db.init_schema(conn)
        already = tripdata.months_already_loaded(conn)
        attempted: list[dict] = []
        new_results = []
        for year, month in tripdata.completed_months(3):
            if (year, month) in already:
                attempted.append({"month": f"{year}-{month:02d}", "status": "already_loaded"})
                continue
            try:
                result = tripdata.download_month(conn, year, month)
            except requests.HTTPError as exc:
                status = getattr(exc.response, "status_code", None)
                if status == 404:
                    attempted.append({"month": f"{year}-{month:02d}", "status": "missing_on_s3"})
                    continue
                raise
            new_results.append(result)
            attempted.append({
                "month": f"{year}-{month:02d}",
                "status": "downloaded",
                "rows_inserted": result.rows_inserted,
                "rows_seen": result.rows_seen,
            })
        if new_results:
            flow = tripdata.rebuild_flow_tables(conn)
            return {
                "attempted": attempted,
                "rows_inserted": sum(r.rows_inserted for r in new_results),
                "flow_rows": flow.flow_rows,
                "route_rows": flow.route_rows,
            }
        return {"attempted": attempted, "rows_inserted": 0}


def _job_retrain_macflow() -> dict:
    """Retrain only macflow_nissm_lite on the current data.

    Lighter than train-nightly (which does all 5 SOTA models). Useful when
    a tripdata sync has just landed new historical months and we want
    macflow's flow-derived features to reflect that without paying the
    full nightly cost.
    """
    args = argparse.Namespace(
        # train_sota.train_single dispatches off args.command via
        # _single_command_to_key — the value must be the hyphenated alias.
        command="macflow-nissm-lite",
        history_hours=24 * 60,
        valid_hours=24 * 7,
        anchor_every_min=config.TRAIN_ANCHOR_EVERY_MIN,
        horizons=list(__import__("divvy.predictor", fromlist=["HORIZONS"]).HORIZONS),
        max_source_rows=2_000_000,
        device="auto",
        register=True,
        activate=None,
        activate_best_sota=False,
        coexist_live=True,
        time_budget_hours=4.0,
        strict=False,
        epochs=8,
        batch_size=4096,
        max_examples=600_000,
        hidden_dim=128,
        station_embedding_dim=32,
        seq_len=24,
        seq_step_minutes=2,
        top_k=16,
        lr=1e-3,
        weight_decay=1e-4,
        seed=42,
        no_sequence=False,
        no_graph=False,
        calibrate=True,
        benchmark_runtime=True,
        stg_max_examples=None,
        stg_epochs=None,
        stg_batch_size=None,
        partition_mode="full",
    )
    return train_sota.train_single(args)


JOBS: dict[str, Callable[[], dict]] = {
    "drain-forecast-queue": _job_drain_forecast_queue,
    "resolve-outcomes": _job_resolve_outcomes,
    "snapshot-metrics": _job_snapshot_metrics,
    "refresh-live-predictions": _job_refresh_live_predictions,
    "refresh-comparison-predictions": _job_refresh_comparison_predictions,
    "refresh-inflight": _job_refresh_inflight,
    "refresh-inferred-flows": _job_refresh_inferred_flows,
    "select-model": _job_select_model,
    "cleanup": _job_cleanup,
    "train-nightly": _job_train_nightly,
    "train-weekly": _job_train_weekly,
    "refresh-graphs": _job_refresh_graphs,
    "sync-tripdata": _job_sync_tripdata,
    "retrain-macflow": _job_retrain_macflow,
}


ALIASES = {
    "drain-queue": "drain-forecast-queue",
    "metrics": "snapshot-metrics",
    "refresh-live-cache": "refresh-live-predictions",
    "refresh-graphs": "refresh-graphs",
}


def normalize_job_name(job_name: str) -> str:
    job = ALIASES.get(job_name, job_name)
    if job not in JOBS:
        raise KeyError(f"Unknown automation job: {job_name}")
    return job


def _acquire_locks(job_name: str) -> tuple[str | None, str | None, dict]:
    with db.session(read_only=False) as conn:
        db.init_schema(conn)
        service_state.mark_stale_locks(conn)
        job_lock = service_state.acquire_job_lock(
            conn,
            job_name,
            ttl_seconds=config.JOB_LOCK_TTL_SECONDS,
            metadata={"service": "automation"},
        )
        if not job_lock.get("acquired"):
            return None, None, {"status": "skipped", "reason": "job_lock_held", "lock": job_lock}
        write_run_id = None
        if job_name in WRITE_JOBS:
            write_lock = service_state.acquire_job_lock(
                conn,
                service_state.WRITE_LOCK_NAME,
                ttl_seconds=config.JOB_LOCK_TTL_SECONDS,
                metadata={"job_name": job_name},
            )
            if not write_lock.get("acquired"):
                service_state.release_job_lock(conn, job_name, job_lock.get("run_id"))
                return None, None, {"status": "skipped", "reason": "write_lock_held", "lock": write_lock}
            write_run_id = str(write_lock["run_id"])
        run_id = service_state.record_job_start(
            conn,
            job_name,
            run_id=str(job_lock["run_id"]),
            service_name="automation",
            metadata={"write_locked": bool(write_run_id)},
        )
        return run_id, write_run_id, {"status": "acquired"}


def _release_locks(job_name: str, run_id: str | None, write_run_id: str | None) -> None:
    with db.session(read_only=False) as conn:
        db.init_schema(conn)
        if write_run_id:
            service_state.release_job_lock(conn, service_state.WRITE_LOCK_NAME, write_run_id)
        if run_id:
            service_state.release_job_lock(conn, job_name, run_id)


def run_once(job_name: str) -> dict:
    job = normalize_job_name(job_name)
    run_id, write_run_id, lock_state = _acquire_locks(job)
    if not run_id:
        return {"job": job, **lock_state}
    try:
        log.info("starting job %s", job)
        result = JOBS[job]()
        with db.session(read_only=False) as conn:
            db.init_schema(conn)
            service_state.record_job_success(
                conn,
                job,
                run_id,
                message="ok",
                metadata=result,
            )
            service_state.heartbeat(conn, "divvy.automation", {"last_job": job})
            try:
                db.refresh_read_replica()
            except Exception:
                log.exception("read replica refresh failed")
        log.info("finished job %s", job)
        return {"job": job, "run_id": run_id, "status": "success", "result": result}
    except Exception as exc:
        with db.session(read_only=False) as conn:
            db.init_schema(conn)
            service_state.record_job_failure(
                conn,
                job,
                run_id,
                error=str(exc),
                message="job failed",
            )
        log.exception("job %s failed: %s", job, exc)
        return {"job": job, "run_id": run_id, "status": "failure", "error": str(exc)}
    finally:
        _release_locks(job, run_id, write_run_id)


def status_payload() -> dict:
    try:
        with db.session(read_only=True, retries=2, retry_sleep=0.1) as conn:
            return service_state.system_status(conn, initialize_schema=False)
    except Exception as exc:
        return {
            "computed_at": _utc_now().isoformat(),
            "db_path": str(config.DB_PATH),
            "read_db_path": str(config.READ_DB_PATH),
            "queue": service_state.queue_backlog(),
            "status": "degraded",
            "error": str(exc),
        }


def health_payload() -> dict:
    try:
        conn_ctx = db.session(read_only=True, retries=2, retry_sleep=0.1)
        conn = conn_ctx.__enter__()
    except Exception as exc:
        status = status_payload()
        return {
            "ok": False,
            "checks": {
                "status_readable": False,
                "prediction_cache_fresh": False,
                "queue_backlog_ok": int((status.get("queue") or {}).get("pending_files") or 0) < 1000,
            },
            "status": status,
            "active_artifact": {"ok": False, "error": str(exc)},
        }
    try:
        status = service_state.system_status(conn)
        active_artifact = model_registry.load_active_artifact(conn)
        artifact_health = model_registry.artifact_health(
            conn,
            (active_artifact or {}).get("artifact_id"),
        )
        cache = status.get("prediction_cache") or {}
        queue = status.get("queue") or {}
        checks = {
            "prediction_cache_fresh": cache.get("status") == "fresh",
            "queue_backlog_ok": int(queue.get("pending_files") or 0) < 1000,
            "active_artifact_ok": bool(artifact_health.get("ok")) if active_artifact else True,
            "disk_ok": not bool((status.get("disk") or {}).get("warning")),
            "no_stale_locks": not any(lock.get("stale") for lock in status.get("locks") or []),
        }
        return {
            "ok": all(checks.values()),
            "checks": checks,
            "status": status,
            "active_artifact": artifact_health,
        }
    finally:
        conn_ctx.__exit__(None, None, None)


def run_supervisor(max_iterations: int | None = None, sleep_seconds: float = 1.0) -> int:
    config.ensure_dirs()
    sched = scheduler.default_scheduler()
    startup_monotonic = time.monotonic()
    for job in sched.jobs:
        if job.name in {"refresh-comparison-predictions", "snapshot-metrics", "select-model", "cleanup"}:
            job.last_started_monotonic = startup_monotonic
    running = True

    def _stop(signum, _frame):
        nonlocal running
        log.info("received signal %s, stopping automation supervisor", signum)
        running = False

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    with db.session(read_only=False) as conn:
        db.init_schema(conn)
        service_state.heartbeat(conn, "divvy.automation", {"state": "starting"})

    iterations = 0
    log.info("divvy automation supervisor starting")
    while running:
        now_monotonic = time.monotonic()
        now_utc = datetime.now(timezone.utc)
        due = sched.due_jobs(now_monotonic, now_utc)
        for job in due:
            job.mark_started(now_monotonic, now_utc)
            result = run_once(job.name)
            if result.get("status") == "skipped":
                log.info("skipped job %s: %s", job.name, result.get("reason"))
        with db.session(read_only=False) as conn:
            db.init_schema(conn)
            service_state.heartbeat(conn, "divvy.automation", {"due_jobs": [job.name for job in due]})
        iterations += 1
        if max_iterations is not None and iterations >= max_iterations:
            break
        time.sleep(max(0.2, float(sleep_seconds)))
    log.info("divvy automation supervisor stopped")
    return 0


def main(argv: list[str] | None = None) -> int:
    _setup_logging()
    parser = argparse.ArgumentParser(description="Divvy set-and-forget automation supervisor")
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run")
    run_p.add_argument("--max-iterations", type=int)
    run_p.add_argument("--sleep-seconds", type=float, default=1.0)
    once_p = sub.add_parser("once")
    once_p.add_argument("--job", required=True)
    sub.add_parser("status")
    sub.add_parser("health")
    start_p = sub.add_parser("start", help="Install, load, and kickstart all Divvy LaunchAgent services")
    start_p.add_argument("--no-dashboard", action="store_true", help="Do not install/start the Streamlit dashboard service")
    stop_p = sub.add_parser("stop", help="Stop all Divvy LaunchAgent services")
    stop_p.add_argument("--uninstall", action="store_true", help="Also remove generated Divvy LaunchAgent plist files")
    restart_p = sub.add_parser("restart", help="Restart all Divvy LaunchAgent services")
    restart_p.add_argument("--no-dashboard", action="store_true", help="Do not install/start the Streamlit dashboard service")
    sub.add_parser("install-launchd")
    sub.add_parser("uninstall-launchd")
    sub.add_parser("launchd-status")
    args = parser.parse_args(argv)

    if args.command == "run":
        return run_supervisor(max_iterations=args.max_iterations, sleep_seconds=args.sleep_seconds)
    if args.command == "once":
        _json_print(run_once(args.job))
        return 0
    if args.command == "status":
        _json_print(status_payload())
        return 0
    if args.command == "health":
        _json_print(health_payload())
        return 0
    if args.command == "start":
        _json_print(launchd.start_launchd(enable_dashboard=not args.no_dashboard))
        return 0
    if args.command == "stop":
        _json_print(launchd.uninstall_launchd() if args.uninstall else launchd.stop_launchd())
        return 0
    if args.command == "restart":
        _json_print(launchd.restart_launchd(enable_dashboard=not args.no_dashboard))
        return 0
    if args.command == "install-launchd":
        _json_print(launchd.install_launchd())
        return 0
    if args.command == "uninstall-launchd":
        _json_print(launchd.uninstall_launchd())
        return 0
    if args.command == "launchd-status":
        _json_print(launchd.launchd_status())
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
