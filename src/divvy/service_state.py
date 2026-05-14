from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

import duckdb

from . import config, db


WRITE_LOCK_NAME = "duckdb-write"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _json_text(value: Any) -> str:
    if value is None:
        return "{}"
    if isinstance(value, str):
        return value
    return json.dumps(value, separators=(",", ":"), sort_keys=True, default=str)


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def _lock_path(job_name: str) -> Path:
    config.ensure_dirs()
    return config.SERVICE_LOCK_DIR / f"{_safe_name(job_name)}.lock"


def _read_lock(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _is_stale(payload: dict, ttl_seconds: int, now: datetime) -> bool:
    expires_at = payload.get("expires_at")
    if expires_at:
        try:
            return pd_timestamp(expires_at) <= now
        except Exception:
            pass
    acquired_at = payload.get("acquired_at")
    if acquired_at:
        try:
            return pd_timestamp(acquired_at) + timedelta(seconds=ttl_seconds) <= now
        except Exception:
            pass
    return True


def pd_timestamp(value) -> datetime:
    import pandas as pd

    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.to_pydatetime()


@dataclass(frozen=True)
class JobLock:
    job_name: str
    run_id: str
    acquired: bool
    lock_path: str
    reason: str | None = None
    stale_recovered: bool = False

    def as_dict(self) -> dict:
        return {
            "job_name": self.job_name,
            "run_id": self.run_id,
            "acquired": self.acquired,
            "lock_path": self.lock_path,
            "reason": self.reason,
            "stale_recovered": self.stale_recovered,
        }


def acquire_job_lock(
    conn: duckdb.DuckDBPyConnection,
    job_name: str,
    ttl_seconds: int | None = None,
    *,
    run_id: str | None = None,
    metadata: dict | None = None,
) -> dict:
    """Acquire a process-level file lock and mirror it into DuckDB state.

    Returns a dictionary with ``acquired`` and ``run_id`` fields. A stale file
    lock is removed when its expiry is past the configured TTL.
    """
    db.init_schema(conn)
    ttl = int(ttl_seconds or config.JOB_LOCK_TTL_SECONDS)
    now = _utc_now()
    expires = now + timedelta(seconds=ttl)
    run_id = run_id or str(uuid.uuid4())
    path = _lock_path(job_name)
    payload = {
        "job_name": job_name,
        "run_id": run_id,
        "owner_pid": os.getpid(),
        "host": socket.gethostname(),
        "acquired_at": now.isoformat(),
        "expires_at": expires.isoformat(),
        "metadata": metadata or {},
    }
    stale_recovered = False

    while True:
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, separators=(",", ":"), default=str)
            break
        except FileExistsError:
            existing = _read_lock(path)
            if _is_stale(existing, ttl, now):
                try:
                    path.unlink()
                    stale_recovered = True
                    continue
                except FileNotFoundError:
                    continue
            return JobLock(
                job_name=job_name,
                run_id=str(existing.get("run_id") or ""),
                acquired=False,
                lock_path=str(path),
                reason="lock_already_held",
            ).as_dict()

    conn.execute(
        """
        INSERT INTO background_job_locks (
          job_name, run_id, owner_pid, acquired_at, expires_at, heartbeat_at,
          lock_path, stale, metadata_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, false, ?)
        ON CONFLICT (job_name) DO UPDATE SET
          run_id = excluded.run_id,
          owner_pid = excluded.owner_pid,
          acquired_at = excluded.acquired_at,
          expires_at = excluded.expires_at,
          heartbeat_at = excluded.heartbeat_at,
          lock_path = excluded.lock_path,
          stale = false,
          metadata_json = excluded.metadata_json
        """,
        [
            job_name,
            run_id,
            os.getpid(),
            now,
            expires,
            now,
            str(path),
            _json_text(metadata or {}),
        ],
    )
    return JobLock(
        job_name=job_name,
        run_id=run_id,
        acquired=True,
        lock_path=str(path),
        stale_recovered=stale_recovered,
    ).as_dict()


def release_job_lock(
    conn: duckdb.DuckDBPyConnection,
    job_name: str,
    run_id: str | None,
) -> None:
    db.init_schema(conn)
    row = conn.execute(
        "SELECT run_id, lock_path FROM background_job_locks WHERE job_name = ?",
        [job_name],
    ).fetchone()
    if row and (not run_id or str(row[0]) == str(run_id)):
        conn.execute("DELETE FROM background_job_locks WHERE job_name = ?", [job_name])
        path = Path(row[1]) if row[1] else _lock_path(job_name)
        try:
            current = _read_lock(path)
            if not current or not run_id or str(current.get("run_id")) == str(run_id):
                path.unlink(missing_ok=True)
        except OSError:
            pass


def touch_job_lock(
    conn: duckdb.DuckDBPyConnection,
    job_name: str,
    run_id: str,
    ttl_seconds: int | None = None,
) -> None:
    db.init_schema(conn)
    now = _utc_now()
    expires = now + timedelta(seconds=int(ttl_seconds or config.JOB_LOCK_TTL_SECONDS))
    conn.execute(
        """
        UPDATE background_job_locks
        SET heartbeat_at = ?, expires_at = ?
        WHERE job_name = ? AND run_id = ?
        """,
        [now, expires, job_name, run_id],
    )
    path = _lock_path(job_name)
    payload = _read_lock(path)
    if payload.get("run_id") == run_id:
        payload["expires_at"] = expires.isoformat()
        payload["heartbeat_at"] = now.isoformat()
        path.write_text(json.dumps(payload, separators=(",", ":"), default=str), encoding="utf-8")


def record_job_start(
    conn: duckdb.DuckDBPyConnection,
    job_name: str,
    *,
    run_id: str | None = None,
    service_name: str = "automation",
    metadata: dict | None = None,
) -> str:
    db.init_schema(conn)
    run_id = run_id or str(uuid.uuid4())
    conn.execute(
        """
        INSERT OR REPLACE INTO background_job_runs (
          run_id, job_name, service_name, status, started_at, metadata_json
        )
        VALUES (?, ?, ?, 'running', ?, ?)
        """,
        [run_id, job_name, service_name, _utc_now(), _json_text(metadata or {})],
    )
    return run_id


def record_job_success(
    conn: duckdb.DuckDBPyConnection,
    job_name: str,
    run_id: str,
    *,
    message: str | None = None,
    metadata: dict | None = None,
) -> None:
    db.init_schema(conn)
    finished = _utc_now()
    conn.execute(
        """
        UPDATE background_job_runs
        SET status = 'success',
            finished_at = ?,
            duration_seconds = EXTRACT(EPOCH FROM (? - started_at)),
            message = ?,
            metadata_json = ?
        WHERE run_id = ? AND job_name = ?
        """,
        [finished, finished, message, _json_text(metadata or {}), run_id, job_name],
    )


def record_job_failure(
    conn: duckdb.DuckDBPyConnection,
    job_name: str,
    run_id: str,
    *,
    error: str,
    message: str | None = None,
    metadata: dict | None = None,
) -> None:
    db.init_schema(conn)
    finished = _utc_now()
    conn.execute(
        """
        UPDATE background_job_runs
        SET status = 'failure',
            finished_at = ?,
            duration_seconds = EXTRACT(EPOCH FROM (? - started_at)),
            message = ?,
            metadata_json = ?,
            error = ?
        WHERE run_id = ? AND job_name = ?
        """,
        [finished, finished, message, _json_text(metadata or {}), error, run_id, job_name],
    )


def heartbeat(
    conn: duckdb.DuckDBPyConnection,
    service_name: str,
    metadata: dict | None = None,
) -> None:
    db.init_schema(conn)
    conn.execute(
        """
        INSERT INTO service_heartbeats (service_name, heartbeat_at, pid, metadata_json)
        VALUES (?, ?, ?, ?)
        ON CONFLICT (service_name) DO UPDATE SET
          heartbeat_at = excluded.heartbeat_at,
          pid = excluded.pid,
          metadata_json = excluded.metadata_json
        """,
        [service_name, _utc_now(), os.getpid(), _json_text(metadata or {})],
    )


@contextmanager
def job_lock(
    conn: duckdb.DuckDBPyConnection,
    job_name: str,
    ttl_seconds: int | None = None,
    *,
    metadata: dict | None = None,
) -> Iterator[dict]:
    lock = acquire_job_lock(conn, job_name, ttl_seconds, metadata=metadata)
    try:
        yield lock
    finally:
        if lock.get("acquired"):
            release_job_lock(conn, job_name, lock.get("run_id"))


def mark_stale_locks(conn: duckdb.DuckDBPyConnection) -> int:
    db.init_schema(conn)
    now = _utc_now()
    rows = conn.execute(
        """
        SELECT job_name, run_id, lock_path
        FROM background_job_locks
        WHERE expires_at IS NOT NULL AND expires_at < ?
        """,
        [now],
    ).fetchall()
    for job_name, run_id, lock_path in rows:
        conn.execute(
            "UPDATE background_job_locks SET stale = true WHERE job_name = ?",
            [job_name],
        )
        path = Path(lock_path) if lock_path else _lock_path(str(job_name))
        payload = _read_lock(path)
        if not payload or str(payload.get("run_id")) == str(run_id):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
    return len(rows)


def get_watermark(
    conn: duckdb.DuckDBPyConnection,
    key: str,
) -> datetime | None:
    """Return the stored watermark for ``key`` as a UTC-naive datetime, or None."""
    db.init_schema(conn)
    row = conn.execute(
        "SELECT value FROM flow_processing_state WHERE key = ?",
        [key],
    ).fetchone()
    if not row or not row[0]:
        return None
    try:
        return pd_timestamp(row[0])
    except Exception:
        return None


def set_watermark(
    conn: duckdb.DuckDBPyConnection,
    key: str,
    value: datetime,
) -> None:
    """Persist ``value`` (UTC-naive datetime) as the watermark for ``key``."""
    db.init_schema(conn)
    if value.tzinfo is not None:
        value = value.astimezone(timezone.utc).replace(tzinfo=None)
    conn.execute(
        """
        INSERT INTO flow_processing_state (key, value, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT (key) DO UPDATE SET
          value = excluded.value,
          updated_at = excluded.updated_at
        """,
        [key, value.isoformat(), _utc_now()],
    )


def queue_backlog() -> dict:
    pending = config.FORECAST_QUEUE_DIR / "pending"
    processing = config.FORECAST_QUEUE_DIR / "processing"
    failed = config.FORECAST_QUEUE_DIR / "failed"
    return {
        "pending_files": len(list(pending.glob("*.json"))) if pending.exists() else 0,
        "processing_files": len(list(processing.glob("*.json"))) if processing.exists() else 0,
        "failed_files": len(list(failed.glob("*.json"))) if failed.exists() else 0,
    }


def latest_job_status(conn: duckdb.DuckDBPyConnection) -> list[dict]:
    try:
        rows = conn.execute(
            """
            WITH ranked AS (
              SELECT *,
                     ROW_NUMBER() OVER (
                       PARTITION BY job_name, status ORDER BY COALESCE(finished_at, started_at) DESC
                     ) AS rn
              FROM background_job_runs
              WHERE status IN ('success', 'failure', 'running')
            )
            SELECT *
            FROM ranked
            WHERE rn = 1
            ORDER BY job_name, status
            """
        ).df()
    except Exception:
        return []
    return rows.to_dict(orient="records")


def system_status(conn: duckdb.DuckDBPyConnection, *, initialize_schema: bool = False) -> dict:
    if initialize_schema:
        db.init_schema(conn)
    else:
        try:
            db.init_schema(conn)
        except Exception:
            pass
    now = _utc_now()
    payload: dict[str, Any] = {
        "computed_at": now.isoformat(),
        "db_path": str(config.DB_PATH),
        "read_db_path": str(config.READ_DB_PATH),
        "queue": queue_backlog(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
    }
    try:
        payload["heartbeats"] = conn.execute(
            """
            SELECT service_name, heartbeat_at, pid, metadata_json,
                   EXTRACT(EPOCH FROM (? - heartbeat_at)) / 60.0 AS age_minutes
            FROM service_heartbeats
            ORDER BY service_name
            """,
            [now],
        ).df().to_dict(orient="records")
    except Exception:
        payload["heartbeats"] = []
    try:
        payload["locks"] = conn.execute(
            """
            SELECT job_name, run_id, owner_pid, acquired_at, expires_at,
                   heartbeat_at, stale, lock_path
            FROM background_job_locks
            ORDER BY acquired_at DESC
            """
        ).df().to_dict(orient="records")
    except Exception:
        payload["locks"] = []
    try:
        payload["jobs"] = latest_job_status(conn)
    except Exception:
        payload["jobs"] = []
    try:
        latest_cache = conn.execute(
            """
            SELECT
              MAX(as_of) AS latest_as_of,
              COUNT(DISTINCT station_id) AS stations_cached,
              COUNT(*) AS rows_cached
            FROM live_station_predictions
            """
        ).fetchone()
        cache_age = None
        if latest_cache and latest_cache[0]:
            cache_age = (now - latest_cache[0]).total_seconds() / 60.0
        payload["prediction_cache"] = {
            "latest_as_of": latest_cache[0].isoformat() if latest_cache and latest_cache[0] else None,
            "age_minutes": cache_age,
            "stations_cached": int(latest_cache[1] or 0) if latest_cache else 0,
            "rows_cached": int(latest_cache[2] or 0) if latest_cache else 0,
            "status": "fresh" if cache_age is not None and cache_age <= config.CACHE_MAX_AGE_MINUTES else "stale",
        }
    except Exception:
        payload["prediction_cache"] = {"status": "unknown"}
    try:
        freshness = conn.execute(
            """
            SELECT
              (SELECT MAX(last_reported) FROM station_status) AS latest_station_reported,
              (SELECT MAX(fetched_at) FROM station_status) AS latest_station_fetch,
              (SELECT MAX(fetched_at) FROM free_bike_status) AS latest_free_bike_fetch
            """
        ).fetchone()
        payload["data_freshness"] = {
            "latest_station_reported": freshness[0].isoformat() if freshness and freshness[0] else None,
            "latest_station_fetch": freshness[1].isoformat() if freshness and freshness[1] else None,
            "latest_free_bike_fetch": freshness[2].isoformat() if freshness and freshness[2] else None,
        }
    except Exception:
        payload["data_freshness"] = {}
    try:
        disk = shutil.disk_usage(config.DATA_DIR)
        payload["disk"] = {
            "free_gb": round(disk.free / (1024 ** 3), 2),
            "total_gb": round(disk.total / (1024 ** 3), 2),
            "warning": disk.free < 2 * 1024 ** 3,
        }
    except Exception:
        payload["disk"] = {}
    return payload
