from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb

from divvy import config, db, service_state


def test_job_locks_prevent_duplicates_and_recover_stale(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config, "SERVICE_LOCK_DIR", tmp_path / "locks")
    conn = duckdb.connect(":memory:")
    db.init_schema(conn)

    first = service_state.acquire_job_lock(conn, "refresh-live-predictions", ttl_seconds=60)
    second = service_state.acquire_job_lock(conn, "refresh-live-predictions", ttl_seconds=60)

    assert first["acquired"] is True
    assert second["acquired"] is False

    path = Path(first["lock_path"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["expires_at"] = (datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=5)).isoformat()
    path.write_text(json.dumps(payload), encoding="utf-8")

    recovered = service_state.acquire_job_lock(conn, "refresh-live-predictions", ttl_seconds=1)
    assert recovered["acquired"] is True
    assert recovered["stale_recovered"] is True

    service_state.release_job_lock(conn, "refresh-live-predictions", recovered["run_id"])
    assert not path.exists()
