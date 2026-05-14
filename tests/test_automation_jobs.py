from __future__ import annotations

from datetime import datetime, timedelta, timezone

import duckdb

from divvy import automation, config, db, predictor


def _seed(path) -> None:
    conn = duckdb.connect(str(path))
    db.init_schema(conn)
    base = datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0) - timedelta(minutes=20)
    conn.executemany(
        "INSERT INTO stations VALUES (?, NULL, ?, ?, ?, ?, 15, 'classic', ?, ?)",
        [("s1", "S1", "Station One", 41.0, -87.0, base, base)],
    )
    conn.executemany(
        "INSERT INTO station_status VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [("s1", base + timedelta(minutes=i), base + timedelta(minutes=i), 5, 1, 0, 10, 0, True, True, True) for i in range(20)],
    )
    conn.close()


def test_automation_run_once_refresh_live_predictions(tmp_path, monkeypatch) -> None:
    predictor._MODEL_CACHE.clear()
    path = tmp_path / "divvy.duckdb"
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "DB_PATH", path)
    monkeypatch.setattr(config, "READ_DB_PATH", tmp_path / "read.duckdb")
    monkeypatch.setattr(config, "FORECAST_QUEUE_DIR", tmp_path / "queue")
    monkeypatch.setattr(config, "SERVICE_LOCK_DIR", tmp_path / "locks")
    monkeypatch.setattr(config, "LOG_DIR", tmp_path / "logs")
    _seed(path)

    result = automation.run_once("refresh-live-predictions")

    assert result["status"] == "success"
    conn = duckdb.connect(str(path))
    rows = conn.execute("SELECT COUNT(*) FROM live_station_predictions").fetchone()[0]
    run = conn.execute("SELECT status FROM background_job_runs WHERE job_name = 'refresh-live-predictions'").fetchone()
    conn.close()
    assert rows == len(predictor.HORIZONS)
    assert run == ("success",)
