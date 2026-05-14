from __future__ import annotations

import duckdb
from datetime import datetime, timedelta, timezone

from divvy import db, predictor, recommendations
from divvy import config
from divvy import dashboard


def _seed_db(path) -> None:
    predictor._MODEL_CACHE.clear()
    conn = duckdb.connect(str(path))
    db.init_schema(conn)
    base = datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0) - timedelta(minutes=35)
    conn.executemany(
        "INSERT INTO stations VALUES (?, NULL, ?, ?, ?, ?, 15, 'classic', ?, ?)",
        [
            ("s1", "S1", "Station One", 41.0, -87.0, base, base),
            ("s2", "S2", "Station Two", 41.004, -87.004, base, base),
        ],
    )
    rows = []
    for minute in range(36):
        ts = base + timedelta(minutes=minute)
        rows.extend([
            ("s1", ts, ts, 5, 1 if minute > 10 else 0, 0, 10, 0, True, True, True),
            ("s2", ts, ts, 6, 2, 0, 9, 0, True, True, True),
        ])
    conn.executemany("INSERT INTO station_status VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    conn.close()


def test_recommendation_payload_has_model_status_and_arrival_probability(tmp_path, monkeypatch) -> None:
    path = tmp_path / "payload.duckdb"
    _seed_db(path)
    monkeypatch.setattr(
        recommendations,
        "fetch_live_free_bikes",
        lambda *args, **kwargs: None,
    )
    conn = duckdb.connect(str(path))
    payload = recommendations.recommend(conn, lat=41.0, lon=-87.0, log_forecasts=False, log_query=False)
    conn.close()

    station = payload["best_practical_station_5_10m"]
    assert payload["active_model_key"] in predictor.SOTA_PRIMARY_MODEL_KEYS
    assert payload["model"]["active_model_source"] in {
        "cold_start_sota_bootstrap",
        "best_sota_recent_performance",
        "forced_env",
    }
    assert station["p_arrival"] is not None
    assert station["arrival_time_minutes"] is not None
    assert "active_model_key" in station


def test_dashboard_status_payload_contains_active_model_and_cache(tmp_path, monkeypatch) -> None:
    path = tmp_path / "dashboard.duckdb"
    _seed_db(path)
    monkeypatch.setattr(config, "DB_PATH", path)
    monkeypatch.setattr(config, "READ_DB_PATH", tmp_path / "read.duckdb")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    conn = duckdb.connect(str(path))
    conn.execute(
        """
        INSERT INTO model_selection_state (
          computed_at, active_model_key, active_artifact_id, active_model_source,
          best_evaluated_model_key, best_sota_model_key, best_baseline_model_key,
          active_equals_best, selection_metric, selection_window_hours,
          min_resolved, reason, metrics_json
        )
        VALUES (now(), 'cc_nissm', 'artifact-1', 'default_cc_nissm',
                'cc_nissm', 'cc_nissm', NULL, true, 'decision_rank_loss',
                168, 100, 'active_is_best_sota', '{}')
        """
    )
    conn.close()
    dashboard._system_status_payload.clear()

    payload = dashboard._system_status_payload()

    assert payload["selection"]["active_model_key"] == "cc_nissm"
    assert "cache" in payload
