from __future__ import annotations

from datetime import datetime, timedelta, timezone

import duckdb
import pandas as pd

from divvy import db, live_cache, predictor, recommendations


def _seed(conn) -> None:
    db.init_schema(conn)
    base = datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0) - timedelta(minutes=30)
    conn.executemany(
        "INSERT INTO stations VALUES (?, NULL, ?, ?, ?, ?, 15, 'classic', ?, ?)",
        [
            ("s1", "S1", "Station One", 41.0, -87.0, base, base),
            ("s2", "S2", "Station Two", 41.004, -87.004, base, base),
        ],
    )
    rows = []
    for minute in range(31):
        ts = base + timedelta(minutes=minute)
        rows.extend([
            ("s1", ts, ts, 5, 1 if minute > 10 else 0, 0, 10, 0, True, True, True),
            ("s2", ts, ts, 6, 2, 0, 9, 0, True, True, True),
        ])
    conn.executemany("INSERT INTO station_status VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)


def test_live_prediction_cache_refresh_writes_rows_and_recommendation_uses_cache(monkeypatch) -> None:
    predictor._MODEL_CACHE.clear()
    conn = duckdb.connect(":memory:")
    _seed(conn)
    result = live_cache.refresh_live_station_predictions(conn, active_only=True)
    assert result["rows_written"] == 2 * len(predictor.HORIZONS)

    def fail_score(*_args, **_kwargs):
        raise AssertionError("request path should not score when cache is fresh")

    monkeypatch.setattr(predictor, "score_candidates", fail_score)
    monkeypatch.setattr(recommendations, "fetch_live_free_bikes", lambda *args, **kwargs: None)
    payload = recommendations.recommend(conn, lat=41.0, lon=-87.0, log_forecasts=False, log_query=False)
    assert payload["prediction_cache_status"] == "fresh"
    assert payload["best_practical_station_5_10m"]["p_arrival"] is not None
    conn.close()


def test_recommendation_falls_back_to_candidate_scoring_when_cache_stale(monkeypatch) -> None:
    predictor._MODEL_CACHE.clear()
    conn = duckdb.connect(":memory:")
    _seed(conn)
    calls = []
    original = predictor.score_candidates

    def wrapped_score(_conn, candidates: pd.DataFrame, *args, **kwargs):
        calls.append(len(candidates))
        return original(_conn, candidates, *args, **kwargs)

    monkeypatch.setattr(predictor, "score_candidates", wrapped_score)
    monkeypatch.setattr(recommendations.config, "CACHE_MAX_AGE_MINUTES", -1)
    monkeypatch.setattr(recommendations, "fetch_live_free_bikes", lambda *args, **kwargs: None)

    payload = recommendations.recommend(conn, lat=41.0, lon=-87.0, log_forecasts=False, log_query=False)
    assert payload["prediction_cache_status"] in {"miss_fallback", "stale_fallback"}
    assert calls and calls[0] == payload["candidate_count"]
    conn.close()
