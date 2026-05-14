from __future__ import annotations

from datetime import datetime, timedelta, timezone

import duckdb
import pandas as pd

from divvy import db, predictor, recommendations
from divvy.dg_nissm import DGNISSMModel


def test_api_request_path_does_not_build_training_examples(monkeypatch) -> None:
    predictor._MODEL_CACHE.clear()
    monkeypatch.setenv("DIVVY_DISABLE_REQUEST_TRAINING", "1")
    conn = duckdb.connect(":memory:")
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

    monkeypatch.setattr(
        predictor,
        "build_training_examples",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("request training called")),
    )
    monkeypatch.setattr(recommendations, "fetch_live_free_bikes", lambda *args, **kwargs: None)

    payload = recommendations.recommend(conn, lat=41.0, lon=-87.0, log_forecasts=False, log_query=False)
    assert payload["candidate_count"] == 1
    assert payload["model"]["model_warning"] is not None
    conn.close()


def test_runtime_suite_does_not_fit_dg_nissm(monkeypatch) -> None:
    predictor._MODEL_CACHE.clear()
    monkeypatch.setenv("DIVVY_DISABLE_REQUEST_TRAINING", "1")
    conn = duckdb.connect(":memory:")
    db.init_schema(conn)

    def _raise_fit(self, *args, **kwargs):
        raise AssertionError("DG-NISSM request-path fit called")

    monkeypatch.setattr(DGNISSMModel, "fit", _raise_fit)
    suite = predictor.get_availability_model_suite(conn, force=True, ttl_seconds=0)

    assert "dg_nissm" in suite.models
    assert not suite.models["dg_nissm"].usable
    conn.close()


def test_score_candidates_skips_untrained_dg_nissm(monkeypatch) -> None:
    predictor._MODEL_CACHE.clear()
    monkeypatch.setenv("DIVVY_DISABLE_REQUEST_TRAINING", "1")
    conn = duckdb.connect(":memory:")
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
    monkeypatch.setattr(recommendations, "fetch_live_free_bikes", lambda *args, **kwargs: None)

    dg = predictor.FittedAvailabilityModel(
        model=DGNISSMModel(),
        trained_at=base,
        n_examples=0,
        n_positive=0,
        n_negative=0,
        method="dg_nissm_unavailable_no_artifact",
        model_key="dg_nissm",
        label=predictor.MODEL_SPECS["dg_nissm"]["label"],
        model_version=predictor.MODEL_SPECS["dg_nissm"]["version"],
    )
    cc = predictor._runtime_fallback_model("cc_nissm", 20, 10)
    suite = predictor.FittedModelSuite({"dg_nissm": dg, "cc_nissm": cc}, active_key="cc_nissm")
    predictor._MODEL_CACHE["availability_suite"] = suite
    predictor._MODEL_CACHE["availability_suite_ts"] = predictor.time.time()

    candidate = pd.DataFrame([
        {
            "station_id": "s1",
            "name": "Station One",
            "short_name": "S1",
            "capacity": 15,
            "lat": 41.0,
            "lon": -87.0,
            "num_bikes_available": 5,
            "num_ebikes_available": 1,
            "num_docks_available": 10,
            "last_reported": base + timedelta(minutes=19),
            "is_renting": True,
            "is_returning": True,
            "distance_km": 0.0,
        }
    ])
    scored, _ = predictor.score_candidates(conn, candidate, horizons=(5,))

    assert "p_has_ebike_5m" in scored.columns
    assert "p_has_ebike_5m_dg_nissm" in scored.columns
    assert scored["model_method_5m_dg_nissm"].iloc[0] == "dg_nissm_unavailable_no_artifact"
    predictor._MODEL_CACHE.clear()
    conn.close()
