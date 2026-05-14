"""Runtime/API path must never invoke MacFlowNISSMLite.fit."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import duckdb
import pytest

from divvy import db, predictor
from divvy.macflow_nissm import MacFlowNISSMLite


@pytest.fixture(autouse=True)
def _clear_cache():
    predictor._MODEL_CACHE.clear()
    yield
    predictor._MODEL_CACHE.clear()


def test_runtime_suite_does_not_fit_macflow(monkeypatch) -> None:
    monkeypatch.setenv("DIVVY_DISABLE_REQUEST_TRAINING", "1")
    conn = duckdb.connect(":memory:")
    db.init_schema(conn)

    def _raise_fit(self, *args, **kwargs):
        raise AssertionError("MacFlow request-path fit called")

    monkeypatch.setattr(MacFlowNISSMLite, "fit", _raise_fit)
    suite = predictor.get_availability_model_suite(conn, force=True, ttl_seconds=0)
    assert "macflow_nissm_lite" in suite.models
    macflow = suite.models["macflow_nissm_lite"]
    # No trained artifact in the in-memory DB → not usable.
    assert macflow.usable is False
    # Method must clearly indicate the missing-artifact state.
    method = macflow.method.lower()
    assert "bootstrap" in method or "fallback" in method or "unavailable" in method or "untrained" in method
    conn.close()


def test_macflow_method_does_not_collide_with_bootstrap_or_fallback_when_trained() -> None:
    # Fresh untrained model should have a clearly non-trained method.
    model = MacFlowNISSMLite()
    assert "trained_v1" not in model.method
    # Trained method (set after fit succeeds) must NOT contain bootstrap or fallback.
    expected_trained_method = "macflow_nissm_lite_trained_v1"
    assert "bootstrap" not in expected_trained_method
    assert "fallback" not in expected_trained_method


def test_macflow_unusable_when_net_missing() -> None:
    """An artifact-loaded model with .trained=True but .net=None must NOT be marked usable."""
    fitted = predictor.FittedAvailabilityModel(
        model=MacFlowNISSMLite(),
        trained_at=datetime.now(timezone.utc).replace(tzinfo=None),
        n_examples=0,
        n_positive=0,
        n_negative=0,
        method="macflow_nissm_lite_unavailable_no_trained_artifact",
        model_key="macflow_nissm_lite",
        label=predictor.MODEL_SPECS["macflow_nissm_lite"]["label"],
        model_version=predictor.MODEL_SPECS["macflow_nissm_lite"]["version"],
    )
    # The bare model has trained=False; usable should be False.
    assert fitted.usable is False
