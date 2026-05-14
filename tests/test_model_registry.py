from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from divvy import config, db, model_registry
from divvy.cc_nissm import CCNISSMModel


def test_model_registry_round_trips_artifact(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    conn = duckdb.connect(":memory:")
    db.init_schema(conn)
    model = CCNISSMModel().fit(pd.DataFrame({"has_ebike": [0, 1]}))

    saved = model_registry.save_artifact(conn, "cc_nissm", model, {"is_active": True}, {"n_train": 2})
    loaded = model_registry.load_latest_artifact(conn, "cc_nissm")

    assert saved["artifact_id"]
    assert loaded is not None
    assert loaded["model"].predict_distribution(pd.DataFrame([{"capacity": 5, "num_ebikes_available": 1, "num_bikes_available": 2, "horizon_minutes": 5}])).iloc[0]["p_has_ebike"] > 0


def test_model_registry_rejects_bootstrap_activation(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    conn = duckdb.connect(":memory:")
    db.init_schema(conn)
    saved = model_registry.save_artifact(
        conn,
        "cc_nissm",
        CCNISSMModel(),
        {"is_active": False},
        {"n_train": 0},
    )

    with pytest.raises(ValueError):
        model_registry.set_active_artifact(conn, saved["artifact_id"])
