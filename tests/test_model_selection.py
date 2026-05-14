from __future__ import annotations

import duckdb
import pandas as pd

from divvy import config, db, model_registry, model_selection
from divvy.cc_nissm import CCNISSMModel


def test_model_selection_keeps_active_when_resolved_outcomes_insufficient(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    conn = duckdb.connect(":memory:")
    db.init_schema(conn)
    model = CCNISSMModel().fit(pd.DataFrame({"has_ebike": [0, 1, 1]}))
    artifact = model_registry.save_artifact(conn, "cc_nissm", model, {"is_active": True}, {"n_train": 3})

    state = model_selection.select_primary_driver(conn, window_hours=168, min_resolved=100)

    assert state["active_model_key"] == "cc_nissm"
    assert state["active_artifact_id"] == artifact["artifact_id"]
    assert state["best_sota_model_key"] is None
    assert state["reason"] == "insufficient_resolved_outcomes"
    persisted = conn.execute("SELECT COUNT(*) FROM model_selection_state").fetchone()[0]
    assert persisted == 1
    conn.close()
