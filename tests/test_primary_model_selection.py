from __future__ import annotations

import duckdb

from divvy import db, predictor


def test_primary_selection_defaults_to_cc_nissm_not_legacy() -> None:
    conn = duckdb.connect(":memory:")
    db.init_schema(conn)
    suite = predictor.FittedModelSuite(
        {
            "cc_nissm": predictor.FittedAvailabilityModel(object(), predictor._utc_now(), 0, 0, 0, "x", "cc_nissm", predictor.MODEL_SPECS["cc_nissm"]["label"], "v"),
            "inventory_world": predictor.FittedAvailabilityModel(object(), predictor._utc_now(), 0, 0, 0, "x", "inventory_world", predictor.MODEL_SPECS["inventory_world"]["label"], "v"),
        }
    )

    active = predictor.resolve_active_model_key(conn, suite)

    assert active == "cc_nissm"
    assert suite.active_source == "cold_start_sota_bootstrap"
