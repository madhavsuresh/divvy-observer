"""End-to-end: train a MacFlow model, register the artifact, reload, predict."""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import duckdb
import numpy as np
import pandas as pd
import pytest

from divvy import config, db, model_registry
from divvy.macflow_nissm import MacFlowNISSMLite, MacFlowNISSMLiteConfig


def _synthetic_examples(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = datetime(2024, 5, 1)
    rows = []
    for i in range(n):
        cap = 15
        e0 = int(min(cap, rng.choice([0, 0, 1, 2, 3, 4, 5])))
        q0 = int(min(cap, e0 + rng.integers(0, 5)))
        rows.append(
            {
                "station_id": f"s{i % 4}",
                "anchor_ts": base + timedelta(minutes=i),
                "horizon_minutes": int(rng.choice([5, 10, 15, 20])),
                "capacity": cap,
                "num_ebikes_available": e0,
                "num_bikes_available": q0,
                "num_docks_available": cap - q0,
                "has_ebike": int(e0 >= 1),
                "obs_e_depart": float(rng.poisson(0.3)),
                "obs_e_arrive": float(rng.poisson(0.3)),
                "obs_c_depart": float(rng.poisson(0.5)),
                "obs_c_arrive": float(rng.poisson(0.5)),
                "example_weight": 1.0,
                "is_renting": True,
                "is_returning": True,
                "trend_5m": 0.0,
                "trend_10m": 0.0,
                "trend_15m": 0.0,
                "churn_rate": 0.0,
                "station_same_hour_rate": 0.4,
                "nearby_same_hour_rate": 0.4,
                "station_neighbor_same_hour_rate": 0.4,
                "status_age_minutes": 1.0,
                "current_ebikes_clipped": float(e0),
                "current_total_bikes_clipped": float(q0),
                "docks_available_clipped": float(cap - q0),
                "ebike_share_of_bikes": float(e0 / max(q0, 1)),
                "dock_availability_fraction": float((cap - q0) / cap),
                "hour_sin": 0.0,
                "hour_cos": 1.0,
                "month_sin": 0.0,
                "month_cos": 1.0,
                "day_of_year_sin": 0.0,
                "day_of_year_cos": 1.0,
                "is_commute_hour": 0,
                "dow": int(i % 7),
                "is_weekend": 0,
            }
        )
    return pd.DataFrame(rows)


def test_save_artifact_round_trips_with_is_trained_artifact():
    df = _synthetic_examples()
    cfg = MacFlowNISSMLiteConfig(
        epochs=1,
        max_examples=400,
        batch_size_cpu=128,
        device="cpu",
        min_train_examples=50,
        min_positive_examples=5,
        min_zero_future_examples=5,
        min_valid_examples=0,
        calibrate=False,
    )
    model = MacFlowNISSMLite(cfg)
    model.fit(df.head(320), df.tail(80))
    assert model.trained

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch.object(config, "DATA_DIR", Path(tmpdir)):
            conn = duckdb.connect(":memory:")
            try:
                db.init_schema(conn)
                saved = model_registry.save_artifact(
                    conn,
                    "macflow_nissm_lite",
                    model,
                    model_family="macflow_nissm_lite",
                    model_version=model.model_version,
                    feature_columns=list(model.feature_columns),
                    horizons=[5, 10, 15, 20],
                    metrics=dict(model.metrics or {}),
                )
                assert saved["artifact_id"]

                loaded = model_registry.load_latest_artifact(conn, "macflow_nissm_lite")
                assert loaded is not None
                assert model_registry.is_trained_artifact(loaded)
                method = model_registry.artifact_method(loaded).lower()
                assert "bootstrap" not in method
                assert "fallback" not in method
                assert method == "macflow_nissm_lite_trained_v1"
                # The loaded model should be able to predict.
                loaded_model = loaded["model"]
                preds = loaded_model.predict_distribution(df.head(5))
                assert len(preds) == 5
                assert (preds["p_has_ebike"] >= 0).all() and (preds["p_has_ebike"] <= 1).all()
            finally:
                conn.close()
