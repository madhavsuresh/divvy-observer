from __future__ import annotations

import math
import os
import pickle
import tempfile
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from divvy.cdg_nmip import CDG_DEBUG_COLUMNS, CDG_DIAGNOSTIC_COLUMNS, CDG_REQUIRED_OUTPUT_COLUMNS
from divvy.macflow_nissm import (
    MACFLOW_NUMERIC_COLUMNS,
    MacFlowNISSMLite,
    MacFlowNISSMLiteConfig,
)
from divvy.mobility_partitions import Partition, ROLE_TO_INT


def _synthetic_examples(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = datetime(2024, 5, 1, 12, 0)
    rows = []
    for i in range(n):
        sid = f"s{i % 6}"
        horizon = (i % 4 + 1) * 5
        cap = 15
        e0 = int(min(cap, rng.poisson(2)))
        q0 = int(min(cap, e0 + rng.integers(0, 5)))
        rows.append(
            {
                "station_id": sid,
                "anchor_ts": base + timedelta(minutes=i),
                "horizon_minutes": horizon,
                "capacity": cap,
                "num_ebikes_available": e0,
                "num_bikes_available": q0,
                "num_docks_available": max(0, cap - q0),
                "has_ebike": int(rng.random() > 0.45),
                "obs_e_depart": float(rng.poisson(0.3)),
                "obs_e_arrive": float(rng.poisson(0.3)),
                "obs_c_depart": float(rng.poisson(0.5)),
                "obs_c_arrive": float(rng.poisson(0.5)),
                "example_weight": 1.0,
                "is_renting": True,
                "is_returning": True,
                "hour_sin": float(np.sin(2 * np.pi * (i % 24) / 24.0)),
                "hour_cos": float(np.cos(2 * np.pi * (i % 24) / 24.0)),
                "dow": int(i % 7),
                "is_weekend": int((i % 7) >= 5),
                "trend_5m": float(rng.normal(0, 1)),
                "trend_10m": float(rng.normal(0, 1)),
                "trend_15m": float(rng.normal(0, 1)),
                "churn_rate": float(rng.normal(0, 1)),
                "station_same_hour_rate": float(rng.uniform(0, 1)),
                "nearby_same_hour_rate": float(rng.uniform(0, 1)),
                "station_neighbor_same_hour_rate": float(rng.uniform(0, 1)),
                "status_age_minutes": float(rng.uniform(0, 5)),
                "current_ebikes_clipped": float(e0),
                "current_total_bikes_clipped": float(q0),
                "docks_available_clipped": float(cap - q0),
                "ebike_share_of_bikes": float(e0 / max(q0, 1)),
                "dock_availability_fraction": float((cap - q0) / cap),
                "month_sin": float(np.sin(2 * np.pi * 5 / 12.0)),
                "month_cos": float(np.cos(2 * np.pi * 5 / 12.0)),
                "day_of_year_sin": 0.0,
                "day_of_year_cos": 0.0,
                "is_commute_hour": 0,
            }
        )
    return pd.DataFrame(rows)


def _fit_quick(df: pd.DataFrame) -> MacFlowNISSMLite:
    config = MacFlowNISSMLiteConfig(
        epochs=2,
        max_examples=2000,
        batch_size_cpu=128,
        device="cpu",
        min_train_examples=50,
        min_positive_examples=10,
        min_zero_future_examples=10,
        min_valid_examples=10,
        calibrate=False,
    )
    model = MacFlowNISSMLite(config)
    train = df.head(int(len(df) * 0.8))
    valid = df.tail(len(df) - len(train))
    model.fit(train, valid)
    return model


def test_fit_sets_trained_method_and_metrics():
    df = _synthetic_examples()
    model = _fit_quick(df)
    assert model.trained is True
    assert model.method == "macflow_nissm_lite_trained_v1"
    assert "bootstrap" not in model.method
    assert "fallback" not in model.method
    assert model.metrics.get("n_train", 0) >= 1
    assert isinstance(model.metrics.get("brier_score"), float)


def test_predict_distribution_has_required_columns():
    df = _synthetic_examples()
    model = _fit_quick(df)
    rows = df.head(10)
    out = model.predict_distribution(rows)
    for col in CDG_REQUIRED_OUTPUT_COLUMNS:
        assert col in out.columns, f"missing {col}"
    for col in CDG_DIAGNOSTIC_COLUMNS:
        assert col in out.columns, f"missing diagnostic {col}"


def test_predict_distribution_debug_has_debug_columns():
    df = _synthetic_examples()
    model = _fit_quick(df)
    rows = df.head(5)
    out = model.predict_distribution(rows, debug=True)
    for col in CDG_DEBUG_COLUMNS:
        assert col in out.columns, f"missing debug column {col}"


def test_predict_distribution_probabilities_are_in_unit_interval():
    df = _synthetic_examples()
    model = _fit_quick(df)
    rows = df.head(50)
    out = model.predict_distribution(rows)
    assert (out["p_has_ebike"] >= 0).all() and (out["p_has_ebike"] <= 1).all()
    assert (out["p_zero"] >= 0).all() and (out["p_zero"] <= 1).all()
    # p_zero == 1 - p_has_ebike (matches DGNISSMModel contract).
    diff = (out["p_zero"] + out["p_has_ebike"] - 1.0).abs()
    assert (diff < 1e-6).all()


def test_p_count_distributions_normalize_to_one():
    df = _synthetic_examples()
    model = _fit_quick(df)
    rows = df.head(20)
    out = model.predict_distribution(rows)
    for pmf in out["p_count_ebikes"]:
        assert abs(sum(pmf.values()) - 1.0) < 1e-5
    for pmf in out["p_count_total"]:
        assert abs(sum(pmf.values()) - 1.0) < 1e-5


def test_artifact_payload_round_trips_through_pickle():
    df = _synthetic_examples()
    model = _fit_quick(df)
    rows = df.head(5)
    before = model.predict_distribution(rows)
    blob = pickle.dumps(model)
    restored = pickle.loads(blob)
    after = restored.predict_distribution(rows)
    np.testing.assert_allclose(
        before["p_has_ebike"].to_numpy(dtype=float),
        after["p_has_ebike"].to_numpy(dtype=float),
        rtol=1e-5,
        atol=1e-6,
    )


def test_save_and_load_from_path():
    df = _synthetic_examples()
    model = _fit_quick(df)
    rows = df.head(5)
    expected = model.predict_distribution(rows)["p_has_ebike"].to_numpy(dtype=float)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "model.pkl")
        model.save(path)
        loaded = MacFlowNISSMLite.load(path)
        actual = loaded.predict_distribution(rows)["p_has_ebike"].to_numpy(dtype=float)
    np.testing.assert_allclose(expected, actual, rtol=1e-5, atol=1e-6)


def test_predict_distribution_empty_input_returns_typed_empty_frame():
    df = _synthetic_examples()
    model = _fit_quick(df)
    out = model.predict_distribution(df.head(0))
    assert len(out) == 0
    for col in CDG_REQUIRED_OUTPUT_COLUMNS:
        assert col in out.columns


def test_predict_distribution_raises_before_fit():
    model = MacFlowNISSMLite()
    rows = _synthetic_examples().head(2)
    with pytest.raises(RuntimeError, match="no trained artifact"):
        model.predict_distribution(rows)


def test_method_not_bootstrap_or_fallback_after_training():
    df = _synthetic_examples()
    model = _fit_quick(df)
    method = model.method.lower()
    assert "bootstrap" not in method
    assert "fallback" not in method


def test_partition_mode_off_zeros_community_columns():
    df = _synthetic_examples()
    config = MacFlowNISSMLiteConfig(
        epochs=1,
        max_examples=500,
        batch_size_cpu=128,
        device="cpu",
        min_train_examples=50,
        min_positive_examples=10,
        min_zero_future_examples=10,
        min_valid_examples=10,
        partition_mode="off",
        calibrate=False,
    )
    model = MacFlowNISSMLite(config)
    model.fit(df.head(450), df.tail(50))
    assert model.trained
    assert model.partition_mode == "off"


def test_p_appears_and_p_survives_are_conditional_on_e0():
    df = _synthetic_examples()
    model = _fit_quick(df)
    # Force one row with e0=0 and one with e0>0.
    rows = df.head(2).copy()
    rows.loc[rows.index[0], "num_ebikes_available"] = 0
    rows.loc[rows.index[1], "num_ebikes_available"] = 3
    out = model.predict_distribution(rows)
    assert not math.isnan(float(out["p_appears"].iloc[0]))
    assert math.isnan(float(out["p_survives"].iloc[0]))
    assert math.isnan(float(out["p_appears"].iloc[1]))
    assert not math.isnan(float(out["p_survives"].iloc[1]))
