"""Capacity-consistency tests: the rollout PMFs must obey 0 <= E <= Q <= K."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from divvy.macflow_nissm import MacFlowNISSMLite, MacFlowNISSMLiteConfig


def _build_rows(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    base = datetime(2024, 5, 1)
    for i in range(n):
        cap = int(rng.integers(8, 30))
        # Bias toward zero ebikes so both classes are present.
        e0 = int(rng.choice([0, 0, 1, 2, 3, 4, 5]))
        q0 = int(rng.integers(e0, cap + 1))
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


def _fit_model(df: pd.DataFrame) -> MacFlowNISSMLite:
    config = MacFlowNISSMLiteConfig(
        epochs=1,
        max_examples=2000,
        batch_size_cpu=128,
        device="cpu",
        min_train_examples=50,
        min_positive_examples=5,
        min_zero_future_examples=5,
        min_valid_examples=0,
        calibrate=False,
    )
    model = MacFlowNISSMLite(config)
    train = df.head(int(len(df) * 0.8))
    valid = df.tail(len(df) - int(len(df) * 0.8))
    model.fit(train, valid)
    if not model.trained:
        raise AssertionError(f"fit skipped: {model.metrics}")
    return model


def test_pmf_normalization_holds():
    df = _build_rows(200)
    model = _fit_model(df)
    out = model.predict_distribution(df.head(100))
    for pmf in out["p_count_ebikes"]:
        s = float(sum(pmf.values()))
        assert abs(s - 1.0) < 1e-5
    for pmf in out["p_count_total"]:
        s = float(sum(pmf.values()))
        assert abs(s - 1.0) < 1e-5


def test_expected_counts_obey_capacity_bounds():
    df = _build_rows(200)
    model = _fit_model(df)
    rows = df.head(100)
    out = model.predict_distribution(rows)
    capacities = pd.to_numeric(rows["capacity"], errors="coerce").to_numpy(dtype=float)
    e_exp = out["expected_ebikes"].to_numpy(dtype=float)
    q_exp = out["expected_total_bikes"].to_numpy(dtype=float)
    assert (e_exp >= 0).all()
    assert (q_exp >= 0).all()
    assert (q_exp <= capacities + 1e-6).all()
    # E_expected <= Q_expected (allow tiny numeric slop).
    assert (e_exp <= q_exp + 1e-6).all()


def test_probabilities_match_pmf_zero():
    df = _build_rows(100)
    model = _fit_model(df)
    out = model.predict_distribution(df.head(50))
    for _, row in out.iterrows():
        pmf = row["p_count_ebikes"]
        p_zero_from_pmf = float(pmf.get("0", 0.0))
        assert abs(row["p_zero"] - p_zero_from_pmf) < 1e-5
        assert abs(row["p_has_ebike"] - (1.0 - p_zero_from_pmf)) < 1e-5
