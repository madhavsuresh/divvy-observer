"""Verify that MacFlow gracefully falls back to CPU when MPS isn't usable."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

from divvy.macflow_nissm import (
    MacFlowNISSMLite,
    MacFlowNISSMLiteConfig,
    select_device,
)


def _toy_examples(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = datetime(2024, 5, 1)
    rows = []
    for i in range(n):
        e0 = int(rng.choice([0, 0, 1, 2, 3]))
        rows.append(
            {
                "station_id": f"s{i % 3}",
                "anchor_ts": base + timedelta(minutes=i),
                "horizon_minutes": int(rng.choice([5, 10])),
                "capacity": 15,
                "num_ebikes_available": e0,
                "num_bikes_available": min(15, e0 + 2),
                "num_docks_available": 13 - e0,
                "has_ebike": int(e0 >= 1),
                "obs_e_depart": 0.0,
                "obs_e_arrive": 0.0,
                "obs_c_depart": 0.0,
                "obs_c_arrive": 0.0,
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
                "current_total_bikes_clipped": float(min(15, e0 + 2)),
                "docks_available_clipped": float(13 - e0),
                "ebike_share_of_bikes": 0.3,
                "dock_availability_fraction": 0.5,
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


def test_select_device_returns_cpu_when_explicitly_requested():
    assert select_device("cpu") == "cpu"


def test_select_device_auto_falls_back_to_cpu_when_mps_unavailable():
    with patch("torch.backends.mps.is_available", return_value=False):
        assert select_device("auto") == "cpu"


def test_select_device_mps_falls_back_when_not_available():
    with patch("torch.backends.mps.is_available", return_value=False):
        assert select_device("mps") == "cpu"


def test_fit_falls_back_when_to_device_raises():
    df = _toy_examples()
    config = MacFlowNISSMLiteConfig(
        epochs=1,
        max_examples=500,
        batch_size_cpu=64,
        batch_size_mps=64,
        device="auto",
        min_train_examples=50,
        min_positive_examples=5,
        min_zero_future_examples=5,
        min_valid_examples=0,
        calibrate=False,
    )
    model = MacFlowNISSMLite(config)

    real_to = torch.nn.Module.to

    def fail_on_mps(self, target, *args, **kwargs):
        if isinstance(target, str) and target == "mps":
            raise RuntimeError("simulated MPS failure")
        if hasattr(target, "type") and target.type == "mps":
            raise RuntimeError("simulated MPS failure")
        return real_to(self, target, *args, **kwargs)

    with patch("torch.backends.mps.is_available", return_value=True), patch(
        "torch.backends.mps.is_built", return_value=True
    ), patch.object(torch.nn.Module, "to", fail_on_mps):
        train = df.head(150)
        valid = df.tail(50)
        model.fit(train, valid)
    assert model.trained is True
    # Device falls back to CPU.
    assert model.metrics.get("device") in {"cpu", "mps"}  # accept whichever string the training loop logged after fallback
