"""Verify MacFlow gracefully falls back when trip/community data is missing."""

from __future__ import annotations

from datetime import datetime, timedelta

import duckdb
import numpy as np
import pandas as pd
import pytest

from divvy import db
from divvy.macflow_nissm import MacFlowNISSMLite, MacFlowNISSMLiteConfig
from divvy.mobility_partitions import build_partition


def _examples(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    base = datetime(2024, 5, 1)
    for i in range(n):
        cap = 15
        e0 = int(min(cap, rng.poisson(2)))
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
                "has_ebike": int(rng.random() > 0.5),
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


def test_fit_succeeds_with_no_partition_or_trip_data():
    df = _examples()
    config = MacFlowNISSMLiteConfig(
        epochs=1,
        max_examples=1000,
        batch_size_cpu=128,
        device="cpu",
        min_train_examples=50,
        min_positive_examples=10,
        min_zero_future_examples=10,
        min_valid_examples=10,
        calibrate=False,
    )
    model = MacFlowNISSMLite(config)
    # No graph_cache → expects single-community fallback.
    model.fit(df.head(350), df.tail(50))
    assert model.trained is True
    assert model.partition is not None
    assert model.partition.algorithm in ("single_community", "label_propagation", "spectral_kmeans", "spatial_kmeans")


def test_predict_falls_back_to_runtime_defaults_when_features_absent():
    df = _examples()
    config = MacFlowNISSMLiteConfig(
        epochs=1,
        max_examples=1000,
        batch_size_cpu=128,
        device="cpu",
        min_train_examples=50,
        min_positive_examples=10,
        min_zero_future_examples=10,
        min_valid_examples=10,
        calibrate=False,
    )
    model = MacFlowNISSMLite(config)
    model.fit(df.head(350), df.tail(50))

    # Strip community columns: simulate a live request frame that doesn't carry them.
    live = df.tail(5).drop(
        columns=[col for col in df.columns if col.startswith("community_") or col == "role_id"],
        errors="ignore",
    )
    out = model.predict_distribution(live)
    assert len(out) == 5
    assert (out["p_has_ebike"] >= 0).all() and (out["p_has_ebike"] <= 1).all()


def test_build_partition_handles_empty_db_gracefully():
    conn = duckdb.connect(":memory:")
    try:
        db.init_schema(conn)
        partition = build_partition(conn, training_window_end=datetime(2024, 6, 1), lookback_days=30)
        # No stations, no trips → empty partition with 0 communities.
        assert partition.n_communities == 0
        assert partition.station_to_community == {}
    finally:
        conn.close()


def test_fit_succeeds_when_divvy_trips_table_missing():
    df = _examples()
    config = MacFlowNISSMLiteConfig(
        epochs=1,
        max_examples=1000,
        batch_size_cpu=128,
        device="cpu",
        min_train_examples=50,
        min_positive_examples=10,
        min_zero_future_examples=10,
        min_valid_examples=10,
        calibrate=False,
    )
    model = MacFlowNISSMLite(config)
    # graph_cache=None and no conn → uses single-community fallback path inside fit().
    model.fit(df.head(350), df.tail(50), graph_cache=None)
    assert model.trained
