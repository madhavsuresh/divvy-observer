from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from divvy.macflow_features import (
    MACFLOW_FEATURE_COLUMNS,
    NEUTRAL_DEFAULTS,
    apply_runtime_defaults,
    attach_macflow_features,
    build_community_runtime_defaults,
    build_station_aggregates,
)
from divvy.mobility_partitions import Partition


def _toy_partition() -> Partition:
    return Partition(
        partition_id="toy",
        computed_at=datetime(2024, 6, 1),
        source_data_start=datetime(2024, 5, 1),
        source_data_end=datetime(2024, 6, 1),
        algorithm="label_propagation",
        n_communities=2,
        station_to_community={"A": 0, "B": 0, "C": 1, "D": 1},
        station_to_role={"A": "core", "B": "boundary", "C": "core", "D": "gateway"},
        boundary_score={"A": 0.1, "B": 0.25, "C": 0.05, "D": 0.45},
        gateway_score={"A": 0.0, "B": 0.5, "C": 0.0, "D": 2.0},
        inbound_internal_share={"A": 0.9, "B": 0.7, "C": 0.95, "D": 0.5},
        outbound_internal_share={"A": 0.85, "B": 0.65, "C": 0.9, "D": 0.45},
        community_to_neighbors={0: [(1, 1.0)], 1: [(0, 1.0)]},
    )


def _examples(stations: list[str], ts: datetime) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "station_id": stations,
            "anchor_ts": [ts] * len(stations),
            "horizon_minutes": [10] * len(stations),
            "capacity": [15] * len(stations),
            "num_ebikes_available": [2] * len(stations),
            "num_bikes_available": [5] * len(stations),
        }
    )


def test_attach_macflow_features_adds_all_columns():
    partition = _toy_partition()
    rows = _examples(["A", "B", "C", "D"], datetime(2024, 5, 30))
    out = attach_macflow_features(rows, partition, station_aggregates=None)
    assert set(MACFLOW_FEATURE_COLUMNS).issubset(out.columns)
    # role_id should match the partition's ROLE mapping.
    expected_roles = {"A": 0, "B": 1, "C": 0, "D": 2}
    for sid, expected in expected_roles.items():
        assert out.loc[out["station_id"] == sid, "role_id"].iloc[0] == expected


def test_attach_macflow_features_returns_neutral_for_off_mode():
    partition = _toy_partition()
    rows = _examples(["A", "B"], datetime(2024, 5, 30))
    out = attach_macflow_features(rows, partition, partition_mode="off")
    for col, default in NEUTRAL_DEFAULTS.items():
        assert (out[col] == default).all(), f"{col} not neutral"


def test_attach_macflow_features_id_only_zeros_role_and_aggregates():
    partition = _toy_partition()
    rows = _examples(["A", "B", "C", "D"], datetime(2024, 5, 30))
    out = attach_macflow_features(rows, partition, partition_mode="id_only")
    # community_id is preserved.
    assert int(out.loc[out["station_id"] == "A", "community_id"].iloc[0]) == 0
    assert int(out.loc[out["station_id"] == "C", "community_id"].iloc[0]) == 1
    # role_id reset to neutral.
    assert (out["role_id"] == NEUTRAL_DEFAULTS["role_id"]).all()
    # aggregates reset.
    assert (out["boundary_score"] == 0.0).all()
    assert (out["community_recent_ebikes_mean"] == 0.0).all()


def test_attach_macflow_features_id_plus_role_keeps_only_those():
    partition = _toy_partition()
    rows = _examples(["A", "B", "C", "D"], datetime(2024, 5, 30))
    out = attach_macflow_features(rows, partition, partition_mode="id_plus_role")
    # community_id and role_id preserved.
    assert int(out.loc[out["station_id"] == "D", "community_id"].iloc[0]) == 1
    assert int(out.loc[out["station_id"] == "D", "role_id"].iloc[0]) == 2
    # boundary_score / gateway_score reset.
    assert (out["boundary_score"] == 0.0).all()
    assert (out["gateway_score"] == 0.0).all()


def test_attach_macflow_features_missing_partition_returns_defaults():
    rows = _examples(["X", "Y"], datetime(2024, 5, 30))
    out = attach_macflow_features(rows, partition=None)
    for col, default in NEUTRAL_DEFAULTS.items():
        assert (out[col] == default).all()


def test_community_aggregates_are_anchor_ts_leak_free():
    partition = _toy_partition()
    base = datetime(2024, 5, 30, 12, 0)
    # Anchors at t and t+10min for station A.
    rows = pd.DataFrame(
        {
            "station_id": ["A", "A"],
            "anchor_ts": [base, base + timedelta(minutes=10)],
            "horizon_minutes": [10, 10],
            "capacity": [15, 15],
            "num_ebikes_available": [1, 1],
            "num_bikes_available": [5, 5],
        }
    )
    # Two stations in community 0 (A and B). B sees a spike *after* the first anchor — this must NOT affect the first row's feature.
    status = pd.DataFrame(
        {
            "station_id": ["A", "A", "B", "B"],
            "observation_ts": [
                base - timedelta(minutes=20),
                base - timedelta(minutes=5),
                base - timedelta(minutes=5),
                base + timedelta(minutes=5),  # FUTURE relative to row 0
            ],
            "num_ebikes_available": [1, 1, 1, 999],
            "num_bikes_available": [5, 5, 5, 999],
            "num_docks_available": [10, 10, 10, 0],
        }
    )
    aggs = build_station_aggregates(status)
    out = attach_macflow_features(rows, partition, station_aggregates=aggs, lookback_minutes=60)
    # Row 0: anchor at base. B's spike at base+5 is FUTURE → must be excluded.
    row0_mean = out.iloc[0]["community_recent_ebikes_mean"]
    # Row 1: anchor at base+10. B's spike at base+5 is PAST → must be included.
    row1_mean = out.iloc[1]["community_recent_ebikes_mean"]
    assert row0_mean < row1_mean, "future observation leaked into past-anchor aggregate"
    assert row0_mean == pytest.approx(1.0, abs=1e-6)  # A=1, B=1 → mean 1
    assert row1_mean > 100.0  # B's 999 spike now visible


def test_community_aggregates_lookback_window_enforced():
    partition = _toy_partition()
    base = datetime(2024, 5, 30, 12, 0)
    rows = _examples(["A"], base)
    # B observation older than lookback should be excluded; only A within window.
    status = pd.DataFrame(
        {
            "station_id": ["A", "B"],
            "observation_ts": [
                base - timedelta(minutes=10),
                base - timedelta(minutes=300),  # outside lookback
            ],
            "num_ebikes_available": [3, 999],
            "num_bikes_available": [6, 999],
            "num_docks_available": [9, 0],
        }
    )
    aggs = build_station_aggregates(status)
    out = attach_macflow_features(rows, partition, station_aggregates=aggs, lookback_minutes=60)
    # Should only see A=3, B excluded.
    assert out.iloc[0]["community_recent_ebikes_mean"] == pytest.approx(3.0, abs=1e-6)


def test_runtime_defaults_apply_to_unseen_rows():
    partition = _toy_partition()
    train = _examples(["A", "B", "C", "D"], datetime(2024, 5, 30))
    train_with_features = attach_macflow_features(train, partition, station_aggregates=None)
    defaults = build_community_runtime_defaults(train_with_features, partition)

    live = pd.DataFrame(
        {
            "station_id": ["A", "Z"],  # Z is unknown
            "anchor_ts": [datetime(2024, 6, 5)] * 2,
            "horizon_minutes": [5, 5],
            "capacity": [15, 15],
            "num_ebikes_available": [0, 0],
            "num_bikes_available": [0, 0],
        }
    )
    out = apply_runtime_defaults(live, defaults)
    # Known station A keeps its community/role info.
    assert int(out.loc[out["station_id"] == "A", "community_id"].iloc[0]) == 0
    assert int(out.loc[out["station_id"] == "A", "role_id"].iloc[0]) == 0
    # Unknown station Z falls back to neutral defaults.
    assert int(out.loc[out["station_id"] == "Z", "role_id"].iloc[0]) == int(NEUTRAL_DEFAULTS["role_id"])


def test_attach_macflow_features_empty_input_returns_empty_with_columns():
    partition = _toy_partition()
    empty = pd.DataFrame(
        columns=[
            "station_id",
            "anchor_ts",
            "horizon_minutes",
            "capacity",
            "num_ebikes_available",
            "num_bikes_available",
        ]
    )
    out = attach_macflow_features(empty, partition)
    for col in MACFLOW_FEATURE_COLUMNS:
        assert col in out.columns
    assert len(out) == 0
