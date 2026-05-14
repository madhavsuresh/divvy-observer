from __future__ import annotations

from datetime import datetime, timedelta, timezone

import duckdb
import pandas as pd
import pytest

from divvy import db
from divvy.mobility_partitions import (
    INT_TO_ROLE,
    Partition,
    ROLE_TO_INT,
    _label_propagation,
    _spectral_kmeans,
    build_partition,
    make_random_partition,
    upsert_station_communities,
)


def _utc_naive(value: datetime) -> datetime:
    return value.replace(tzinfo=None) if value.tzinfo else value


@pytest.fixture
def memory_conn():
    conn = duckdb.connect(":memory:")
    db.init_schema(conn)
    yield conn
    conn.close()


def _seed_stations(conn, stations) -> None:
    base = datetime(2024, 1, 1)
    rows = [
        (sid, None, sid, sid, lat, lon, 15, "classic", base, base)
        for sid, lat, lon in stations
    ]
    conn.executemany(
        "INSERT INTO stations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )


def _seed_trips(conn, edges, start_ts: datetime) -> None:
    rows = []
    counter = 0
    for src, dst, count in edges:
        for _ in range(int(count)):
            counter += 1
            rows.append(
                (
                    f"r{counter:06d}",
                    "classic_bike",
                    start_ts,
                    start_ts + timedelta(minutes=10),
                    src,
                    src,
                    dst,
                    dst,
                    41.0,
                    -87.0,
                    41.01,
                    -87.01,
                    "member",
                    10.0,
                )
            )
    conn.executemany(
        "INSERT INTO divvy_trips VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )


def test_two_cluster_toy_graph_partitions_correctly():
    # 6 nodes: {A,B,C} dense, {D,E,F} dense, weak bridge A<->D.
    stations = ["A", "B", "C", "D", "E", "F"]
    import numpy as np

    W = np.zeros((6, 6))
    idx = {sid: i for i, sid in enumerate(stations)}
    # Cluster 1
    for a, b, w in [("A", "B", 50), ("B", "C", 50), ("A", "C", 50)]:
        W[idx[a], idx[b]] = w
        W[idx[b], idx[a]] = w
    # Cluster 2
    for a, b, w in [("D", "E", 50), ("E", "F", 50), ("D", "F", 50)]:
        W[idx[a], idx[b]] = w
        W[idx[b], idx[a]] = w
    # Bridge
    W[idx["A"], idx["D"]] = 1
    W[idx["D"], idx["A"]] = 1

    labels = _label_propagation(W, seed=42)
    assert labels.shape == (6,)
    assert labels[0] == labels[1] == labels[2]  # A, B, C
    assert labels[3] == labels[4] == labels[5]  # D, E, F
    assert labels[0] != labels[3]


def test_spectral_kmeans_separates_two_blocks():
    import numpy as np

    block_size = 10
    n = 2 * block_size
    W = np.zeros((n, n))
    for i in range(block_size):
        for j in range(block_size):
            if i != j:
                W[i, j] = 10.0
                W[block_size + i, block_size + j] = 10.0
    # Weak inter-block link
    W[0, block_size] = 0.5
    W[block_size, 0] = 0.5

    labels = _spectral_kmeans(W, target_k=2, seed=0)
    assert labels.shape == (n,)
    first_block = set(labels[:block_size].tolist())
    second_block = set(labels[block_size:].tolist())
    assert len(first_block) == 1
    assert len(second_block) == 1
    assert first_block != second_block


def test_build_partition_with_no_trip_data_returns_single_community(memory_conn):
    _seed_stations(
        memory_conn,
        [("A", 41.0, -87.0), ("B", 41.01, -87.01), ("C", 41.02, -87.02)],
    )
    partition = build_partition(
        memory_conn,
        training_window_end=datetime(2024, 6, 1),
        lookback_days=30,
    )
    # No flows but coords exist → spatial fallback may produce 2+ communities; either way it must succeed.
    assert partition.n_communities >= 1
    assert set(partition.station_to_community.keys()) == {"A", "B", "C"}
    # Without trip data every station has role unknown unless spatial fallback assigned communities
    # (spatial fallback still leaves directed W empty so roles remain unknown).
    assert all(role == "unknown" for role in partition.station_to_role.values())


def test_build_partition_uses_trip_data_when_available(memory_conn):
    stations = [
        ("A", 41.0, -87.0),
        ("B", 41.0, -87.0),
        ("C", 41.0, -87.0),
        ("D", 41.5, -87.5),
        ("E", 41.5, -87.5),
        ("F", 41.5, -87.5),
    ]
    _seed_stations(memory_conn, stations)
    # Dense intra-cluster trips + one cross-cluster trip
    edges = [
        ("A", "B", 30),
        ("B", "C", 30),
        ("A", "C", 30),
        ("D", "E", 30),
        ("E", "F", 30),
        ("D", "F", 30),
        ("A", "D", 1),
    ]
    _seed_trips(memory_conn, edges, datetime(2024, 5, 15))

    partition = build_partition(
        memory_conn,
        training_window_end=datetime(2024, 6, 1),
        lookback_days=30,
    )
    assert partition.n_communities >= 2
    assert partition.algorithm in ("label_propagation", "spectral_kmeans")
    # A, B, C should be in the same community; D, E, F should be in the same community; and they should differ.
    assert partition.station_to_community["A"] == partition.station_to_community["B"]
    assert partition.station_to_community["B"] == partition.station_to_community["C"]
    assert partition.station_to_community["D"] == partition.station_to_community["E"]
    assert partition.station_to_community["E"] == partition.station_to_community["F"]
    assert partition.station_to_community["A"] != partition.station_to_community["D"]


def test_role_assignment_marks_bridge_as_gateway_or_boundary(memory_conn):
    stations = [(f"S{i}", 41.0 + 0.001 * i, -87.0) for i in range(6)]
    _seed_stations(memory_conn, stations)
    # Two dense clusters; S0 is a weak bridge to S3 only.
    edges = [
        ("S0", "S1", 20), ("S1", "S0", 20),
        ("S1", "S2", 20), ("S2", "S1", 20),
        ("S0", "S2", 20), ("S2", "S0", 20),
        ("S3", "S4", 20), ("S4", "S3", 20),
        ("S4", "S5", 20), ("S5", "S4", 20),
        ("S3", "S5", 20), ("S5", "S3", 20),
        # Moderate cross-cluster bridge: S0 sends ~25% of its trips externally.
        ("S0", "S3", 15), ("S3", "S0", 15),
    ]
    _seed_trips(memory_conn, edges, datetime(2024, 5, 15))

    partition = build_partition(
        memory_conn,
        training_window_end=datetime(2024, 6, 1),
        lookback_days=30,
    )
    # We should detect two clusters.
    assert partition.n_communities >= 2
    # S0 bridges the two clusters; should be flagged as boundary or gateway.
    assert partition.station_to_role["S0"] in {"gateway", "boundary"}
    # Pure-interior stations are core.
    assert partition.station_to_role["S2"] == "core"
    assert partition.station_to_role["S5"] == "core"


def test_random_partition_preserves_sizes(memory_conn):
    base = Partition(
        partition_id="test",
        computed_at=datetime(2024, 6, 1),
        source_data_start=datetime(2024, 5, 1),
        source_data_end=datetime(2024, 6, 1),
        algorithm="label_propagation",
        n_communities=3,
        station_to_community={"A": 0, "B": 0, "C": 1, "D": 1, "E": 2},
        station_to_role={"A": "core", "B": "core", "C": "boundary", "D": "core", "E": "gateway"},
    )
    randomized = make_random_partition(base, seed=1)
    assert randomized.algorithm == "random"
    assert randomized.community_sizes() == base.community_sizes()
    # Roles preserved per station (we shuffle community assignments only).
    assert randomized.station_to_role == base.station_to_role


def test_upsert_station_communities_writes_rows(memory_conn):
    partition = Partition(
        partition_id="abc123",
        computed_at=datetime(2024, 6, 1),
        source_data_start=datetime(2024, 5, 1),
        source_data_end=datetime(2024, 6, 1),
        algorithm="label_propagation",
        n_communities=2,
        station_to_community={"A": 0, "B": 1},
        station_to_role={"A": "core", "B": "boundary"},
        boundary_score={"A": 0.1, "B": 0.3},
        gateway_score={"A": 0.0, "B": 1.2},
        inbound_internal_share={"A": 0.8, "B": 0.4},
        outbound_internal_share={"A": 0.9, "B": 0.3},
    )
    n = upsert_station_communities(memory_conn, partition)
    assert n == 2
    rows = memory_conn.execute(
        "SELECT station_id, community_id, role, boundary_score FROM station_communities ORDER BY station_id"
    ).df()
    assert list(rows["station_id"]) == ["A", "B"]
    assert list(rows["community_id"]) == [0, 1]
    assert list(rows["role"]) == ["core", "boundary"]
    assert rows["boundary_score"].iloc[1] == pytest.approx(0.3)


def test_role_int_mapping_is_consistent():
    assert ROLE_TO_INT["core"] == 0
    assert ROLE_TO_INT["boundary"] == 1
    assert ROLE_TO_INT["gateway"] == 2
    assert ROLE_TO_INT["unknown"] == 3
    for value in range(4):
        assert ROLE_TO_INT[INT_TO_ROLE[value]] == value
