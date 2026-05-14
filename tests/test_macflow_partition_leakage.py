from __future__ import annotations

from datetime import datetime, timedelta

import duckdb
import pytest

from divvy import db
from divvy.mobility_partitions import build_partition, upsert_station_communities


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


def _seed_trip(conn, ride_id, src, dst, started_at) -> None:
    conn.execute(
        "INSERT INTO divvy_trips VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            ride_id,
            "classic_bike",
            started_at,
            started_at + timedelta(minutes=10),
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
        ),
    )


def test_partition_excludes_trips_after_training_window_end(memory_conn):
    _seed_stations(
        memory_conn,
        [
            ("A", 41.0, -87.0),
            ("B", 41.0, -87.0),
            ("C", 41.0, -87.0),
            ("D", 41.5, -87.5),
            ("E", 41.5, -87.5),
            ("F", 41.5, -87.5),
        ],
    )
    T = datetime(2024, 6, 1)
    # Pre-window trips: 3 within {A,B,C}.
    for i, (s, d) in enumerate([("A", "B"), ("B", "C"), ("A", "C")] * 5):
        _seed_trip(memory_conn, f"pre{i:03d}", s, d, T - timedelta(days=2))
    # Post-window trips: 30 connecting {A} to {D,E,F} — would force A into the same cluster as D/E/F if leaked.
    for i in range(30):
        _seed_trip(memory_conn, f"post{i:03d}", "A", "D", T + timedelta(days=1))

    partition = build_partition(memory_conn, training_window_end=T, lookback_days=30)
    assert partition.source_data_end == T
    # Source data start should NOT extend past T.
    assert partition.source_data_start is None or partition.source_data_start <= T


def test_partition_upserts_into_station_communities_table(memory_conn):
    _seed_stations(
        memory_conn,
        [
            ("A", 41.0, -87.0),
            ("B", 41.0, -87.0),
            ("C", 41.0, -87.0),
            ("D", 41.5, -87.5),
            ("E", 41.5, -87.5),
            ("F", 41.5, -87.5),
        ],
    )
    T = datetime(2024, 6, 1)
    for i, (s, d) in enumerate([("A", "B"), ("B", "C"), ("D", "E"), ("E", "F")] * 5):
        _seed_trip(memory_conn, f"r{i:03d}", s, d, T - timedelta(days=2))
    partition = build_partition(memory_conn, training_window_end=T, lookback_days=30)
    n = upsert_station_communities(memory_conn, partition)
    assert n == len(partition.station_to_community)
    rows = memory_conn.execute(
        "SELECT COUNT(*) FROM station_communities WHERE partition_id = ?",
        [partition.partition_id],
    ).fetchone()
    assert rows[0] == len(partition.station_to_community)


def test_upsert_replaces_existing_rows_for_same_partition_id(memory_conn):
    _seed_stations(memory_conn, [("A", 41.0, -87.0), ("B", 41.0, -87.0)])
    T = datetime(2024, 6, 1)
    _seed_trip(memory_conn, "t1", "A", "B", T - timedelta(days=2))
    partition = build_partition(memory_conn, training_window_end=T, lookback_days=30)
    upsert_station_communities(memory_conn, partition)
    upsert_station_communities(memory_conn, partition)  # idempotent
    rows = memory_conn.execute(
        "SELECT COUNT(*) FROM station_communities WHERE partition_id = ?",
        [partition.partition_id],
    ).fetchone()
    assert rows[0] == len(partition.station_to_community)
