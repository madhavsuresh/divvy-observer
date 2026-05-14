from __future__ import annotations

from datetime import datetime

import duckdb

from divvy import db, dynamic_graph


def test_dynamic_graph_builds_sparse_distance_edges() -> None:
    conn = duckdb.connect(":memory:")
    db.init_schema(conn)
    now = datetime(2026, 1, 1)
    conn.executemany(
        "INSERT INTO stations VALUES (?, NULL, ?, ?, ?, ?, 10, 'classic', ?, ?)",
        [
            ("a", "A", "A", 41.0, -87.0, now, now),
            ("b", "B", "B", 41.001, -87.001, now, now),
            ("c", "C", "C", 41.01, -87.01, now, now),
        ],
    )

    edges = dynamic_graph.build_dynamic_graph_edges(conn, now, top_k=1)

    assert not edges.empty
    assert set(edges["relation"]) == {"distance"}
    assert edges.groupby("dst_station_id").size().max() == 1
