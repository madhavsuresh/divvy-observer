from __future__ import annotations

from datetime import datetime, timedelta

import duckdb

from divvy import db, label_builder


def test_backward_asof_labels_hold_state_between_changes() -> None:
    conn = duckdb.connect(":memory:")
    db.init_schema(conn)
    base = datetime(2026, 1, 1, 10, 0, 0)
    conn.execute(
        """
        INSERT INTO stations VALUES
        ('s1', NULL, 'S1', 'Station', 41.0, -87.0, 10, 'classic', ?, ?)
        """,
        [base, base],
    )
    conn.executemany(
        """
        INSERT INTO station_status VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            ("s1", base, base, 3, 1, 0, 7, 0, True, True, True),
            ("s1", base + timedelta(minutes=20), base + timedelta(minutes=20), 4, 2, 0, 6, 0, True, True, True),
        ],
    )

    examples = label_builder.build_leak_free_examples(
        conn,
        start_ts=base,
        end_ts=base + timedelta(minutes=10),
        station_ids=["s1"],
        horizons=(5,),
        anchor_every_min=5,
        max_label_status_age_min=20,
    )

    assert not examples.empty
    labels = examples.set_index("target_at")["e_future"].to_dict()
    assert labels[base + timedelta(minutes=5)] == 1
    assert labels[base + timedelta(minutes=10)] == 1
    assert labels[base + timedelta(minutes=15)] == 1
    assert (examples["label_reported_at"] <= examples["target_at"]).all()
    assert (examples["current_reported_at"] <= examples["anchor_ts"]).all()
    assert {"current_status_age_minutes", "label_status_age_minutes"}.issubset(examples.columns)


def test_fetched_at_is_training_clock_for_stale_gbfs_last_reported() -> None:
    conn = duckdb.connect(":memory:")
    db.init_schema(conn)
    fetched_at = datetime(2026, 5, 13, 20, 0, 0)
    stale_reported = datetime(2025, 6, 13, 10, 0, 0)
    conn.execute(
        """
        INSERT INTO stations VALUES
        ('s1', NULL, 'S1', 'Station', 41.0, -87.0, 10, 'classic', ?, ?)
        """,
        [fetched_at, fetched_at],
    )
    conn.execute(
        """
        INSERT INTO station_status VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("s1", stale_reported, fetched_at, 3, 1, 0, 7, 0, True, True, True),
    )

    historical = label_builder.build_leak_free_examples(
        conn,
        start_ts=stale_reported,
        end_ts=stale_reported + timedelta(minutes=5),
        station_ids=["s1"],
        horizons=(5,),
        anchor_every_min=5,
        max_current_status_age_min=30,
        max_label_status_age_min=30,
    )
    examples = label_builder.build_leak_free_examples(
        conn,
        start_ts=fetched_at,
        end_ts=fetched_at,
        station_ids=["s1"],
        horizons=(5,),
        anchor_every_min=5,
        max_current_status_age_min=30,
        max_label_status_age_min=30,
    )

    assert historical.empty
    assert len(examples) == 1
    row = examples.iloc[0]
    assert row["anchor_ts"] == fetched_at
    assert row["fetched_at"] == fetched_at
    assert row["source_last_reported"] == stale_reported
    assert row["status_age_minutes"] > 400_000
    assert row["current_status_age_minutes"] == 0.0
