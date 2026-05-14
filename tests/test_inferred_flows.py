from __future__ import annotations

from datetime import datetime, timedelta, timezone

import duckdb

from divvy import db, inferred_flows, service_state


def _bucket(ts: datetime, minutes: int = 10) -> datetime:
    floored = ts.replace(second=0, microsecond=0)
    return floored.replace(minute=(floored.minute // minutes) * minutes)


def _seed_station(conn: duckdb.DuckDBPyConnection, base: datetime) -> None:
    conn.executemany(
        "INSERT INTO stations VALUES (?, NULL, ?, ?, ?, ?, 15, 'classic', ?, ?)",
        [("s1", "S1", "Station One", 41.0, -87.0, base, base)],
    )


def _insert_status(
    conn: duckdb.DuckDBPyConnection,
    station_id: str,
    ts: datetime,
    bikes: int,
    ebikes: int,
) -> None:
    docks = 15 - bikes
    conn.execute(
        "INSERT INTO station_status VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [station_id, ts, ts, bikes, ebikes, 0, max(0, docks), 0, True, True, True],
    )


def test_monotonic_decrease_produces_departures() -> None:
    conn = duckdb.connect(":memory:")
    db.init_schema(conn)
    base = datetime(2026, 1, 1, 12, 0, 0)
    _seed_station(conn, base)
    # 10 ticks one minute apart, ebikes decreasing 10 -> 1 (9 departures total)
    for i in range(10):
        _insert_status(conn, "s1", base + timedelta(minutes=i), bikes=10 - i, ebikes=10 - i)

    n = inferred_flows.compute_window(conn, base - timedelta(minutes=1), base + timedelta(minutes=15))
    assert n >= 1

    row = conn.execute(
        """
        SELECT SUM(ebike_departures), SUM(ebike_arrivals),
               SUM(classic_departures), SUM(classic_arrivals),
               SUM(rebalancing_events)
        FROM station_inferred_flows WHERE station_id = 's1'
        """
    ).fetchone()
    # 9 single-bike decrements; threshold is 4 so none filtered
    assert row[0] == 9
    assert row[1] == 0
    # Classic count = num_bikes_available - num_ebikes_available = 0 throughout
    assert row[2] == 0
    assert row[3] == 0
    assert row[4] == 0


def test_mixed_up_and_down_within_bucket() -> None:
    conn = duckdb.connect(":memory:")
    db.init_schema(conn)
    base = datetime(2026, 1, 1, 12, 0, 0)
    _seed_station(conn, base)
    # ebikes: 5 -> 4 (departure) -> 6 (2 arrivals) -> 5 (departure)
    _insert_status(conn, "s1", base, bikes=5, ebikes=5)
    _insert_status(conn, "s1", base + timedelta(minutes=1), bikes=4, ebikes=4)
    _insert_status(conn, "s1", base + timedelta(minutes=2), bikes=6, ebikes=6)
    _insert_status(conn, "s1", base + timedelta(minutes=3), bikes=5, ebikes=5)

    inferred_flows.compute_window(conn, base - timedelta(minutes=1), base + timedelta(minutes=15))
    row = conn.execute(
        """
        SELECT SUM(ebike_departures), SUM(ebike_arrivals)
        FROM station_inferred_flows WHERE station_id = 's1'
        """
    ).fetchone()
    assert row[0] == 2  # one + one
    assert row[1] == 2  # the 4->6 jump


def test_large_swing_flagged_as_rebalancing() -> None:
    conn = duckdb.connect(":memory:")
    db.init_schema(conn)
    base = datetime(2026, 1, 1, 12, 0, 0)
    _seed_station(conn, base)
    _insert_status(conn, "s1", base, bikes=10, ebikes=10)
    # Drop of 6 in one tick -> exceeds threshold (4), counted as rebalance
    _insert_status(conn, "s1", base + timedelta(minutes=1), bikes=4, ebikes=4)

    inferred_flows.compute_window(conn, base - timedelta(minutes=1), base + timedelta(minutes=15))
    row = conn.execute(
        """
        SELECT SUM(ebike_departures), SUM(ebike_arrivals), SUM(rebalancing_events)
        FROM station_inferred_flows WHERE station_id = 's1'
        """
    ).fetchone()
    assert row[0] == 0
    assert row[1] == 0
    assert row[2] == 1


def test_idempotent_rerun() -> None:
    conn = duckdb.connect(":memory:")
    db.init_schema(conn)
    base = datetime(2026, 1, 1, 12, 0, 0)
    _seed_station(conn, base)
    for i in range(5):
        _insert_status(conn, "s1", base + timedelta(minutes=i), bikes=5 - i, ebikes=5 - i)

    window_start = base - timedelta(minutes=1)
    window_end = base + timedelta(minutes=15)
    inferred_flows.compute_window(conn, window_start, window_end)
    first_count = conn.execute("SELECT COUNT(*), SUM(ebike_departures) FROM station_inferred_flows").fetchone()

    inferred_flows.compute_window(conn, window_start, window_end)
    second_count = conn.execute("SELECT COUNT(*), SUM(ebike_departures) FROM station_inferred_flows").fetchone()

    assert first_count == second_count


def test_empty_window_returns_no_rows() -> None:
    conn = duckdb.connect(":memory:")
    db.init_schema(conn)
    base = datetime(2026, 1, 1, 12, 0, 0)
    _seed_station(conn, base)
    n = inferred_flows.compute_window(conn, base, base + timedelta(minutes=5))
    assert n == 0
    count = conn.execute("SELECT COUNT(*) FROM station_inferred_flows").fetchone()[0]
    assert count == 0


def test_run_incremental_advances_watermark() -> None:
    conn = duckdb.connect(":memory:")
    db.init_schema(conn)
    base = datetime(2026, 1, 1, 12, 0, 0)
    _seed_station(conn, base)
    for i in range(5):
        _insert_status(conn, "s1", base + timedelta(minutes=i), bikes=5 - i, ebikes=5 - i)

    result = inferred_flows.run_incremental(conn)
    assert result["status"] == "ok"
    watermark = service_state.get_watermark(conn, inferred_flows.WATERMARK_KEY)
    assert watermark is not None
    assert watermark >= base + timedelta(minutes=4)

    # Second incremental run on the same data should not error and produce a stable count
    rows_before = conn.execute("SELECT COUNT(*) FROM station_inferred_flows").fetchone()[0]
    inferred_flows.run_incremental(conn)
    rows_after = conn.execute("SELECT COUNT(*) FROM station_inferred_flows").fetchone()[0]
    assert rows_after == rows_before


def test_incremental_rerun_with_partial_bucket_overlap_is_stable() -> None:
    """Regression: an incremental run that re-touches a bucket must rewrite
    its full count, not just the deltas inside the overlap window."""
    conn = duckdb.connect(":memory:")
    db.init_schema(conn)
    base = datetime(2026, 1, 1, 12, 0, 0)  # start of a 10-min bucket
    _seed_station(conn, base)
    # Six 1-minute deltas all within the same 10-min bucket [12:00, 12:10)
    for i in range(6):
        _insert_status(conn, "s1", base + timedelta(minutes=i), bikes=10 - i, ebikes=10 - i)

    inferred_flows.run_incremental(conn)
    first = conn.execute(
        "SELECT departures, arrivals FROM station_inferred_flows WHERE station_id='s1'"
    ).fetchall()

    # Run again with no new data -- watermark sits inside the bucket
    inferred_flows.run_incremental(conn)
    second = conn.execute(
        "SELECT departures, arrivals FROM station_inferred_flows WHERE station_id='s1'"
    ).fetchall()
    assert first == second


def test_backfill_wipes_and_recomputes() -> None:
    conn = duckdb.connect(":memory:")
    db.init_schema(conn)
    base = datetime(2026, 1, 1, 12, 0, 0)
    _seed_station(conn, base)
    for i in range(5):
        _insert_status(conn, "s1", base + timedelta(minutes=i), bikes=5 - i, ebikes=5 - i)

    inferred_flows.run_incremental(conn)
    rows_before = conn.execute("SELECT COUNT(*) FROM station_inferred_flows").fetchone()[0]
    assert rows_before > 0

    result = inferred_flows.run_backfill(conn)
    assert result["status"] == "ok"
    rows_after = conn.execute("SELECT COUNT(*) FROM station_inferred_flows").fetchone()[0]
    assert rows_after == rows_before
