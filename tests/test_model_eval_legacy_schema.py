from __future__ import annotations

from datetime import datetime, timedelta, timezone

import duckdb

from divvy import model_eval


def test_performance_summary_tolerates_pre_arrival_schema() -> None:
    conn = duckdb.connect(":memory:")
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    conn.execute(
        """
        CREATE TABLE model_forecasts (
          forecast_id TEXT PRIMARY KEY,
          request_id TEXT,
          source TEXT,
          model_key TEXT,
          model_label TEXT,
          model_version TEXT,
          query_place_key TEXT,
          query_label TEXT,
          station_id TEXT,
          station_name TEXT,
          forecasted_at TIMESTAMP,
          target_at TIMESTAMP,
          horizon_minutes INTEGER,
          distance_km DOUBLE,
          current_ebikes INTEGER,
          p_has_ebike DOUBLE,
          confidence TEXT,
          is_recommended BOOLEAN,
          expected_ebikes DOUBLE,
          expected_total_bikes DOUBLE,
          p_count_ebikes_json JSON,
          p_capacity_violation DOUBLE,
          p_dock_constrained_arrival DOUBLE
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE model_outcomes (
          forecast_id TEXT PRIMARY KEY,
          station_id TEXT,
          horizon_minutes INTEGER,
          target_at TIMESTAMP,
          observed_at TIMESTAMP,
          observed_ebikes INTEGER,
          observed_has_ebike BOOLEAN,
          resolved_at TIMESTAMP,
          observed_total_bikes INTEGER,
          observed_docks INTEGER,
          count_log_prob DOUBLE,
          crps DOUBLE
        )
        """
    )
    conn.execute(
        """
        INSERT INTO model_forecasts VALUES (
          'f1', 'r1', 'test', 'inventory_world', 'Inventory', 'v1',
          '41.000,-87.000', 'place', 's1', 'Station',
          ?, ?, 10, 0.2, 1, 0.7, 'high', true, 1.2, 4.0,
          '{"0":0.3,"1":0.7}', 0.0, 0.0
        )
        """,
        [now - timedelta(minutes=20), now - timedelta(minutes=10)],
    )
    conn.execute(
        """
        INSERT INTO model_outcomes VALUES (
          'f1', 's1', 10, ?, ?, 1, true, ?, 4, 6, 0.2, 0.1
        )
        """,
        [now - timedelta(minutes=10), now - timedelta(minutes=9), now],
    )

    summary = model_eval.performance_summary(conn, window_hours=1, resolve=False, initialize_schema=False)

    assert summary["overall"]["n"] == 1
    assert summary["model_leaderboard"][0]["model_key"] == "inventory_world"
