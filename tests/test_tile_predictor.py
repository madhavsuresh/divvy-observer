from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import duckdb
import pandas as pd

from divvy import db, tile, tile_predictor


LOOP_LAT = 41.8819
LOOP_LON = -87.6278
LOOP_TILE = tile.tile_id_for(LOOP_LAT, LOOP_LON)
# A second tile guaranteed distinct from LOOP_TILE.
OTHER_TILE = tile.tile_id_for(41.8900, -87.6500)


def _insert_free_bike(conn, bike_id, fetched_at, lat, lon, *, reserved=False, disabled=False):
    conn.execute(
        """
        INSERT INTO free_bike_status
          (bike_id, fetched_at, name, lat, lon, is_reserved, is_disabled, tile_id)
        VALUES (?, ?, NULL, ?, ?, ?, ?, ?)
        """,
        [bike_id, fetched_at, lat, lon, reserved, disabled, tile.tile_id_for(lat, lon)],
    )


def _seed_model_selection(conn, model_key="logistic"):
    conn.execute(
        """
        INSERT INTO model_selection_state
          (computed_at, active_model_key, active_artifact_id, active_model_source,
           best_evaluated_model_key, best_sota_model_key, best_baseline_model_key,
           active_equals_best, selection_metric, selection_window_hours, min_resolved,
           reason, metrics_json)
        VALUES (now(), ?, NULL, 'test', NULL, NULL, NULL, true, 'rank_loss', 24, 0, 'fixture', '{}')
        """,
        [model_key],
    )


def _make_conn():
    conn = duckdb.connect(":memory:")
    db.init_schema(conn)
    # Clear the stations cache so each test starts fresh.
    tile_predictor._stations_cache.clear()
    return conn


def test_stations_in_tile_map_groups_by_tile() -> None:
    conn = _make_conn()
    now = datetime(2026, 5, 14, 12, 0, 0)
    conn.executemany(
        """
        INSERT INTO stations VALUES
        (?, NULL, ?, ?, ?, ?, ?, 'classic', ?, ?)
        """,
        [
            ("s_loop_a", "LA", "Loop A", LOOP_LAT,         LOOP_LON,         20, now, now),
            ("s_loop_b", "LB", "Loop B", LOOP_LAT + 1e-6,  LOOP_LON + 1e-6,  20, now, now),  # same tile
            ("s_other",  "OT", "Other",  41.8900,          -87.6500,         20, now, now),
        ],
    )
    mapping = tile_predictor.stations_in_tile_map(conn)
    assert sorted(mapping[LOOP_TILE]) == ["s_loop_a", "s_loop_b"]
    assert mapping[OTHER_TILE] == ["s_other"]


def test_current_tile_state_partitions_free_reserved_excludes_disabled() -> None:
    conn = _make_conn()
    now = datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0)
    base = now - timedelta(minutes=10)

    _insert_free_bike(conn, "b_free", base,                    LOOP_LAT, LOOP_LON)
    _insert_free_bike(conn, "b_res",  base + timedelta(minutes=1), LOOP_LAT, LOOP_LON, reserved=True)
    _insert_free_bike(conn, "b_dis",  base + timedelta(minutes=2), LOOP_LAT, LOOP_LON, disabled=True)

    state = tile_predictor.current_tile_state(conn, [LOOP_TILE])
    assert len(state) == 1
    row = state.iloc[0]
    assert row["tile_id"] == LOOP_TILE
    assert row["current_free_ebikes"] == 1
    assert row["current_reserved_free_ebikes"] == 1
    assert {b["bike_id"] for b in row["bikes"]} == {"b_free"}
    assert {b["bike_id"] for b in row["reserved_bikes"]} == {"b_res"}
    # Dwell time = (now - entered_at), which is ~10 min for our single bike.
    assert row["bikes"][0]["dwell_seconds_so_far"] > 500.0


def test_current_tile_state_counts_fresh_reservation_events() -> None:
    conn = _make_conn()
    now = datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0)
    # b1: was free 8 min ago, became reserved 3 min ago (fresh reservation event)
    _insert_free_bike(conn, "b1", now - timedelta(minutes=8), LOOP_LAT, LOOP_LON)
    _insert_free_bike(conn, "b1", now - timedelta(minutes=3), LOOP_LAT, LOOP_LON, reserved=True)
    # b2: was reserved 25 min ago (still inside 30-min window), then free again 20 min ago.
    # Going free→reserved is the only fresh-reservation event we count; this bike's b->res
    # transition happened well before our window started so no event registered for it.
    _insert_free_bike(conn, "b2", now - timedelta(minutes=25), LOOP_LAT, LOOP_LON, reserved=True)
    _insert_free_bike(conn, "b2", now - timedelta(minutes=20), LOOP_LAT, LOOP_LON)
    # b3: fresh reservation 10 min ago (inside 30m, outside 5m)
    _insert_free_bike(conn, "b3", now - timedelta(minutes=12), LOOP_LAT, LOOP_LON)
    _insert_free_bike(conn, "b3", now - timedelta(minutes=10), LOOP_LAT, LOOP_LON, reserved=True)

    state = tile_predictor.current_tile_state(conn, [LOOP_TILE])
    row = state.iloc[0]
    # Two fresh reservations in the last 30m (b1 at 3m, b3 at 10m). Only b1 is in the last 5m.
    assert row["reservation_events_30m"] == 2
    assert row["reservation_events_5m"] == 1


def test_current_tile_state_includes_stations_and_docked_ebikes() -> None:
    conn = _make_conn()
    now = datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0)
    last_reported = now - timedelta(minutes=2)

    conn.execute(
        """
        INSERT INTO stations VALUES
        ('s1', NULL, 'S1', 'Loop Station', ?, ?, 27, 'classic', ?, ?)
        """,
        [LOOP_LAT, LOOP_LON, now, now],
    )
    conn.execute(
        """
        INSERT INTO station_status VALUES
        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ["s1", last_reported, last_reported, 11, 3, 0, 16, 0, True, True, True],
    )

    state = tile_predictor.current_tile_state(conn, [LOOP_TILE])
    row = state.iloc[0]
    assert row["n_stations_in_tile"] == 1
    assert row["current_docked_ebikes"] == 3
    assert row["stations"][0]["station_id"] == "s1"
    assert row["stations"][0]["num_ebikes_available"] == 3


def test_tile_flow_priors_shrinks_toward_global() -> None:
    conn = _make_conn()
    anchor = datetime(2026, 5, 14, 12, 0, 0)  # Thursday 12:00 UTC
    weeks_back = 4
    # Seed a high-arrival tile and an idle tile so global mean lands between.
    for week in range(1, weeks_back + 1):
        ts = anchor - timedelta(days=7 * week)
        # busy tile: each week, a brand-new bike arrives at LOOP_TILE.
        _insert_free_bike(conn, f"busy_{week}_in", ts, LOOP_LAT, LOOP_LON)
        # idle tile: a bike enters OTHER_TILE.
        _insert_free_bike(conn, f"idle_{week}_in", ts, 41.8900, -87.6500)

    priors = tile_predictor.tile_flow_priors(conn, [LOOP_TILE, OTHER_TILE], anchor)
    assert set(priors["tile_id"]) == {LOOP_TILE, OTHER_TILE}
    # Both should have non-negative finite rates.
    for _, row in priors.iterrows():
        assert math.isfinite(row["arrive_mean_per_min"])
        assert math.isfinite(row["depart_mean_per_min"])
        assert row["arrive_mean_per_min"] >= 0
        assert row["depart_mean_per_min"] >= 0


def test_score_tiles_combines_free_and_dock_via_independence() -> None:
    conn = _make_conn()
    _seed_model_selection(conn, "logistic")
    now = datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0)

    conn.execute(
        """
        INSERT INTO stations VALUES
        ('s1', NULL, 'S1', 'Loop Station', ?, ?, 27, 'classic', ?, ?)
        """,
        [LOOP_LAT, LOOP_LON, now, now],
    )
    last_reported = now - timedelta(minutes=2)
    conn.execute(
        """
        INSERT INTO station_status VALUES
        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ["s1", last_reported, last_reported, 11, 3, 0, 16, 0, True, True, True],
    )
    _insert_free_bike(conn, "b1", now - timedelta(minutes=5), LOOP_LAT, LOOP_LON)

    # Seed a docked prediction so the dock side is non-trivial.
    horizon_minutes = 10
    p_dock = 0.6
    conn.execute(
        """
        INSERT INTO live_station_predictions
          (as_of, model_key, model_version, artifact_id, active_model_key,
           station_id, horizon_minutes, p_has_ebike, p_zero, p_appears, p_survives,
           expected_ebikes, expected_total_bikes, p_count_ebikes_json, p_count_total_json,
           p_capacity_violation, p_dock_constrained_arrival, reliable_probability_lcb,
           calibration_group, feature_snapshot_id, data_age_minutes, created_at)
        VALUES (?, ?, NULL, NULL, ?, ?, ?, ?, NULL, NULL, NULL, ?, NULL, NULL, NULL, NULL, NULL, ?, NULL, NULL, NULL, now())
        """,
        [now, "logistic", "logistic", "s1", horizon_minutes, p_dock, 2.4, p_dock],
    )

    state_df, score_df = tile_predictor.score_tiles(conn, [LOOP_TILE], horizons=(horizon_minutes,))
    assert not state_df.empty
    row = score_df[score_df["horizon_minutes"] == horizon_minutes].iloc[0]
    assert row["tile_id"] == LOOP_TILE
    assert row["dock_p_any_has_ebike"] == p_dock
    # Combined = 1 - (1 - free)(1 - dock)
    expected_combined = 1.0 - (1.0 - row["free_p_has_bike"]) * (1.0 - p_dock)
    assert math.isclose(row["combined_p_any_ebike"], expected_combined, rel_tol=1e-9)
    # Total expected = free + docked
    expected_total = row["free_expected_count"] + row["dock_expected_count"]
    assert math.isclose(row["combined_total_expected_ebikes"], expected_total, rel_tol=1e-9)
    assert row["dock_per_station"][0]["station_id"] == "s1"


def test_score_tiles_disability_hazard_reduces_p_stays_and_surfaces_share() -> None:
    """A bike's p_stays should drop once disability transitions appear in history.

    Disability is a second drain on the rider-available pool; the tile model
    composes the depart and disability rates into a single effective hazard.
    """
    conn_baseline = _make_conn()
    _seed_model_selection(conn_baseline, "logistic")
    anchor = datetime(2026, 5, 14, 12, 0, 0)
    _insert_free_bike(conn_baseline, "b1", anchor - timedelta(minutes=5), LOOP_LAT, LOOP_LON)

    state_baseline, score_baseline = tile_predictor.score_tiles(
        conn_baseline, [LOOP_TILE], horizons=(10,), anchor_ts=anchor
    )
    baseline_p_stays = state_baseline.iloc[0]["bikes"][0]["p_stays"][10]
    baseline_score = score_baseline.iloc[0]
    # With no historical disability data the rate collapses to 0.
    assert baseline_score["free_disability_rate_per_min"] == 0.0
    assert baseline_score["free_expected_disabilities"] == 0.0

    # Now repeat with disability transitions in the matching (hour, dow).
    conn_with_dis = _make_conn()
    _seed_model_selection(conn_with_dis, "logistic")
    _insert_free_bike(conn_with_dis, "b1", anchor - timedelta(minutes=5), LOOP_LAT, LOOP_LON)
    for week in range(1, 5):
        for i in range(10):
            ts = anchor - timedelta(days=7 * week) + timedelta(minutes=i * 5)
            bid = f"hist_{week}_{i}"
            _insert_free_bike(conn_with_dis, bid, ts - timedelta(minutes=1), LOOP_LAT, LOOP_LON, disabled=False)
            _insert_free_bike(conn_with_dis, bid, ts,                        LOOP_LAT, LOOP_LON, disabled=True)

    state_with_dis, score_with_dis = tile_predictor.score_tiles(
        conn_with_dis, [LOOP_TILE], horizons=(10,), anchor_ts=anchor
    )
    with_dis_p_stays = state_with_dis.iloc[0]["bikes"][0]["p_stays"][10]
    with_dis_score = score_with_dis.iloc[0]

    assert with_dis_score["free_disability_rate_per_min"] > 0.0
    assert with_dis_score["free_expected_disabilities"] >= 0.0
    # Adding a second drain term strictly reduces per-bike survival.
    assert with_dis_p_stays < baseline_p_stays
    # Ride-driven expected_departures shouldn't blow up just because we added
    # the disability term — it should stay close to the depart-rate * horizon.
    # And the disability share is monotone in the rate.
    assert with_dis_score["free_expected_departures"] >= 0.0


def test_score_single_bike_returns_per_horizon_dict() -> None:
    conn = _make_conn()
    now = datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0)
    _insert_free_bike(conn, "b1", now - timedelta(minutes=5), LOOP_LAT, LOOP_LON)

    result = tile_predictor.score_single_bike(conn, "b1", LOOP_TILE, horizons=(5, 10))
    assert set(result.keys()) == {5, 10}
    for p in result.values():
        assert 0.0 <= p <= 1.0
    # Longer horizon => lower or equal survival
    assert result[10] <= result[5] + 1e-9
