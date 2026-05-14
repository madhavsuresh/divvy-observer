from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import duckdb
import pandas as pd
import pytest

from divvy import db, disabled_predictor, tile, tile_predictor


LOOP_LAT = 41.8819
LOOP_LON = -87.6278
LOOP_TILE = tile.tile_id_for(LOOP_LAT, LOOP_LON)
OTHER_LAT = 41.8900
OTHER_LON = -87.6500
OTHER_TILE = tile.tile_id_for(OTHER_LAT, OTHER_LON)


def _make_conn():
    conn = duckdb.connect(":memory:")
    db.init_schema(conn)
    tile_predictor._stations_cache.clear()
    return conn


def _insert_free_bike(conn, bike_id, fetched_at, lat, lon, *, reserved=False, disabled=False):
    conn.execute(
        """
        INSERT INTO free_bike_status
          (bike_id, fetched_at, name, lat, lon, is_reserved, is_disabled, tile_id)
        VALUES (?, ?, NULL, ?, ?, ?, ?, ?)
        """,
        [bike_id, fetched_at, lat, lon, reserved, disabled, tile.tile_id_for(lat, lon)],
    )


def _insert_station(conn, station_id, lat, lon, now):
    conn.execute(
        """
        INSERT INTO stations VALUES
        (?, NULL, ?, ?, ?, ?, 27, 'classic', ?, ?)
        """,
        [station_id, station_id, f"Station {station_id}", lat, lon, now, now],
    )


def _insert_station_status(
    conn,
    station_id,
    last_reported,
    *,
    bikes=11,
    ebikes=3,
    bikes_disabled=0,
    docks_available=16,
    docks_disabled=0,
):
    conn.execute(
        """
        INSERT INTO station_status VALUES
        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            station_id,
            last_reported,
            last_reported,
            bikes,
            ebikes,
            bikes_disabled,
            docks_available,
            docks_disabled,
            True,
            True,
            True,
        ],
    )


# ---------------------------------------------------------------------------
# current_tile_disability_state
# ---------------------------------------------------------------------------

def test_current_state_tracks_disabled_free_bike_with_dwell_time() -> None:
    conn = _make_conn()
    now = datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0)
    # b1: was free 30 min ago, became disabled 15 min ago. Latest event is the disabled one.
    _insert_free_bike(conn, "b1", now - timedelta(minutes=30), LOOP_LAT, LOOP_LON, disabled=False)
    _insert_free_bike(conn, "b1", now - timedelta(minutes=15), LOOP_LAT, LOOP_LON, disabled=True)

    state = disabled_predictor.current_tile_disability_state(conn, [LOOP_TILE])
    assert len(state) == 1
    row = state.iloc[0]
    assert row["tile_id"] == LOOP_TILE
    assert row["current_disabled_free_ebikes"] == 1
    assert row["current_disabled_docked_bikes"] == 0
    bikes = row["disabled_free_bikes"]
    assert len(bikes) == 1
    # disabled_since should match the false→true transition, so dwell ~15 min.
    dwell_minutes = bikes[0]["dwell_seconds_so_far"] / 60.0
    assert 13.5 < dwell_minutes < 16.5


def test_current_state_counts_free_disability_and_repair_transitions() -> None:
    conn = _make_conn()
    now = datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0)
    # b1: false→true (disability event) 6h ago.
    _insert_free_bike(conn, "b1", now - timedelta(hours=10), LOOP_LAT, LOOP_LON, disabled=False)
    _insert_free_bike(conn, "b1", now - timedelta(hours=6),  LOOP_LAT, LOOP_LON, disabled=True)
    # b2: true→false (repair event) 2h ago, still in tile.
    _insert_free_bike(conn, "b2", now - timedelta(hours=8), LOOP_LAT, LOOP_LON, disabled=True)
    _insert_free_bike(conn, "b2", now - timedelta(hours=2), LOOP_LAT, LOOP_LON, disabled=False)
    # b3 in OTHER_TILE — should not be counted under LOOP_TILE.
    _insert_free_bike(conn, "b3", now - timedelta(hours=4), OTHER_LAT, OTHER_LON, disabled=False)
    _insert_free_bike(conn, "b3", now - timedelta(hours=1), OTHER_LAT, OTHER_LON, disabled=True)

    state = disabled_predictor.current_tile_disability_state(conn, [LOOP_TILE, OTHER_TILE])
    loop = state[state["tile_id"] == LOOP_TILE].iloc[0]
    other = state[state["tile_id"] == OTHER_TILE].iloc[0]

    assert loop["free_disability_events_24h"] == 1  # only b1
    assert loop["free_repair_events_24h"] == 1      # only b2
    assert loop["current_disabled_free_ebikes"] == 1  # b1 still disabled

    assert other["free_disability_events_24h"] == 1  # b3
    assert other["current_disabled_free_ebikes"] == 1


def test_current_state_aggregates_docked_disability_across_stations_and_counts_events() -> None:
    conn = _make_conn()
    now = datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0)
    _insert_station(conn, "s1", LOOP_LAT, LOOP_LON, now)
    _insert_station(conn, "s2", LOOP_LAT + 1e-6, LOOP_LON + 1e-6, now)  # same tile

    # s1 history: 0 → 2 (disability events=2) → 1 (repair event=1). Final state: 1.
    _insert_station_status(conn, "s1", now - timedelta(hours=8), bikes_disabled=0)
    _insert_station_status(conn, "s1", now - timedelta(hours=5), bikes_disabled=2)
    _insert_station_status(conn, "s1", now - timedelta(hours=2), bikes_disabled=1)

    # s2 history: 3 → 5 (disability events=2). Final state: 5.
    _insert_station_status(conn, "s2", now - timedelta(hours=6), bikes_disabled=3, docks_disabled=0)
    _insert_station_status(conn, "s2", now - timedelta(hours=3), bikes_disabled=5, docks_disabled=2)

    state = disabled_predictor.current_tile_disability_state(conn, [LOOP_TILE])
    row = state.iloc[0]
    # Latest state: s1=1, s2=5, totalling 6.
    assert row["current_disabled_docked_bikes"] == 6
    assert row["current_disabled_docks"] == 2
    # Transition counts: 2 (s1 0→2) + 2 (s2 3→5) = 4 disability events;
    # 1 (s1 2→1) repair event.
    assert row["dock_bike_disability_events_24h"] == 4
    assert row["dock_bike_repair_events_24h"] == 1
    # Combined event counts include the free side (zero here).
    assert row["disability_events_24h"] == 4
    assert row["repair_events_24h"] == 1
    # Dock-only dock-events came from s2.
    assert row["dock_disability_events_24h"] == 2
    # Bike-hours of disability is the integral of the count over time and
    # should be strictly positive given the history above.
    assert row["bike_hours_disabled_24h"] > 0
    # n_stations_in_tile should reflect both stations.
    assert row["n_stations_in_tile"] == 2


def test_current_state_handles_empty_tile() -> None:
    conn = _make_conn()
    state = disabled_predictor.current_tile_disability_state(conn, [LOOP_TILE])
    assert len(state) == 1
    row = state.iloc[0]
    assert row["current_disabled_free_ebikes"] == 0
    assert row["current_disabled_docked_bikes"] == 0
    assert row["disability_events_24h"] == 0
    assert row["disabled_free_bikes"] == []
    assert row["disabled_stations"] == []


def test_current_state_returns_empty_frame_for_no_tile_ids() -> None:
    conn = _make_conn()
    state = disabled_predictor.current_tile_disability_state(conn, [])
    assert state.empty


# ---------------------------------------------------------------------------
# dwell_time_for_disabled
# ---------------------------------------------------------------------------

def test_dwell_time_reports_median_p90_for_free_and_bike_hours_for_docked() -> None:
    conn = _make_conn()
    now = datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0)
    # Three currently-disabled free bikes with dwells 10, 30, 120 minutes.
    for i, mins in enumerate([10, 30, 120]):
        bid = f"b{i}"
        _insert_free_bike(conn, bid, now - timedelta(minutes=mins + 5), LOOP_LAT, LOOP_LON, disabled=False)
        _insert_free_bike(conn, bid, now - timedelta(minutes=mins),     LOOP_LAT, LOOP_LON, disabled=True)
    # A station with a non-trivial bike-hours-disabled integral.
    _insert_station(conn, "s1", LOOP_LAT, LOOP_LON, now)
    _insert_station_status(conn, "s1", now - timedelta(hours=10), bikes_disabled=0)
    _insert_station_status(conn, "s1", now - timedelta(hours=4),  bikes_disabled=2)
    _insert_station_status(conn, "s1", now - timedelta(hours=1),  bikes_disabled=2)

    df = disabled_predictor.dwell_time_for_disabled(conn, [LOOP_TILE])
    assert len(df) == 1
    row = df.iloc[0]
    assert row["n_disabled_free"] == 3
    # Median of [10, 30, 120] = 30; p90 = 102 (90th percentile of {10,30,120}).
    assert 28 <= row["median_free_dwell_minutes"] <= 32
    assert 95 <= row["p90_free_dwell_minutes"] <= 125
    # bike_hours_disabled should be positive (the s1 history has 6h at 2-bike avg
    # then 3h at 2-bike avg => ~6+6=12 bike-hours, roughly).
    assert row["bike_hours_disabled_24h"] > 5.0


def test_dwell_time_returns_zeros_when_no_disabled_bikes() -> None:
    conn = _make_conn()
    df = disabled_predictor.dwell_time_for_disabled(conn, [LOOP_TILE])
    row = df.iloc[0]
    assert row["n_disabled_free"] == 0
    assert row["median_free_dwell_minutes"] == 0.0
    assert row["p90_free_dwell_minutes"] == 0.0
    assert row["bike_hours_disabled_24h"] == 0.0


# ---------------------------------------------------------------------------
# repair_rate_priors and predict_repair_time
# ---------------------------------------------------------------------------

def test_repair_rate_priors_shrinks_toward_global() -> None:
    conn = _make_conn()
    anchor = datetime(2026, 5, 14, 12, 0, 0)  # Thursday 12:00 UTC
    _insert_station(conn, "s_loop", LOOP_LAT, LOOP_LON, anchor)
    _insert_station(conn, "s_other", OTHER_LAT, OTHER_LON, anchor)

    # Seed 4 weeks of historical observations at the same hour-of-day. For each
    # week, a disabled count rises then falls, generating repair events.
    weeks_back = 4
    for week in range(1, weeks_back + 1):
        base = anchor - timedelta(days=7 * week)
        # busy tile s_loop: 0 → 3 → 0 (3 disability events, 3 repair events)
        _insert_station_status(conn, "s_loop", base - timedelta(minutes=10), bikes_disabled=0)
        _insert_station_status(conn, "s_loop", base,                         bikes_disabled=3)
        _insert_station_status(conn, "s_loop", base + timedelta(minutes=30), bikes_disabled=0)
        # idle tile s_other: 0 (no changes)
        _insert_station_status(conn, "s_other", base, bikes_disabled=0)

    priors = disabled_predictor.repair_rate_priors(
        conn, [LOOP_TILE, OTHER_TILE], anchor, lookback_days=28
    )
    by_tile = {r["tile_id"]: r for _, r in priors.iterrows()}
    # Both rates should be finite and non-negative.
    assert by_tile[LOOP_TILE]["repair_rate_per_hour"] >= 0
    assert by_tile[OTHER_TILE]["repair_rate_per_hour"] >= 0
    # The busy tile should have observed strictly more events than the idle one.
    assert by_tile[LOOP_TILE]["observed_events"] > by_tile[OTHER_TILE]["observed_events"]
    # Both tiles see the same global_rate.
    assert by_tile[LOOP_TILE]["global_rate_per_hour"] == by_tile[OTHER_TILE]["global_rate_per_hour"]


def test_predict_repair_time_returns_horizon_dict_with_valid_probabilities() -> None:
    conn = _make_conn()
    anchor = datetime(2026, 5, 14, 12, 0, 0)
    _insert_station(conn, "s1", LOOP_LAT, LOOP_LON, anchor)
    # Seed some history so the EB rate is non-zero.
    for week in range(1, 5):
        base = anchor - timedelta(days=7 * week)
        _insert_station_status(conn, "s1", base - timedelta(minutes=10), bikes_disabled=0)
        _insert_station_status(conn, "s1", base,                         bikes_disabled=2)
        _insert_station_status(conn, "s1", base + timedelta(minutes=30), bikes_disabled=0)

    result = disabled_predictor.predict_repair_time(
        conn, LOOP_TILE, current_disabled_count=3, anchor_ts=anchor
    )
    assert set(result.keys()) == set(disabled_predictor.HORIZONS_HOURS)
    for h, payload in result.items():
        assert 0.0 <= payload["p_any_repair"] <= 1.0
        assert 0.0 <= payload["p_all_repaired"] <= 1.0
        assert payload["expected_repairs"] >= 0.0
        assert payload["rate_per_hour"] >= 0.0
    # Longer horizons should have monotonically non-decreasing P(any repair).
    horizons_sorted = sorted(result.keys())
    p_seq = [result[h]["p_any_repair"] for h in horizons_sorted]
    assert all(a <= b + 1e-9 for a, b in zip(p_seq[:-1], p_seq[1:]))


def test_predict_repair_time_zero_disabled_count_gives_zero_event_probability() -> None:
    conn = _make_conn()
    anchor = datetime(2026, 5, 14, 12, 0, 0)
    _insert_station(conn, "s1", LOOP_LAT, LOOP_LON, anchor)
    # Some history so the rate is non-zero.
    for week in range(1, 5):
        base = anchor - timedelta(days=7 * week)
        _insert_station_status(conn, "s1", base - timedelta(minutes=10), bikes_disabled=0)
        _insert_station_status(conn, "s1", base,                         bikes_disabled=2)
        _insert_station_status(conn, "s1", base + timedelta(minutes=30), bikes_disabled=0)

    result = disabled_predictor.predict_repair_time(
        conn, LOOP_TILE, current_disabled_count=0, anchor_ts=anchor
    )
    for payload in result.values():
        # No bikes at risk => zero event probability.
        assert payload["p_any_repair"] == 0.0
        assert payload["expected_repairs"] == 0.0
        # P(all repaired) = 1 when there's nothing to repair.
        assert payload["p_all_repaired"] == 1.0


def test_predict_repair_time_empty_history_returns_zero_rates() -> None:
    conn = _make_conn()
    anchor = datetime(2026, 5, 14, 12, 0, 0)
    result = disabled_predictor.predict_repair_time(
        conn, LOOP_TILE, current_disabled_count=2, anchor_ts=anchor
    )
    for payload in result.values():
        assert payload["rate_per_hour"] == 0.0
        # With no historical repair rate, P(any repair) collapses to 0.
        assert payload["p_any_repair"] == 0.0


# ---------------------------------------------------------------------------
# score_tiles_disability
# ---------------------------------------------------------------------------

def test_score_tiles_disability_combines_state_and_horizon_scores() -> None:
    conn = _make_conn()
    now = datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0)
    _insert_station(conn, "s1", LOOP_LAT, LOOP_LON, now)
    _insert_station_status(conn, "s1", now - timedelta(hours=4), bikes_disabled=0)
    _insert_station_status(conn, "s1", now - timedelta(hours=2), bikes_disabled=3)
    _insert_free_bike(conn, "b1", now - timedelta(minutes=45), LOOP_LAT, LOOP_LON, disabled=False)
    _insert_free_bike(conn, "b1", now - timedelta(minutes=30), LOOP_LAT, LOOP_LON, disabled=True)

    state_df, score_df = disabled_predictor.score_tiles_disability(
        conn, [LOOP_TILE], anchor_ts=now, horizons_hours=(1.0, 6.0)
    )
    assert not state_df.empty
    state_row = state_df.iloc[0]
    assert state_row["current_disabled_free_ebikes"] == 1
    assert state_row["current_disabled_docked_bikes"] == 3
    assert state_row["median_free_dwell_minutes"] > 0

    # Two horizons should be present.
    assert sorted(score_df["horizon_hours"].tolist()) == [1.0, 6.0]
    for _, srow in score_df.iterrows():
        assert srow["n_disabled"] == 4
        assert 0.0 <= srow["p_any_repair"] <= 1.0
        assert 0.0 <= srow["p_all_repaired"] <= 1.0
        assert srow["expected_repairs"] >= 0.0


def test_score_tiles_disability_returns_empty_for_unseen_tile() -> None:
    conn = _make_conn()
    state_df, score_df = disabled_predictor.score_tiles_disability(
        conn, [], anchor_ts=datetime(2026, 5, 14, 12, 0, 0)
    )
    assert state_df.empty
    assert score_df.empty


# ---------------------------------------------------------------------------
# free_disability_rate_priors — feeds into tile_predictor.score_tiles.
# ---------------------------------------------------------------------------

def test_free_disability_rate_priors_picks_up_matching_hour_dow_transitions() -> None:
    conn = _make_conn()
    anchor = datetime(2026, 5, 14, 12, 0, 0)  # Thursday 12:00 UTC

    # Seed false→true transitions at the matching hour/dow over 4 weeks.
    for week in range(1, 5):
        for i in range(6):
            ts = anchor - timedelta(days=7 * week) + timedelta(minutes=i * 5)
            bid = f"hist_{week}_{i}"
            _insert_free_bike(conn, bid, ts - timedelta(minutes=1), LOOP_LAT, LOOP_LON, disabled=False)
            _insert_free_bike(conn, bid, ts,                        LOOP_LAT, LOOP_LON, disabled=True)

    priors = disabled_predictor.free_disability_rate_priors(
        conn, [LOOP_TILE, OTHER_TILE], anchor, lookback_days=28
    )
    by_tile = {r["tile_id"]: float(r["disability_rate_per_min"]) for _, r in priors.iterrows()}
    # LOOP_TILE saw real events; OTHER_TILE saw none. Both should be finite and
    # non-negative; LOOP_TILE should be strictly larger.
    assert by_tile[LOOP_TILE] > 0
    assert by_tile[OTHER_TILE] >= 0
    assert by_tile[LOOP_TILE] > by_tile[OTHER_TILE]


def test_free_disability_rate_priors_zero_when_no_history() -> None:
    conn = _make_conn()
    anchor = datetime(2026, 5, 14, 12, 0, 0)
    priors = disabled_predictor.free_disability_rate_priors(
        conn, [LOOP_TILE], anchor, lookback_days=28
    )
    # With no transitions anywhere, the global rate is 0 and the EB estimate
    # collapses to 0 too.
    assert float(priors["disability_rate_per_min"].iloc[0]) == 0.0


def test_free_disability_rate_priors_returns_empty_for_no_tile_ids() -> None:
    conn = _make_conn()
    priors = disabled_predictor.free_disability_rate_priors(
        conn, [], datetime(2026, 5, 14, 12, 0, 0)
    )
    assert priors.empty
