"""Tests for the multi-bike acquisition planner and PMF rework."""
from __future__ import annotations

import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from divvy import inventory_dp, predictor, recommendations


# --- PMF rework --------------------------------------------------------------


def test_rollout_exposes_full_pmf() -> None:
    res = inventory_dp.rollout_inventory_distribution(
        capacity=10,
        current_ebikes=3,
        current_total_bikes=6,
        ebike_departure_mean=0.5,
        classic_departure_mean=0.2,
        ebike_arrival_mean=0.4,
        classic_arrival_mean=0.3,
    )
    assert res.p_count_ebikes_full is not None
    assert len(res.p_count_ebikes_full) == 11
    total = sum(res.p_count_ebikes_full)
    assert total == pytest.approx(1.0, abs=1e-6)
    assert res.p_count_ebikes_full[0] == pytest.approx(res.p_zero, abs=1e-6)


def test_rollout_multistep_exposes_full_pmf() -> None:
    res = inventory_dp.rollout_inventory_distribution_multistep(
        capacity=8,
        current_ebikes=2,
        current_total_bikes=4,
        intensity_sequence=[
            {"ebike_depart_mean": 0.05, "classic_depart_mean": 0.05,
             "ebike_arrive_mean": 0.07, "classic_arrive_mean": 0.04}
        ] * 5,
    )
    assert res.p_count_ebikes_full is not None
    assert len(res.p_count_ebikes_full) == 9
    assert sum(res.p_count_ebikes_full) == pytest.approx(1.0, abs=1e-6)


def test_align_count_full_to_p_zero_preserves_total() -> None:
    raw = [0.1, 0.4, 0.25, 0.15, 0.1]
    out = predictor._align_count_full_to_p_zero(raw, 0.4)
    assert sum(out) == pytest.approx(1.0)
    assert out[0] == pytest.approx(0.4)
    # Positive support rescaled but rank order preserved.
    assert out[1] > out[2] > out[3] > out[4]


def test_align_count_full_handles_all_zero_positive_mass() -> None:
    raw = [0.9, 0.0, 0.0, 0.0]
    out = predictor._align_count_full_to_p_zero(raw, 0.5)
    assert sum(out) == pytest.approx(1.0)
    assert out[0] == pytest.approx(0.5)
    assert out[1] == pytest.approx(0.5)


# --- live_cache encode/decode -----------------------------------------------


def test_count_ebikes_pmf_payload_includes_full() -> None:
    from divvy import live_cache

    row = pd.Series({
        "p_count_ebikes_10m_active": {"0": 0.2, "1": 0.4, "2": 0.3, "3": 0.1},
        "p_count_ebikes_full_10m_active": [0.2, 0.4, 0.3, 0.1],
        "active_model_key": "active",
    })
    payload = live_cache._count_ebikes_pmf_payload(row, 10, "active")
    assert payload is not None
    assert payload.get("__full__") == [0.2, 0.4, 0.3, 0.1]
    assert payload.get("0") == 0.2

    text = live_cache._json_text(payload)
    parsed = json.loads(text)
    assert parsed["__full__"] == [0.2, 0.4, 0.3, 0.1]
    assert parsed["1"] == 0.4


def test_count_ebikes_pmf_payload_legacy_bucket_only() -> None:
    from divvy import live_cache

    row = pd.Series({
        "p_count_ebikes_10m_active": {"0": 0.3, "1": 0.4, "2": 0.3},
        "active_model_key": "active",
    })
    payload = live_cache._count_ebikes_pmf_payload(row, 10, "active")
    assert payload is not None
    assert "__full__" not in payload
    assert payload.get("1") == 0.4


# --- recommendations helpers -------------------------------------------------


def test_station_count_pmf_prefers_full() -> None:
    row = pd.Series({
        "p_count_ebikes_10m": {"__full__": [0.1, 0.2, 0.3, 0.2, 0.1, 0.05, 0.05]},
        "num_ebikes_available": 3,
        "capacity": 6,
    })
    pmf, degraded = recommendations._station_count_pmf(row, 10)
    assert pmf.size == 7
    assert pmf.sum() == pytest.approx(1.0)
    assert degraded is False


def test_station_count_pmf_legacy_bucket_marks_degraded() -> None:
    row = pd.Series({
        "p_count_ebikes_10m": {"0": 0.4, "1": 0.3, "2": 0.2, "3": 0.05, "4": 0.03, "5_plus": 0.02},
        "num_ebikes_available": 0,
        "capacity": 5,
    })
    pmf, degraded = recommendations._station_count_pmf(row, 10)
    assert pmf.size == 6
    assert pmf.sum() == pytest.approx(1.0)
    assert degraded is True


def test_station_count_pmf_missing_falls_back_to_point_mass() -> None:
    row = pd.Series({
        "num_ebikes_available": 2,
        "capacity": 6,
    })
    pmf, degraded = recommendations._station_count_pmf(row, 10)
    assert degraded is True
    assert pmf.size == 3
    assert pmf[2] == pytest.approx(1.0)


def test_free_bike_pmf_uses_p_stays() -> None:
    bike = {"p_stays": {5: 0.9, 10: 0.7}}
    pmf = recommendations._free_bike_count_pmf(bike, 10)
    assert pmf[1] == pytest.approx(0.7)
    assert pmf[0] == pytest.approx(0.3)


def test_convolve_and_tail() -> None:
    a = np.array([0.5, 0.5])  # P(0)=0.5, P(1)=0.5
    b = np.array([0.4, 0.6])
    c = recommendations._convolve_pmf(a, b)
    # Combined: P(0)=.5*.4=.2, P(1)=.5*.6+.5*.4=.5, P(2)=.5*.6=.3
    assert c[0] == pytest.approx(0.2)
    assert c[1] == pytest.approx(0.5)
    assert c[2] == pytest.approx(0.3)
    assert recommendations._tail_prob_ge_k(c, 2) == pytest.approx(0.3)
    assert recommendations._tail_prob_ge_k(c, 1) == pytest.approx(0.8)
    assert recommendations._tail_prob_ge_k(c, 5) == pytest.approx(0.0)


def test_greedy_plan_builds_until_target() -> None:
    candidates = [
        {
            "stop_type": "station",
            "station_id": "s1",
            "name": "A",
            "lat": 41.88,
            "lon": -87.63,
            "distance_from_user_km": 0.1,
            "walk_minutes_from_user": 2.25,
            "horizon_minutes": 5,
            "pmf": np.array([0.1, 0.2, 0.7]),
            "expected_pickup": 1.6,
            "p_has_ebike_at_horizon": 0.9,
            "current_ebikes": 3,
            "capacity": 10,
            "degraded": False,
        },
        {
            "stop_type": "station",
            "station_id": "s2",
            "name": "B",
            "lat": 41.881,
            "lon": -87.631,
            "distance_from_user_km": 0.15,
            "walk_minutes_from_user": 2.9,
            "horizon_minutes": 5,
            "pmf": np.array([0.3, 0.5, 0.2]),
            "expected_pickup": 0.9,
            "p_has_ebike_at_horizon": 0.7,
            "current_ebikes": 2,
            "capacity": 10,
            "degraded": False,
        },
        {
            "stop_type": "free_bike",
            "bike_id": "b1",
            "name": "Bike-99",
            "lat": 41.882,
            "lon": -87.632,
            "distance_from_user_km": 0.18,
            "walk_minutes_from_user": 3.3,
            "horizon_minutes": 5,
            "pmf": np.array([0.15, 0.85]),
            "expected_pickup": 0.85,
            "p_has_ebike_at_horizon": 0.85,
            "p_stays_source": "tile_predictor",
            "degraded": False,
        },
    ]
    plan = recommendations._build_plan_greedy(
        candidates,
        k=2,
        target_probability=0.85,
        user_lat=41.88,
        user_lon=-87.63,
        walk_penalty=0.01,
        max_stops=4,
    )
    assert plan["success_probability"] >= 0.85
    assert plan["feasible"] is True
    # No internal "pmf" key in returned stops.
    for stop in plan["stops"]:
        assert "pmf" not in stop


def test_greedy_plan_k7_picks_multiple_stops() -> None:
    base = {
        "stop_type": "station",
        "lat": 41.88,
        "lon": -87.63,
        "distance_from_user_km": 0.1,
        "walk_minutes_from_user": 2.25,
        "horizon_minutes": 5,
        "expected_pickup": 1.5,
        "p_has_ebike_at_horizon": 0.9,
        "current_ebikes": 3,
        "capacity": 10,
        "degraded": False,
    }
    candidates = []
    rng = np.random.default_rng(0)
    for i in range(8):
        c = dict(base)
        c["station_id"] = f"s{i}"
        c["name"] = f"Station {i}"
        c["lat"] = 41.88 + 0.001 * i
        c["lon"] = -87.63 + 0.001 * i
        # Each station: PMF concentrated around 2 ebikes.
        full = np.zeros(11)
        full[1] = 0.15
        full[2] = 0.55
        full[3] = 0.25
        full[4] = 0.05
        full = full / full.sum()
        c["pmf"] = full
        candidates.append(c)
    plan = recommendations._build_plan_greedy(
        candidates,
        k=7,
        target_probability=0.85,
        user_lat=41.88,
        user_lon=-87.63,
        walk_penalty=0.01,
        max_stops=14,
    )
    assert plan["success_probability"] > 0.5
    assert len(plan["stops"]) >= 3


def test_plan_multi_bike_acquisition_end_to_end_with_stubs() -> None:
    """Integration-style: stub recommend() and tile_predictor.score_tiles to drive the planner."""
    fake_scored = pd.DataFrame([
        {
            "station_id": "s1",
            "name": "Station 1",
            "lat": 41.881,
            "lon": -87.631,
            "distance_km": 0.12,
            "num_ebikes_available": 4,
            "capacity": 19,
            "p_count_ebikes_10m": {"__full__": [0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05]},
        },
        {
            "station_id": "s2",
            "name": "Station 2",
            "lat": 41.882,
            "lon": -87.632,
            "distance_km": 0.18,
            "num_ebikes_available": 2,
            "capacity": 15,
            "p_count_ebikes_10m": {"__full__": [0.2, 0.3, 0.3, 0.15, 0.05]},
        },
    ])
    fake_recommend_payload = {
        "_scored_for_logging": fake_scored,
        "_recommended_station_ids": ["s1", "s2"],
        "_recommended_station_ids_by_model": {},
        "model": {"active_model_key": "stub"},
        "candidate_count": 2,
        "free_bike_error": None,
        "active_model_key": "stub",
        "prediction_cache_status": "miss_fallback",
        "db_freshness_minutes": 0.5,
        "query_place_key": "test",
    }
    fake_state_df = pd.DataFrame([
        {
            "tile_id": "tile_a",
            "bikes": [
                {
                    "bike_id": "b9",
                    "name": "Bike-9",
                    "lat": 41.883,
                    "lon": -87.633,
                    "p_stays": {5: 0.95, 10: 0.85, 15: 0.7, 20: 0.55},
                    "dwell_seconds_so_far": 300,
                },
            ],
        }
    ])
    with patch.object(recommendations, "recommend", return_value=fake_recommend_payload), \
         patch.object(recommendations.tile_predictor, "score_tiles", return_value=(fake_state_df, pd.DataFrame())), \
         patch.object(recommendations, "fetch_live_free_bike_ids", return_value={"b9"}):
        result = recommendations.plan_multi_bike_acquisition(
            conn=None,
            lat=41.88,
            lon=-87.63,
            k=3,
            include_internal=True,
        )

    assert result["k"] == 3
    assert result["candidate_count"] == 3  # 2 stations + 1 free bike
    plan_stops = result["plan"]["stops"]
    assert len(plan_stops) >= 1
    # The plan's recommended_station_ids must be a subset of the included stations,
    # and must NOT include the free bike id.
    assert set(result["recommended_station_ids"]).issubset({"s1", "s2"})
    assert "b9" not in result["recommended_station_ids"]
    assert all(stop.get("station_id") != "b9" for stop in plan_stops if stop.get("stop_type") == "station")
    # selected_free_bikes contains the bike with its source.
    if result["selected_free_bikes"]:
        assert result["selected_free_bikes"][0]["p_stays_source"] == "tile_predictor"


def test_fetch_free_bikes_with_persistence_filters_by_radius() -> None:
    """A bike outside the radius must be dropped after the post-filter."""
    fake_state_df = pd.DataFrame([
        {
            "tile_id": "tile_a",
            "bikes": [
                {"bike_id": "near", "name": "n", "lat": 41.881, "lon": -87.631,
                 "p_stays": {5: 0.9, 10: 0.8}, "dwell_seconds_so_far": 60},
                {"bike_id": "far", "name": "f", "lat": 41.92, "lon": -87.62,
                 "p_stays": {5: 0.9, 10: 0.8}, "dwell_seconds_so_far": 60},
            ],
        }
    ])
    with patch.object(recommendations.tile_predictor, "score_tiles",
                      return_value=(fake_state_df, pd.DataFrame())), \
         patch.object(recommendations, "fetch_live_free_bike_ids",
                      return_value={"near", "far"}):
        bikes = recommendations.fetch_free_bikes_with_persistence(
            conn=None, lat=41.88, lon=-87.63, radius_km=0.5
        )
    ids = {b["bike_id"] for b in bikes}
    assert "near" in ids
    assert "far" not in ids


# --- API endpoint ------------------------------------------------------------


def test_poisson_fallback_replaces_bucket_when_expected_present() -> None:
    """Bucket-only rows with expected_ebikes should yield a Poisson PMF, not the bucketed one."""
    row = pd.Series({
        "p_count_ebikes_10m": {"0": 0.005, "1": 0.0, "2": 0.0, "3": 0.0, "4": 0.0001, "5_plus": 0.995},
        "expected_ebikes_10m": 7.0,
        "num_ebikes_available": 7,
        "capacity": 15,
    })
    pmf, degraded = recommendations._station_count_pmf(row, 10)
    assert degraded is True
    # Poisson(7) on [0..15] should have P(=7) high and meaningful spread, not 99.5% lumped at index 5.
    assert pmf.size == 16  # capacity + 1
    # P(=7) is the mode for Poisson(7); should be ~0.149
    assert 0.1 < pmf[7] < 0.20
    # Tail mass at index 5+ should be < 0.99 (a real Poisson has spread).
    assert pmf[5:].sum() < 0.99
    # Should still sum to 1.
    assert pmf.sum() == pytest.approx(1.0, abs=1e-6)


def test_greedy_min_stops_picks_backup_after_target_met() -> None:
    """Once target probability is reached, Phase 2 should add backup stops favoring proximity."""
    # One station so good it satisfies P(>=2) >= 0.85 alone:
    big_station = {
        "stop_type": "station",
        "station_id": "big",
        "name": "Big",
        "lat": 41.92, "lon": -87.69,                       # far
        "distance_from_user_km": 0.9,
        "walk_minutes_from_user": 13.0,
        "horizon_minutes": 10,
        "expected_pickup": 7.0,
        "p_has_ebike_at_horizon": 0.99,
        "current_ebikes": 7,
        "capacity": 15,
        "degraded": False,
        "pmf": recommendations._poisson_pmf_truncated(7.0, 15),
    }
    # Nearby free bike that should be picked as a backup despite low expected (1.0):
    near_free = {
        "stop_type": "free_bike",
        "bike_id": "near",
        "name": "near-bike",
        "lat": 41.916, "lon": -87.687,                     # right at user
        "distance_from_user_km": 0.05,
        "walk_minutes_from_user": 1.6,
        "horizon_minutes": 10,
        "expected_pickup": 0.95,
        "p_has_ebike_at_horizon": 0.95,
        "p_stays_source": "tile_predictor",
        "degraded": False,
        "pmf": np.array([0.05, 0.95]),
    }
    plan = recommendations._build_plan_greedy(
        [big_station, near_free],
        k=2,
        target_probability=0.85,
        user_lat=41.916, user_lon=-87.687,
        walk_penalty=0.01,
        max_stops=4,
        min_stops=2,                # force at least 2 stops
    )
    types = [s["stop_type"] for s in plan["stops"]]
    assert "free_bike" in types
    assert "station" in types
    assert len(plan["stops"]) == 2


def test_unused_free_bikes_returned_when_not_picked() -> None:
    """If free bikes exist but the plan didn't need them, surface them in unused_free_bikes."""
    fake_scored = pd.DataFrame([
        {
            "station_id": "s1",
            "name": "Big",
            "lat": 41.92, "lon": -87.69,
            "distance_km": 0.9,
            "num_ebikes_available": 8,
            "capacity": 15,
            "expected_ebikes_10m": 8.0,
            "p_count_ebikes_10m": {"__full__": list(recommendations._poisson_pmf_truncated(8.0, 15))},
        },
    ])
    fake_recommend_payload = {
        "_scored_for_logging": fake_scored,
        "_recommended_station_ids": ["s1"],
        "_recommended_station_ids_by_model": {},
        "model": {"active_model_key": "stub"},
        "candidate_count": 1,
        "free_bike_error": None,
        "active_model_key": "stub",
        "prediction_cache_status": "miss_fallback",
        "db_freshness_minutes": 0.5,
        "query_place_key": "test",
    }
    fake_state_df = pd.DataFrame([
        {
            "tile_id": "tile_a",
            "bikes": [
                {"bike_id": "b1", "name": "Bike-1", "lat": 41.9165, "lon": -87.6875,
                 "p_stays": {5: 0.9, 10: 0.85}, "dwell_seconds_so_far": 120},
                {"bike_id": "b2", "name": "Bike-2", "lat": 41.9167, "lon": -87.6868,
                 "p_stays": {5: 0.9, 10: 0.80}, "dwell_seconds_so_far": 60},
            ],
        }
    ])
    with patch.object(recommendations, "recommend", return_value=fake_recommend_payload), \
         patch.object(recommendations.tile_predictor, "score_tiles", return_value=(fake_state_df, pd.DataFrame())), \
         patch.object(recommendations, "fetch_live_free_bike_ids", return_value={"b1", "b2"}):
        result = recommendations.plan_multi_bike_acquisition(
            conn=None, lat=41.9160608, lon=-87.6866236, k=2,
        )
    assert "unused_free_bikes" in result
    # Every free-bike candidate is either picked into the plan or surfaced as
    # a backup — never silently dropped.
    picked_ids = {s.get("bike_id") for s in result["plan"]["stops"] if s.get("stop_type") == "free_bike"}
    unused_ids = {b["bike_id"] for b in result["unused_free_bikes"]}
    assert picked_ids | unused_ids == {"b1", "b2"}
    assert not (picked_ids & unused_ids)  # disjoint


def test_multi_bike_request_validation_bounds() -> None:
    from divvy.api import MultiBikePlanRequest
    from pydantic import ValidationError

    # k must be in [1, 10].
    with pytest.raises(ValidationError):
        MultiBikePlanRequest(lat=41.88, lon=-87.63, k=0)
    with pytest.raises(ValidationError):
        MultiBikePlanRequest(lat=41.88, lon=-87.63, k=11)

    # search_radius cannot be smaller than near_radius.
    with pytest.raises(ValidationError):
        MultiBikePlanRequest(lat=41.88, lon=-87.63, k=3,
                             near_radius_km=1.5, search_radius_km=0.5)

    # valid one
    req = MultiBikePlanRequest(lat=41.88, lon=-87.63, k=5)
    assert req.k == 5
    assert req.search_radius_km == pytest.approx(1.5)
