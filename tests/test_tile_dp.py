from __future__ import annotations

import math

from divvy import tile_dp


def test_zero_flows_preserve_state() -> None:
    result = tile_dp.tile_rollout(
        current_free_ebikes=3,
        depart_mean_per_horizon=0.0,
        arrive_mean_per_horizon=0.0,
        horizon=10,
    )
    assert math.isclose(result["p_has_bike"], 1.0, abs_tol=1e-9)
    assert math.isclose(result["p_survives"], 1.0, abs_tol=1e-9)
    assert math.isclose(result["expected_count"], 3.0, abs_tol=1e-6)
    assert math.isclose(result["expected_arrivals"], 0.0, abs_tol=1e-9)
    assert math.isclose(result["expected_departures"], 0.0, abs_tol=1e-9)


def test_empty_tile_only_appears_via_arrivals() -> None:
    result = tile_dp.tile_rollout(
        current_free_ebikes=0,
        depart_mean_per_horizon=0.0,
        arrive_mean_per_horizon=2.0,
        horizon=10,
    )
    assert result["p_survives"] == 0.0
    assert result["p_appears"] > 0.0
    assert math.isclose(result["p_appears"], result["p_has_bike"])
    assert result["expected_count"] > 0.0


def test_high_departure_collapses_inventory() -> None:
    busy = tile_dp.tile_rollout(
        current_free_ebikes=2,
        depart_mean_per_horizon=10.0,
        arrive_mean_per_horizon=0.0,
        horizon=10,
    )
    calm = tile_dp.tile_rollout(
        current_free_ebikes=2,
        depart_mean_per_horizon=0.1,
        arrive_mean_per_horizon=0.0,
        horizon=10,
    )
    assert busy["p_has_bike"] < calm["p_has_bike"]
    assert busy["expected_count"] < calm["expected_count"]
    assert busy["expected_departures"] > calm["expected_departures"]


def test_per_bike_survival_matches_thinned_poisson_closed_form() -> None:
    p_stays = tile_dp.per_bike_survival(
        depart_mean_per_horizon=2.0,
        current_free_ebikes=4,
        horizon=10,
    )
    # P(stays) = exp(-mu/n) = exp(-2/4) = exp(-0.5)
    assert math.isclose(p_stays, math.exp(-0.5), rel_tol=1e-9)


def test_per_bike_survival_unity_when_no_departures() -> None:
    assert tile_dp.per_bike_survival(
        depart_mean_per_horizon=0.0,
        current_free_ebikes=3,
        horizon=10,
    ) == 1.0


def test_per_bike_survival_handles_empty_tile() -> None:
    # Edge case: caller may ask about a horizon during which n becomes 0; we still
    # need a finite number. Divide-by-zero is handled by treating n=0 as n=1.
    p = tile_dp.per_bike_survival(
        depart_mean_per_horizon=1.0,
        current_free_ebikes=0,
        horizon=10,
    )
    assert 0.0 < p < 1.0
