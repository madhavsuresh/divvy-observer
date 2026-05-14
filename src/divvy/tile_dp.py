"""Tile-level forward distribution for free-floating ebikes.

Thin wrapper around ``inventory_dp.rollout_inventory_distribution_multistep`` —
treats a tile like a station with effectively infinite capacity (no dock
constraints) and zero classic-bike flow. The same ZINB intensity machinery
gives us a forward count distribution per horizon; from it we derive the same
``p_has_ebike``/``p_appears``/``p_survives`` family the dock predictor uses.

Per-bike survival is the thinned-Poisson closed form
``P(stays h min) = exp(-h * μ_depart / n)``, where μ_depart is the tile-level
departure intensity and n is the current free-ebike count. Each bike is one
of n indistinguishable targets for the next departure event.
"""
from __future__ import annotations

import math
from typing import Any

from . import inventory_dp


TILE_CAPACITY_CAP = 40  # non-binding ceiling; well above any observed tile occupancy


def tile_rollout(
    *,
    current_free_ebikes: int,
    depart_mean_per_horizon: float,
    arrive_mean_per_horizon: float,
    horizon: int,
    capacity_cap: int = TILE_CAPACITY_CAP,
    theta: float = 20.0,
    zeta: float = 0.0,
) -> dict[str, Any]:
    """Return the forward distribution and survival probabilities for one tile.

    ``depart_mean_per_horizon`` and ``arrive_mean_per_horizon`` are the expected
    *total* ebike departures / arrivals across the full ``horizon`` minutes.
    Per-minute intensities are derived by dividing by ``horizon`` and replicated
    across the rollout's minute-by-minute steps.
    """
    horizon = max(1, int(horizon))
    intensity = {
        "ebike_depart_mean": max(0.0, float(depart_mean_per_horizon)) / horizon,
        "ebike_arrive_mean": max(0.0, float(arrive_mean_per_horizon)) / horizon,
        "ebike_depart_theta": theta,
        "ebike_arrive_theta": theta,
        "ebike_depart_zero_inflation": zeta,
        "ebike_arrive_zero_inflation": zeta,
        "classic_depart_mean": 0.0,
        "classic_arrive_mean": 0.0,
    }
    res = inventory_dp.rollout_inventory_distribution_multistep(
        capacity=capacity_cap,
        current_ebikes=int(current_free_ebikes),
        current_total_bikes=int(current_free_ebikes),
        intensity_sequence=[intensity] * horizon,
    )
    p_has_bike = float(res.p_has_ebike)
    has_now = int(current_free_ebikes) > 0
    return {
        "p_has_bike": p_has_bike,
        "p_zero": float(res.p_zero),
        "p_survives": p_has_bike if has_now else 0.0,
        "p_appears": 0.0 if has_now else p_has_bike,
        "expected_count": float(res.expected_ebikes),
        "p_count": dict(res.p_count_ebikes),
        "expected_arrivals": float(res.expected_ebike_arrivals),
        "expected_departures": float(res.expected_ebike_departures),
    }


def per_bike_survival(
    *,
    depart_mean_per_horizon: float,
    current_free_ebikes: int,
    horizon: int,
) -> float:
    """Closed-form thinned-Poisson survival probability for one specific bike.

    Treats the tile's departures as a Poisson process with intensity
    ``depart_mean_per_horizon`` over the next ``horizon`` minutes. Each event
    selects one of the ``current_free_ebikes`` bikes uniformly at random, so
    the per-bike departure intensity is ``depart_mean_per_horizon / n`` and
    the survival probability is ``exp(-h · μ / n)``. (Both ``h`` and the
    division-by-h cancel because depart_mean_per_horizon is already aggregated
    across the full horizon — we want survival across that same window.)
    """
    n = max(1, int(current_free_ebikes))
    mu = max(0.0, float(depart_mean_per_horizon))
    rate = mu / n
    return math.exp(-rate)
