from __future__ import annotations

import numpy as np
import pytest

from divvy import inventory_dp


def test_transition_distribution_enforces_feasible_states() -> None:
    c = 4
    pi = np.zeros((c + 1, c + 1))
    pi[0, 4] = 1.0
    out, metrics = inventory_dp.transition_distribution(
        c,
        pi,
        np.array([0.0, 1.0]),
        np.array([1.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0]),
    )

    assert out.sum() == pytest.approx(1.0)
    assert metrics["expected_ebike_departures"] == pytest.approx(0.0)
    for e in range(c + 1):
        for q in range(c + 1):
            if not (0 <= e <= q <= c):
                assert out[e, q] == 0.0


def test_transition_flags_suppress_flows() -> None:
    c = 3
    pi = np.zeros((c + 1, c + 1))
    pi[1, 3] = 1.0
    out, metrics = inventory_dp.transition_distribution(
        c,
        pi,
        np.array([0.0, 1.0]),
        np.array([0.0, 1.0]),
        np.array([0.0, 1.0]),
        np.array([0.0, 1.0]),
        is_renting=False,
        is_returning=False,
    )

    assert out[1, 3] == pytest.approx(1.0)
    assert metrics["expected_ebike_departures"] == pytest.approx(0.0)
    assert metrics["expected_ebike_arrivals"] == pytest.approx(0.0)


def test_zinb_pmf_sums_to_one() -> None:
    pmf = inventory_dp.zinb_pmf(mean=2.0, theta=3.0, zero_inflation=0.25, max_k=8)
    assert pmf.sum() == pytest.approx(1.0)
    assert pmf[0] >= 0.25
