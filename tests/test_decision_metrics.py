from __future__ import annotations

import pytest

from divvy import decision_metrics


def test_decision_metrics_compute_ece_and_regret() -> None:
    ece = decision_metrics.ece_score([1, 0, 1, 0], [0.9, 0.1, 0.6, 0.4], n_bins=2)
    regret = decision_metrics.distance_adjusted_regret(0.7, 1.0)

    assert ece is not None
    assert 0 <= ece <= 1
    assert regret == pytest.approx(0.3)
