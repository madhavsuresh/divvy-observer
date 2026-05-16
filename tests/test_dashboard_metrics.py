"""Tests for the calibration math in ``divvy.dashboard_metrics``.

These cover the pure-math primitives without touching DuckDB; the resolved
forecast pull (``resolved_forecasts``) is exercised through the dashboard's
integration test path in ``test_dashboard_payload``.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from divvy import dashboard_metrics as dm


# ---------------------------------------------------------------------------
# Wilson interval
# ---------------------------------------------------------------------------


def test_wilson_interval_bounds_within_unit_interval():
    lo, hi = dm._wilson_interval(50, 100)
    assert 0.0 <= lo <= hi <= 1.0


def test_wilson_interval_unanimous():
    # All successes: lower bound > 0, upper bound = 1.
    lo, hi = dm._wilson_interval(10, 10)
    assert lo > 0.0
    assert hi == pytest.approx(1.0, abs=1e-9)


def test_wilson_interval_empty_returns_full_range():
    lo, hi = dm._wilson_interval(0, 0)
    assert (lo, hi) == (0.0, 1.0)


# ---------------------------------------------------------------------------
# Reliability curve
# ---------------------------------------------------------------------------


def _synthetic_calibrated(n: int = 5000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    p = rng.uniform(0, 1, size=n)
    y = (rng.uniform(0, 1, size=n) < p).astype(int)
    return pd.DataFrame({
        "model_key": ["m"] * n,
        "model_label": ["m"] * n,
        "p_has_ebike": p,
        "observed_has_ebike": y,
    })


def test_reliability_curve_calibrated_data_hugs_diagonal():
    df = _synthetic_calibrated(n=10_000)
    curve = dm.reliability_curve(df, n_bins=10, min_per_bin=50)
    # Each bin's observed rate should be within ~0.05 of the predicted mean
    # for calibrated data at this sample size.
    gaps = (curve["predicted_mean"] - curve["observed_rate"]).abs()
    assert gaps.max() < 0.06, f"Largest gap = {gaps.max():.3f}"


def test_reliability_curve_empty_input_returns_empty_frame():
    df = pd.DataFrame()
    out = dm.reliability_curve(df)
    assert out.empty


def test_reliability_curve_returns_n_and_ci_columns():
    df = _synthetic_calibrated(n=500)
    curve = dm.reliability_curve(df, min_per_bin=5)
    for col in ("predicted_mean", "observed_rate", "observed_ci_low", "observed_ci_high", "n"):
        assert col in curve.columns
    # CI ordering invariant.
    assert (curve["observed_ci_low"] <= curve["observed_rate"] + 1e-9).all()
    assert (curve["observed_rate"] <= curve["observed_ci_high"] + 1e-9).all()


# ---------------------------------------------------------------------------
# Score distribution
# ---------------------------------------------------------------------------


def test_score_distribution_densities_sum_to_one_per_outcome():
    df = _synthetic_calibrated(n=2000)
    dist = dm.score_distribution(df, n_bins=20)
    # Per-outcome densities sum to 1 (the histogram is relative-frequency normalized).
    for outcome, group in dist.groupby("outcome"):
        assert group["density"].sum() == pytest.approx(1.0, abs=1e-9)


def test_score_distribution_well_discriminating_separates_means():
    rng = np.random.default_rng(42)
    n = 5000
    y = rng.integers(0, 2, size=n)
    # Well-discriminating: when y=1 push prob near 0.85, when y=0 push near 0.15.
    p = np.where(y == 1, rng.beta(8, 2, size=n), rng.beta(2, 8, size=n))
    df = pd.DataFrame({
        "model_key": ["m"] * n, "model_label": ["m"] * n,
        "p_has_ebike": p, "observed_has_ebike": y,
    })
    dist = dm.score_distribution(df)
    mean_when_pos = (dist[dist["outcome"] == "y=1 (had bike)"]["bin_mid"]
                     * dist[dist["outcome"] == "y=1 (had bike)"]["density"]).sum()
    mean_when_neg = (dist[dist["outcome"] == "y=0 (no bike)"]["bin_mid"]
                     * dist[dist["outcome"] == "y=0 (no bike)"]["density"]).sum()
    assert mean_when_pos - mean_when_neg > 0.4


# ---------------------------------------------------------------------------
# Sharpness
# ---------------------------------------------------------------------------


def test_sharpness_variance_extremes():
    # Always 0.5 → max variance.
    assert dm.sharpness_variance([0.5] * 10) == pytest.approx(0.25)
    # Always 0 or 1 → zero variance.
    assert dm.sharpness_variance([0.0, 1.0, 0.0, 1.0]) == pytest.approx(0.0)


def test_sharpness_variance_empty():
    assert dm.sharpness_variance([]) is None


def test_sharpness_ece_scatter_requires_min_bucket():
    df = pd.DataFrame({
        "model_key": ["m"] * 5,
        "horizon_minutes": [10] * 5,
        "hour_band": ["morning"] * 5,
        "p_has_ebike": [0.6, 0.7, 0.5, 0.8, 0.9],
        "observed_has_ebike": [1, 1, 0, 1, 1],
    })
    out = dm.sharpness_ece_scatter(df, min_per_bucket=30)
    assert out.empty


# ---------------------------------------------------------------------------
# Brier decomposition
# ---------------------------------------------------------------------------


def test_brier_decomposition_identity_holds():
    df = _synthetic_calibrated(n=10_000)
    decomp = dm.brier_decomposition(df, n_bins=20)
    # BS = REL - RES + UNC (within binning error)
    recovered = decomp["reliability"] - decomp["resolution"] + decomp["uncertainty"]
    assert recovered == pytest.approx(decomp["brier"], abs=0.001)


def test_brier_decomposition_perfect_model_has_zero_components():
    # Perfect predictions: p == y everywhere.
    y = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 1] * 100)
    df = pd.DataFrame({
        "p_has_ebike": y.astype(float),
        "observed_has_ebike": y,
    })
    decomp = dm.brier_decomposition(df, n_bins=10)
    assert decomp["brier"] == pytest.approx(0.0, abs=1e-9)
    assert decomp["reliability"] == pytest.approx(0.0, abs=1e-9)


def test_brier_decomposition_empty():
    decomp = dm.brier_decomposition(pd.DataFrame())
    assert decomp["brier"] is None
    assert decomp["n"] == 0


def test_brier_decomposition_by_model_groups_correctly():
    df1 = _synthetic_calibrated(n=2000, seed=1)
    df1["model_key"] = "a"
    df1["model_label"] = "Model A"
    df2 = _synthetic_calibrated(n=2000, seed=2)
    df2["model_key"] = "b"
    df2["model_label"] = "Model B"
    combined = pd.concat([df1, df2], ignore_index=True)
    out = dm.brier_decomposition_by_model(combined)
    assert set(out["model_key"]) == {"a", "b"}
    assert "brier" in out.columns


# ---------------------------------------------------------------------------
# Randomized PIT (count PMF)
# ---------------------------------------------------------------------------


def test_parse_count_pmf_normalizes():
    pmf = dm._parse_count_pmf({"0": 2.0, "1": 2.0, "2": 4.0, "5_plus": 2.0})
    assert sum(pmf.values()) == pytest.approx(1.0)


def test_parse_count_pmf_handles_json_string():
    raw = json.dumps({"0": 0.5, "1": 0.5})
    pmf = dm._parse_count_pmf(raw)
    assert pmf is not None
    assert pmf["0"] == pytest.approx(0.5)


def test_parse_count_pmf_rejects_garbage():
    assert dm._parse_count_pmf(None) is None
    assert dm._parse_count_pmf("not-json") is None
    assert dm._parse_count_pmf({"99_plus": 1.0}) is None


def test_randomized_pit_returns_value_in_unit_interval():
    rng = np.random.default_rng(0)
    pmf = {"0": 0.2, "1": 0.3, "2": 0.3, "3": 0.1, "4": 0.05, "5_plus": 0.05}
    for k in (0, 1, 2, 5):
        u = dm.randomized_pit(pmf, k, rng)
        assert u is not None
        assert 0.0 <= u <= 1.0


def test_randomized_pit_calibrated_yields_uniform():
    """If we draw observed values from the predicted PMF, PIT should be ~ uniform.

    Averaged over 8 independent seeds to keep the test stable: a single seed
    can land in the tail of the chi-squared null distribution by chance.
    """
    pmf = {"0": 0.2, "1": 0.3, "2": 0.3, "3": 0.1, "4": 0.05, "5_plus": 0.05}
    bins = ["0", "1", "2", "3", "4", "5_plus"]
    probs = np.array([pmf[b] for b in bins])
    n = 10_000
    chi_values = []
    for seed in range(8):
        rng = np.random.default_rng(100 + seed)
        pmf_rng = np.random.default_rng(seed)
        draws = pmf_rng.choice(len(bins), size=n, p=probs)
        observed = [int(b) if b != "5_plus" else 5 for b in [bins[d] for d in draws]]
        pits = np.array([dm.randomized_pit(pmf, o, rng) for o in observed])
        hist, _ = np.histogram(pits, bins=10, range=(0, 1))
        expected = n / 10
        chi_values.append(((hist - expected) ** 2 / expected).sum())
    # Mean chi-squared under the null with df=9 is exactly 9. Anything < 20
    # is decisive evidence that the PIT is uniform — well below the p=0.05
    # critical value of 16.9 for a one-shot test, and the *mean* of 8 runs
    # should sit comfortably below that.
    mean_chi = float(np.mean(chi_values))
    assert mean_chi < 20.0, (
        f"Mean chi-sq across 8 seeds = {mean_chi:.1f} (raw values {chi_values}); "
        "PIT may not be uniform."
    )


def test_count_pit_histogram_shape():
    rng = np.random.default_rng(0)
    pmf = {"0": 0.4, "1": 0.3, "2": 0.2, "3": 0.05, "4": 0.03, "5_plus": 0.02}
    n = 200
    df = pd.DataFrame({
        "model_key": ["m"] * n,
        "model_label": ["m"] * n,
        "p_count_ebikes_json": [pmf] * n,
        "observed_ebikes": rng.choice([0, 1, 2, 3], size=n, p=[0.4, 0.3, 0.2, 0.1]),
    })
    out = dm.count_pit_histogram(df, n_bins=10)
    assert not out.empty
    assert {"bin_mid", "density", "n"}.issubset(out.columns)


# ---------------------------------------------------------------------------
# Time-of-week features + coverage heatmap
# ---------------------------------------------------------------------------


def test_time_of_week_features_adds_columns():
    df = pd.DataFrame({"forecasted_at": pd.to_datetime([
        "2025-01-06T15:00:00Z",  # Monday 9am Chicago
        "2025-01-11T22:00:00Z",  # Saturday 4pm Chicago
    ])})
    out = dm.time_of_week_features(df)
    assert "local_hour" in out.columns
    assert "day_of_week" in out.columns
    assert "weekday_or_weekend" in out.columns
    assert set(out["weekday_or_weekend"]) == {"weekday", "weekend"}


def test_coverage_heatmap_drops_sparse_cells():
    n = 100
    df = pd.DataFrame({
        "model_key": ["m"] * n,
        "model_label": ["m"] * n,
        "forecasted_at": pd.date_range("2025-01-06", periods=n, freq="1h", tz="UTC"),
        "p_has_ebike": np.linspace(0.1, 0.9, n),
        "observed_has_ebike": np.random.default_rng(0).integers(0, 2, size=n),
    })
    out = dm.coverage_heatmap_data(df, min_per_cell=50)
    # Each (day, hour) cell here has only ~4 samples, all should be dropped.
    assert out.empty


# ---------------------------------------------------------------------------
# Skill score
# ---------------------------------------------------------------------------


def test_skill_score_positive_when_model_beats_baseline():
    assert dm.skill_score(0.1, 0.2) == pytest.approx(0.5)


def test_skill_score_handles_nan_and_zero():
    assert dm.skill_score(None, 0.2) is None
    assert dm.skill_score(0.1, None) is None
    assert dm.skill_score(0.1, 0.0) is None


def test_attach_skill_scores_uses_baseline_key():
    rows = [
        {"model_key": "empirical", "brier_score": 0.20},
        {"model_key": "model_a", "brier_score": 0.15},
        {"model_key": "model_b", "brier_score": 0.25},
    ]
    out = dm.attach_skill_scores(rows, baseline_key="empirical")
    by_key = {r["model_key"]: r for r in out}
    assert by_key["model_a"]["skill_score"] == pytest.approx(0.25)
    assert by_key["model_b"]["skill_score"] == pytest.approx(-0.25)
    assert by_key["empirical"]["skill_score"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Dot-grid positions (rider chart)
# ---------------------------------------------------------------------------


def test_dot_grid_positions_filled_count_matches_probability():
    out = dm.dot_grid_positions(0.73, n=100)
    assert int(out["filled"].sum()) == 73


def test_dot_grid_positions_clamps_probability():
    assert int(dm.dot_grid_positions(1.5, n=100)["filled"].sum()) == 100
    assert int(dm.dot_grid_positions(-0.2, n=100)["filled"].sum()) == 0


def test_dot_grid_positions_handles_none_and_nan():
    assert int(dm.dot_grid_positions(None)["filled"].sum()) == 0
    assert int(dm.dot_grid_positions(float("nan"))["filled"].sum()) == 0


def test_dot_grid_positions_shape():
    out = dm.dot_grid_positions(0.5, n=100, cols=10)
    assert len(out) == 100
    assert set(out["x"]) == set(range(10))
    assert set(out["y"]) == set(range(10))
