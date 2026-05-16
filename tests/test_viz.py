"""Tests for the chart constructors in ``divvy.viz``.

Charts are validated by calling ``.to_dict()`` — Altair raises a
SchemaValidationError if the spec is malformed. Empty inputs must produce
the placeholder chart (no exception, no rendered content).
"""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import pytest

from divvy import dashboard_metrics as dm
from divvy import viz


# ---------------------------------------------------------------------------
# Common fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_resolved():
    rng = np.random.default_rng(0)
    n = 800
    p = rng.uniform(0, 1, size=n)
    y = (rng.uniform(0, 1, size=n) < p).astype(int)
    horizons = rng.choice([5, 10, 30], size=n)
    return pd.DataFrame({
        "model_key": rng.choice(["m1", "m2"], size=n),
        "model_label": rng.choice(["Model A", "Model B"], size=n),
        "horizon_minutes": horizons,
        "p_has_ebike": p,
        "observed_has_ebike": y,
        "forecasted_at": pd.date_range("2025-01-06", periods=n, freq="15min", tz="UTC"),
        "station_id": ["s1"] * (n // 2) + ["s2"] * (n - n // 2),
        "station_name": ["Station A"] * (n // 2) + ["Station B"] * (n - n // 2),
        "current_ebikes": rng.integers(0, 6, size=n),
    })


# ---------------------------------------------------------------------------
# Dot grid (rider-facing icon array)
# ---------------------------------------------------------------------------


def test_dot_grid_chart_renders_valid_spec():
    positions = dm.dot_grid_positions(0.65, n=100)
    chart = viz.dot_grid_chart(positions, probability=0.65, title="Test Station")
    spec = chart.to_dict()
    assert spec.get("mark") == {"filled": True, "size": 120, "stroke": None, "type": "point"}


def test_dot_grid_chart_handles_empty():
    chart = viz.dot_grid_chart(pd.DataFrame(), probability=None, title="—")
    # Empty placeholder must still serialize.
    spec = chart.to_dict()
    assert isinstance(spec, dict)


def test_dot_grid_chart_handles_zero_probability():
    positions = dm.dot_grid_positions(0.0, n=100)
    chart = viz.dot_grid_chart(positions, probability=0.0, title="Empty")
    assert chart.to_dict() is not None


# ---------------------------------------------------------------------------
# Reliability diagram
# ---------------------------------------------------------------------------


def test_reliability_diagram_chart_renders(synthetic_resolved):
    curve = dm.reliability_curve(synthetic_resolved, min_per_bin=5)
    chart = viz.reliability_diagram_chart(curve)
    assert chart.to_dict() is not None


def test_reliability_diagram_chart_facets_when_requested(synthetic_resolved):
    curve = dm.reliability_curve(
        synthetic_resolved,
        group_cols=("model_key", "model_label", "horizon_minutes"),
        min_per_bin=5,
    )
    chart = viz.reliability_diagram_chart(curve, facet_col="horizon_minutes")
    spec = chart.to_dict()
    assert "facet" in spec or any("facet" in str(v) for v in spec.values())


def test_reliability_diagram_empty():
    chart = viz.reliability_diagram_chart(pd.DataFrame())
    assert chart.to_dict() is not None


# ---------------------------------------------------------------------------
# Score distribution
# ---------------------------------------------------------------------------


def test_score_distribution_chart_renders(synthetic_resolved):
    dist = dm.score_distribution(synthetic_resolved)
    chart = viz.score_distribution_chart(dist, facet_col=None)
    assert chart.to_dict() is not None


def test_score_distribution_chart_facets(synthetic_resolved):
    dist = dm.score_distribution(synthetic_resolved)
    chart = viz.score_distribution_chart(dist, facet_col="model_label")
    assert chart.to_dict() is not None


# ---------------------------------------------------------------------------
# Sharpness-ECE scatter
# ---------------------------------------------------------------------------


def test_sharpness_ece_chart_renders():
    df = pd.DataFrame({
        "model_key": ["m1", "m1", "m2", "m2"],
        "horizon_minutes": [10, 30, 10, 30],
        "hour_band": ["morning", "morning", "evening", "evening"],
        "sharpness": [0.1, 0.15, 0.2, 0.12],
        "ece": [0.05, 0.10, 0.03, 0.04],
        "n": [400, 300, 200, 150],
        "mean_prediction": [0.5] * 4,
        "observed_rate": [0.5] * 4,
    })
    chart = viz.sharpness_ece_chart(df)
    assert chart.to_dict() is not None


def test_sharpness_ece_chart_empty():
    chart = viz.sharpness_ece_chart(pd.DataFrame())
    assert chart.to_dict() is not None


# ---------------------------------------------------------------------------
# Coverage heatmap
# ---------------------------------------------------------------------------


def test_coverage_heatmap_chart_renders(synthetic_resolved):
    data = dm.coverage_heatmap_data(synthetic_resolved, min_per_cell=2)
    chart = viz.coverage_heatmap_chart(data, metric="calibration_gap")
    assert chart.to_dict() is not None


def test_coverage_heatmap_supports_different_metrics(synthetic_resolved):
    data = dm.coverage_heatmap_data(synthetic_resolved, min_per_cell=2)
    if data.empty:
        pytest.skip("not enough samples per cell")
    for metric in ("calibration_gap", "ece", "observed_rate"):
        chart = viz.coverage_heatmap_chart(data, metric=metric)
        assert chart.to_dict() is not None


# ---------------------------------------------------------------------------
# Brier decomposition chart
# ---------------------------------------------------------------------------


def test_brier_decomposition_chart_renders(synthetic_resolved):
    decomp = dm.brier_decomposition_by_model(synthetic_resolved)
    chart = viz.brier_decomposition_chart(decomp)
    assert chart.to_dict() is not None


def test_brier_decomposition_chart_empty():
    chart = viz.brier_decomposition_chart(pd.DataFrame())
    assert chart.to_dict() is not None


# ---------------------------------------------------------------------------
# Metric trend
# ---------------------------------------------------------------------------


def test_metric_trend_chart_renders():
    df = pd.DataFrame({
        "computed_at": pd.date_range("2025-05-01", periods=10, freq="D"),
        "model_key": ["m1"] * 10,
        "brier_score": np.linspace(0.20, 0.15, 10),
        "n": [1000] * 10,
    })
    chart = viz.metric_trend_chart(df, value_col="brier_score")
    assert chart.to_dict() is not None


# ---------------------------------------------------------------------------
# Horizon curve
# ---------------------------------------------------------------------------


def test_horizon_curve_chart_renders():
    df = pd.DataFrame({
        "horizon_minutes": [5, 10, 15, 20, 30, 45, 60, 90],
        "p_has_ebike": [0.85, 0.78, 0.72, 0.68, 0.60, 0.55, 0.50, 0.42],
    })
    chart = viz.horizon_curve_chart(df)
    assert chart.to_dict() is not None


def test_horizon_curve_chart_with_band():
    df = pd.DataFrame({
        "horizon_minutes": [5, 10, 15, 20],
        "p_has_ebike": [0.8, 0.75, 0.70, 0.65],
        "p_low": [0.7, 0.65, 0.6, 0.55],
        "p_high": [0.9, 0.85, 0.80, 0.75],
    })
    chart = viz.horizon_curve_chart(df, confidence_band=True)
    assert chart.to_dict() is not None


def test_horizon_curve_multi_model():
    df = pd.DataFrame({
        "horizon_minutes": [5, 10, 15] * 2,
        "p_has_ebike": [0.85, 0.78, 0.72, 0.80, 0.74, 0.68],
        "model_label": ["M1"] * 3 + ["M2"] * 3,
    })
    chart = viz.horizon_curve_chart(df)
    assert chart.to_dict() is not None


# ---------------------------------------------------------------------------
# Top-k hit rate
# ---------------------------------------------------------------------------


def test_topk_hitrate_chart_renders():
    df = pd.DataFrame({
        "model_label": ["A", "B"],
        "model_key": ["a", "b"],
        "n_requests": [100, 100],
        "top1_hit_rate": [0.4, 0.35],
        "top3_hit_rate": [0.7, 0.65],
        "top5_hit_rate": [0.85, 0.78],
    })
    chart = viz.topk_hitrate_chart(df, k_values=(1, 3, 5))
    assert chart.to_dict() is not None


def test_topk_hitrate_empty():
    chart = viz.topk_hitrate_chart(pd.DataFrame())
    assert chart.to_dict() is not None


# ---------------------------------------------------------------------------
# Count PMF
# ---------------------------------------------------------------------------


def test_count_pmf_chart_renders():
    pmf = {"0": 0.1, "1": 0.2, "2": 0.3, "3": 0.2, "4": 0.1, "5_plus": 0.1}
    chart = viz.count_pmf_chart(pmf)
    assert chart.to_dict() is not None


def test_count_pmf_chart_empty():
    chart = viz.count_pmf_chart(None)
    assert chart.to_dict() is not None


# ---------------------------------------------------------------------------
# Station-horizon heatmap
# ---------------------------------------------------------------------------


def test_station_horizon_heatmap_renders():
    df = pd.DataFrame({
        "station_label": ["A", "A", "B", "B"],
        "horizon_minutes": [10, 30, 10, 30],
        "p_has_ebike": [0.8, 0.6, 0.75, 0.55],
        "distance_km": [0.3, 0.3, 0.5, 0.5],
    })
    chart = viz.station_horizon_heatmap(df)
    assert chart.to_dict() is not None


# ---------------------------------------------------------------------------
# Leaderboard frame ordering
# ---------------------------------------------------------------------------


def test_leaderboard_frame_orders_columns():
    rows = [{
        "model_label": "A", "model_key": "a", "n": 100,
        "brier_score": 0.1, "log_loss": 0.5, "ece": 0.03,
        "rank_loss": 0.12, "decision_rank_loss": 0.14, "skill_score": 0.2,
        "observed_rate": 0.5, "mean_prediction": 0.55,
        "extra_field": 99,
    }]
    df = viz.leaderboard_frame(rows)
    cols = df.columns.tolist()
    # rank not present in inputs so it shouldn't appear.
    assert "model_label" in cols
    assert cols.index("model_label") < cols.index("brier_score")
    assert cols.index("brier_score") < cols.index("log_loss")
    # Extra fields appear at the end.
    assert cols[-1] == "extra_field"


def test_leaderboard_frame_empty_input():
    assert viz.leaderboard_frame([]).empty


# ---------------------------------------------------------------------------
# Probability formatter
# ---------------------------------------------------------------------------


def test_format_probability_handles_edge_cases():
    assert viz.format_probability(None) == "—"
    assert viz.format_probability(float("nan")) == "—"
    assert viz.format_probability(0.55) == "55%"
    assert viz.format_probability(1.0) == "100%"


# ---------------------------------------------------------------------------
# Count PIT histogram chart
# ---------------------------------------------------------------------------


def test_count_pit_histogram_chart_renders():
    df = pd.DataFrame({
        "model_label": ["A"] * 10 + ["B"] * 10,
        "model_key": ["a"] * 10 + ["b"] * 10,
        "bin_mid": list(np.linspace(0.05, 0.95, 10)) * 2,
        "density": [1.0] * 20,
        "n": [200] * 20,
    })
    chart = viz.count_pit_histogram_chart(df, facet_col="model_label")
    assert chart.to_dict() is not None


# ---------------------------------------------------------------------------
# Regret distribution
# ---------------------------------------------------------------------------


def test_regret_distribution_chart_renders():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "model_label": rng.choice(["A", "B"], size=200),
        "distance_adjusted_regret": rng.exponential(0.2, size=200),
    })
    chart = viz.regret_distribution_chart(df)
    assert chart.to_dict() is not None


def test_regret_distribution_chart_handles_empty():
    chart = viz.regret_distribution_chart(pd.DataFrame())
    assert chart.to_dict() is not None
