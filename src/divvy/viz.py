"""Altair chart constructors for the divvy-observer dashboard.

Every function in this module returns an Altair chart (or layered chart)
configured to render correctly at small sizes. Functions accept the exact
DataFrame shape produced by ``dashboard_metrics`` so the dashboard layer
stays a thin glue layer.

See ``CALIBRATION_VIZ_DESIGN.md`` for the design rationale behind each
chart family.
"""

from __future__ import annotations

import math

import altair as alt
import pandas as pd


# Shared palette: ColorBrewer Dark2, friendly to colourblind viewers.
MODEL_PALETTE = "category10"
DIVERGING_PALETTE = "redblue"
SEQUENTIAL_PALETTE = "reds"
CHART_HEIGHT = 220


def _empty_placeholder(title: str, message: str = "No resolved outcomes yet.") -> alt.Chart:
    return (
        alt.Chart(pd.DataFrame({"x": [0], "y": [0], "msg": [message]}))
        .mark_text(align="center", baseline="middle", fontSize=12, color="#888")
        .encode(x=alt.X("x:Q", axis=None), y=alt.Y("y:Q", axis=None), text="msg:N")
        .properties(title=title, height=140, width="container")
    )


# ---------------------------------------------------------------------------
# 1. Rider-facing icon array (frequency dot grid)
# ---------------------------------------------------------------------------


def dot_grid_chart(
    positions: pd.DataFrame,
    *,
    probability: float | None,
    title: str | None = None,
    filled_color: str = "#1f77b4",
    empty_color: str = "#e8e8e8",
    dot_size: int = 120,
) -> alt.Chart:
    """Render a frequency dot grid from ``dashboard_metrics.dot_grid_positions``.

    ``positions`` is a DataFrame with columns ``x``, ``y``, ``filled``. The
    title and the probability subtitle are rendered as Altair properties so
    the same chart can sit inside a streamlit column without external labels.
    """
    if positions.empty:
        return _empty_placeholder(title or "—")
    label = "—" if probability is None else f"{probability * 100:.0f}%"
    title_struct = alt.TitleParams(
        text=title or " ",
        subtitle=f"{label} chance of finding a bike",
        anchor="start",
        subtitleColor="#444",
        subtitleFontSize=12,
    )
    chart = (
        alt.Chart(positions)
        .mark_point(filled=True, size=dot_size, stroke=None)
        .encode(
            x=alt.X("x:O", axis=None, scale=alt.Scale(padding=0.4)),
            y=alt.Y("y:O", axis=None, scale=alt.Scale(padding=0.4)),
            color=alt.Color(
                "filled:N",
                scale=alt.Scale(domain=[True, False], range=[filled_color, empty_color]),
                legend=None,
            ),
            tooltip=[alt.Tooltip("idx:Q", title="dot #"), "filled:N"],
        )
        .properties(title=title_struct, width=200, height=200)
    )
    return chart


# ---------------------------------------------------------------------------
# 2. Reliability diagram
# ---------------------------------------------------------------------------


def reliability_diagram_chart(
    df: pd.DataFrame,
    *,
    facet_col: str | None = None,
    title: str = "Reliability diagram",
) -> alt.Chart:
    """Render a binary reliability diagram with Wilson 95% intervals.

    ``df`` is the shape returned by ``dashboard_metrics.reliability_curve``.
    Plots predicted_mean (x) vs observed_rate (y) per model. A diagonal
    reference line and equal axis ranges keep "calibrated = on the line"
    visually unambiguous.

    If ``facet_col`` is supplied (e.g. ``"horizon_minutes"``) the chart
    facets into small multiples.
    """
    if df.empty:
        return _empty_placeholder(title)

    base = alt.Chart(df)

    # Diagonal reference line — rendered first so points sit on top.
    diagonal_data = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]})
    diagonal = (
        alt.Chart(diagonal_data)
        .mark_line(strokeDash=[6, 4], color="#888")
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("y:Q", scale=alt.Scale(domain=[0, 1])),
        )
    )

    error_bars = base.mark_rule(strokeWidth=2, opacity=0.6).encode(
        x=alt.X("predicted_mean:Q", scale=alt.Scale(domain=[0, 1]), title="Predicted P(has bike)"),
        y=alt.Y("observed_ci_low:Q", scale=alt.Scale(domain=[0, 1]), title="Observed rate"),
        y2="observed_ci_high:Q",
        color=alt.Color("model_label:N", scale=alt.Scale(scheme=MODEL_PALETTE), legend=alt.Legend(title="Model")),
    )

    points = base.mark_circle(opacity=0.9).encode(
        x=alt.X("predicted_mean:Q", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("observed_rate:Q", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("model_label:N", scale=alt.Scale(scheme=MODEL_PALETTE), legend=alt.Legend(title="Model")),
        size=alt.Size("n:Q", scale=alt.Scale(range=[40, 350]), legend=alt.Legend(title="N")),
        tooltip=[
            alt.Tooltip("model_label:N", title="Model"),
            alt.Tooltip("predicted_mean:Q", format=".2f", title="Predicted"),
            alt.Tooltip("observed_rate:Q", format=".2f", title="Observed"),
            alt.Tooltip("observed_ci_low:Q", format=".2f", title="CI lo"),
            alt.Tooltip("observed_ci_high:Q", format=".2f", title="CI hi"),
            alt.Tooltip("n:Q", title="N", format=","),
        ],
    )

    layered = alt.layer(diagonal, error_bars, points)
    if facet_col is not None and facet_col in df.columns:
        return layered.facet(
            facet=alt.Facet(f"{facet_col}:O", title=facet_col.replace("_", " ").title()),
            columns=4,
            data=df,
        ).properties(title=title)
    return layered.properties(title=title, height=CHART_HEIGHT, width="container")


# ---------------------------------------------------------------------------
# 3. Score-distribution histogram (binary discrimination)
# ---------------------------------------------------------------------------


def score_distribution_chart(
    df: pd.DataFrame,
    *,
    facet_col: str | None = "model_label",
    title: str = "Predicted-probability distribution by outcome",
) -> alt.Chart:
    """Twin density-normalized histograms split by realised outcome."""
    if df.empty:
        return _empty_placeholder(title)
    base = alt.Chart(df).mark_area(opacity=0.55, interpolate="step-after").encode(
        x=alt.X("bin_mid:Q", scale=alt.Scale(domain=[0, 1]), title="Predicted P(has bike)"),
        y=alt.Y("density:Q", title="Relative frequency"),
        color=alt.Color(
            "outcome:N",
            scale=alt.Scale(
                domain=["y=1 (had bike)", "y=0 (no bike)"],
                range=["#1f77b4", "#d62728"],
            ),
            legend=alt.Legend(title="Outcome"),
        ),
        tooltip=[
            alt.Tooltip("outcome:N", title="Outcome"),
            alt.Tooltip("bin_mid:Q", format=".2f", title="P bin"),
            alt.Tooltip("density:Q", format=".3f", title="Density"),
            alt.Tooltip("n_outcome:Q", format=",", title="N (this outcome)"),
        ],
    )
    if facet_col and facet_col in df.columns:
        return base.facet(
            facet=alt.Facet(f"{facet_col}:N", title=facet_col.replace("_", " ").title()),
            columns=3,
            data=df,
        ).properties(title=title)
    return base.properties(title=title, height=CHART_HEIGHT, width="container")


# ---------------------------------------------------------------------------
# 4. Randomized PIT histogram for the count PMF
# ---------------------------------------------------------------------------


def count_pit_histogram_chart(
    df: pd.DataFrame,
    *,
    facet_col: str | None = "model_label",
    title: str = "Randomized PIT (count PMF)",
) -> alt.Chart:
    """Render the count-PIT histogram. Reference line at density=1 (flat = calibrated)."""
    if df.empty:
        return _empty_placeholder(title)

    reference = (
        alt.Chart(pd.DataFrame({"y": [1.0]}))
        .mark_rule(strokeDash=[4, 4], color="#888")
        .encode(y="y:Q")
    )
    bars = alt.Chart(df).mark_bar(opacity=0.8).encode(
        x=alt.X("bin_mid:Q", bin=alt.Bin(maxbins=10), title="PIT value"),
        y=alt.Y("density:Q", title="Density"),
        color=alt.Color("model_label:N", scale=alt.Scale(scheme=MODEL_PALETTE), legend=None),
        tooltip=[
            alt.Tooltip("model_label:N", title="Model"),
            alt.Tooltip("bin_mid:Q", format=".2f", title="PIT bin"),
            alt.Tooltip("density:Q", format=".3f", title="Density"),
            alt.Tooltip("n:Q", title="N", format=","),
        ],
    )
    layered = bars + reference
    if facet_col and facet_col in df.columns:
        return layered.facet(
            facet=alt.Facet(f"{facet_col}:N", title="Model"),
            columns=3,
            data=df,
        ).properties(title=title)
    return layered.properties(title=title, height=CHART_HEIGHT, width="container")


# ---------------------------------------------------------------------------
# 5. Sharpness ↔ ECE scatter
# ---------------------------------------------------------------------------


def sharpness_ece_chart(
    df: pd.DataFrame,
    *,
    title: str = "Sharpness vs calibration",
) -> alt.Chart:
    """Per-bucket scatter, x = sharpness (lower is sharper), y = ECE.

    Target zone is bottom-left (sharp AND calibrated). Top-left is the
    worst quadrant: confident-but-wrong.
    """
    if df.empty:
        return _empty_placeholder(title)
    chart = alt.Chart(df).mark_circle(opacity=0.7).encode(
        x=alt.X(
            "sharpness:Q",
            title="Sharpness (mean p(1−p); lower = sharper)",
            scale=alt.Scale(domain=[0, 0.25]),
        ),
        y=alt.Y(
            "ece:Q",
            title="ECE (calibration error)",
            scale=alt.Scale(domain=[0, 0.30]),
        ),
        color=alt.Color("model_key:N", scale=alt.Scale(scheme=MODEL_PALETTE), legend=alt.Legend(title="Model")),
        size=alt.Size("n:Q", scale=alt.Scale(range=[30, 400]), legend=alt.Legend(title="N")),
        shape=alt.Shape("horizon_minutes:O", legend=alt.Legend(title="Horizon (min)")),
        tooltip=[
            alt.Tooltip("model_key:N", title="Model"),
            alt.Tooltip("horizon_minutes:Q", title="Horizon"),
            alt.Tooltip("sharpness:Q", format=".3f"),
            alt.Tooltip("ece:Q", format=".3f"),
            alt.Tooltip("mean_prediction:Q", format=".2f", title="Mean pred"),
            alt.Tooltip("observed_rate:Q", format=".2f", title="Observed"),
            alt.Tooltip("n:Q", title="N", format=","),
        ],
    )
    return chart.properties(title=title, height=320, width="container")


# ---------------------------------------------------------------------------
# 6. Coverage / ECE heatmap by hour × day-of-week
# ---------------------------------------------------------------------------


_DOW_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def coverage_heatmap_chart(
    df: pd.DataFrame,
    *,
    metric: str = "calibration_gap",
    title: str | None = None,
) -> alt.Chart:
    """Per-model heatmap of calibration error across hour × day-of-week.

    ``metric`` can be:
      - ``"calibration_gap"`` (default): mean_prediction − observed_rate. Diverging
        palette — red = overconfident, blue = underconfident, white = on target.
      - ``"ece"``: sequential red palette — darker = worse.
      - ``"observed_rate"``: sequential — useful as a sanity-check overlay.
    """
    if df.empty:
        return _empty_placeholder(title or f"Coverage heatmap — {metric}")
    if metric == "calibration_gap":
        color = alt.Color(
            f"{metric}:Q",
            title="Pred − observed",
            scale=alt.Scale(scheme=DIVERGING_PALETTE, domainMid=0),
        )
    else:
        color = alt.Color(
            f"{metric}:Q",
            title=metric.replace("_", " ").title(),
            scale=alt.Scale(scheme=SEQUENTIAL_PALETTE),
        )

    chart = alt.Chart(df).mark_rect().encode(
        x=alt.X("local_hour:O", title="Hour of day", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("day_of_week:N", title="Day", sort=_DOW_ORDER),
        color=color,
        tooltip=[
            alt.Tooltip("model_label:N", title="Model"),
            alt.Tooltip("day_of_week:N", title="Day"),
            alt.Tooltip("local_hour:Q", title="Hour"),
            alt.Tooltip("n:Q", title="N", format=","),
            alt.Tooltip("mean_prediction:Q", format=".2f", title="Mean pred"),
            alt.Tooltip("observed_rate:Q", format=".2f", title="Observed"),
            alt.Tooltip("ece:Q", format=".3f", title="ECE"),
            alt.Tooltip("calibration_gap:Q", format=".3f", title="Gap"),
        ],
    )
    if "model_label" in df.columns and df["model_label"].nunique() > 1:
        return chart.facet(
            facet=alt.Facet("model_label:N", title="Model"),
            columns=2,
            data=df,
        ).properties(title=title or f"Calibration by time of week — {metric}")
    return chart.properties(title=title or f"Calibration by time of week — {metric}", height=220, width="container")


# ---------------------------------------------------------------------------
# 7. Brier decomposition bar chart
# ---------------------------------------------------------------------------


def brier_decomposition_chart(
    df: pd.DataFrame,
    *,
    title: str = "Brier score decomposition",
) -> alt.Chart:
    """Per-model stacked bars showing reliability − resolution + uncertainty.

    Reliability is "calibration error" (lower is better). Resolution is
    "discrimination" (higher is better). Uncertainty is fixed by the data.
    """
    if df.empty:
        return _empty_placeholder(title)
    melted = df.melt(
        id_vars=[c for c in ("model_key", "model_label") if c in df.columns],
        value_vars=[c for c in ("reliability", "resolution", "uncertainty") if c in df.columns],
        var_name="component",
        value_name="value",
    )
    if melted.empty:
        return _empty_placeholder(title)
    component_order = ["uncertainty", "reliability", "resolution"]
    melted["component"] = pd.Categorical(melted["component"], categories=component_order, ordered=True)
    color_map = {
        "uncertainty": "#bdbdbd",
        "reliability": "#d62728",
        "resolution": "#2ca02c",
    }
    chart = alt.Chart(melted).mark_bar().encode(
        y=alt.Y("model_label:N", title="Model", sort="-x"),
        x=alt.X("value:Q", title="Component contribution"),
        color=alt.Color(
            "component:N",
            scale=alt.Scale(domain=list(color_map.keys()), range=list(color_map.values())),
            legend=alt.Legend(title="Component"),
        ),
        tooltip=[
            alt.Tooltip("model_label:N", title="Model"),
            alt.Tooltip("component:N", title="Component"),
            alt.Tooltip("value:Q", format=".4f"),
        ],
    )
    return chart.properties(title=title, height=alt.Step(28), width="container")


# ---------------------------------------------------------------------------
# 8. Rolling metric trend
# ---------------------------------------------------------------------------


def metric_trend_chart(
    df: pd.DataFrame,
    *,
    value_col: str = "brier_score",
    title: str | None = None,
) -> alt.Chart:
    """Per-model rolling metric trend line.

    Expects columns ``computed_at``, ``model_key``, ``value_col``.
    """
    if df.empty or value_col not in df:
        return _empty_placeholder(title or f"{value_col} over time")
    chart = alt.Chart(df).mark_line(point=alt.OverlayMarkDef(size=30)).encode(
        x=alt.X("computed_at:T", title="Computed at"),
        y=alt.Y(f"{value_col}:Q", title=value_col.replace("_", " ").title()),
        color=alt.Color("model_key:N", scale=alt.Scale(scheme=MODEL_PALETTE), legend=alt.Legend(title="Model")),
        tooltip=[
            alt.Tooltip("model_key:N", title="Model"),
            alt.Tooltip("computed_at:T", title="When"),
            alt.Tooltip(f"{value_col}:Q", format=".4f"),
            alt.Tooltip("n:Q", title="N", format=","),
        ],
    )
    return chart.properties(title=title or f"{value_col} over time", height=CHART_HEIGHT, width="container")


# ---------------------------------------------------------------------------
# 9. Per-horizon forecast curve (rider-facing)
# ---------------------------------------------------------------------------


def horizon_curve_chart(
    df: pd.DataFrame,
    *,
    title: str = "Probability of finding a bike, by horizon",
    confidence_band: bool = False,
) -> alt.Chart:
    """Per-horizon P(has bike) curve for a single station.

    Expects columns ``horizon_minutes``, ``p_has_ebike``, plus optionally
    ``p_low``, ``p_high`` for a confidence band when ``confidence_band=True``.
    If a ``model_label`` column exists, draws one line per model.
    """
    if df.empty:
        return _empty_placeholder(title)
    color_kwargs = {}
    if "model_label" in df.columns and df["model_label"].nunique() > 1:
        color_kwargs["color"] = alt.Color("model_label:N", scale=alt.Scale(scheme=MODEL_PALETTE), legend=alt.Legend(title="Model"))

    line = alt.Chart(df).mark_line(point=True, strokeWidth=2).encode(
        x=alt.X("horizon_minutes:Q", title="Minutes from now"),
        y=alt.Y("p_has_ebike:Q", title="P(has bike)", scale=alt.Scale(domain=[0, 1])),
        tooltip=[
            alt.Tooltip("horizon_minutes:Q", title="Horizon"),
            alt.Tooltip("p_has_ebike:Q", format=".2f", title="P(has bike)"),
        ] + ([alt.Tooltip("model_label:N", title="Model")] if "model_label" in df.columns else []),
        **color_kwargs,
    )

    if confidence_band and "p_low" in df.columns and "p_high" in df.columns:
        band = alt.Chart(df).mark_area(opacity=0.18).encode(
            x="horizon_minutes:Q",
            y=alt.Y("p_low:Q", scale=alt.Scale(domain=[0, 1])),
            y2="p_high:Q",
            **color_kwargs,
        )
        return (band + line).properties(title=title, height=CHART_HEIGHT, width="container")
    return line.properties(title=title, height=CHART_HEIGHT, width="container")


# ---------------------------------------------------------------------------
# 10. Top-k recommendation hit rate
# ---------------------------------------------------------------------------


def topk_hitrate_chart(
    df: pd.DataFrame,
    *,
    k_values: tuple[int, ...] = (1, 3, 5),
    title: str = "Top-k recommendation hit rate",
) -> alt.Chart:
    """Per-model top-k hit rate. ``df`` has columns
    ``model_label``, ``top1_hit_rate``, ``top3_hit_rate``, ``top5_hit_rate``,
    ``n_requests``.
    """
    if df.empty:
        return _empty_placeholder(title)
    melt_cols = [f"top{k}_hit_rate" for k in k_values if f"top{k}_hit_rate" in df.columns]
    if not melt_cols:
        return _empty_placeholder(title)
    melted = df.melt(
        id_vars=[c for c in ("model_label", "model_key", "n_requests") if c in df.columns],
        value_vars=melt_cols,
        var_name="k",
        value_name="hit_rate",
    )
    melted["k"] = melted["k"].str.extract(r"top(\d+)_").astype(int)

    chart = alt.Chart(melted).mark_bar().encode(
        x=alt.X("k:O", title="Top-k"),
        y=alt.Y("hit_rate:Q", title="Hit rate", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("model_label:N", scale=alt.Scale(scheme=MODEL_PALETTE), legend=alt.Legend(title="Model")),
        column=alt.Column("model_label:N", title=None, header=alt.Header(labelAngle=-30, labelAlign="right")),
        tooltip=[
            alt.Tooltip("model_label:N", title="Model"),
            alt.Tooltip("k:O", title="k"),
            alt.Tooltip("hit_rate:Q", format=".1%", title="Hit rate"),
            alt.Tooltip("n_requests:Q", title="Requests", format=","),
        ],
    )
    return chart.properties(title=title)


# ---------------------------------------------------------------------------
# 11. Regret distribution
# ---------------------------------------------------------------------------


def regret_distribution_chart(
    df: pd.DataFrame,
    *,
    regret_col: str = "distance_adjusted_regret",
    title: str = "Distance-adjusted regret distribution",
) -> alt.Chart:
    """Per-model regret distribution as a violin-style density."""
    if df.empty or regret_col not in df:
        return _empty_placeholder(title)
    work = df.dropna(subset=[regret_col]).copy()
    if work.empty:
        return _empty_placeholder(title)
    chart = alt.Chart(work).transform_density(
        regret_col,
        groupby=["model_label"],
        as_=[regret_col, "density"],
    ).mark_area(opacity=0.55).encode(
        x=alt.X(f"{regret_col}:Q", title="Regret (lower is better)"),
        y=alt.Y("density:Q", title="Density"),
        color=alt.Color("model_label:N", scale=alt.Scale(scheme=MODEL_PALETTE), legend=alt.Legend(title="Model")),
    )
    return chart.properties(title=title, height=CHART_HEIGHT, width="container")


# ---------------------------------------------------------------------------
# 12. Count PMF bar (per-station explainer)
# ---------------------------------------------------------------------------


def count_pmf_chart(
    pmf: dict[str, float] | None,
    *,
    title: str = "P(number of bikes)",
) -> alt.Chart:
    """Bar chart of the ebike count PMF for a single station/horizon.

    Renders bins 0, 1, 2, 3, 4, ≥5 with their predicted probabilities.
    """
    if not pmf:
        return _empty_placeholder(title, message="No PMF available for this station.")
    label_map = {"0": "0", "1": "1", "2": "2", "3": "3", "4": "4", "5_plus": "≥5"}
    rows = []
    for key, label in label_map.items():
        if key in pmf:
            rows.append({"bucket": label, "prob": float(pmf[key]), "order": list(label_map).index(key)})
    if not rows:
        return _empty_placeholder(title)
    df = pd.DataFrame(rows)
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X("bucket:N", sort=[label_map[k] for k in label_map], title="ebikes"),
        y=alt.Y("prob:Q", title="Probability", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("prob:Q", scale=alt.Scale(scheme="blues"), legend=None),
        tooltip=[alt.Tooltip("bucket:N"), alt.Tooltip("prob:Q", format=".2f")],
    )
    return chart.properties(title=title, height=160, width="container")


# ---------------------------------------------------------------------------
# 13. Station-comparison heatmap (for the Find-a-Bike grid)
# ---------------------------------------------------------------------------


def station_horizon_heatmap(
    df: pd.DataFrame,
    *,
    title: str = "P(has bike) by station × horizon",
) -> alt.Chart:
    """Compact heatmap: rows = candidate stations, columns = horizons, color
    = P(has bike). Used in the Find-a-Bike screen to show a recommendation
    cohort at a glance.

    Expects columns ``station_label``, ``horizon_minutes``, ``p_has_ebike``,
    ``distance_km`` (for tooltip).
    """
    if df.empty:
        return _empty_placeholder(title)
    chart = alt.Chart(df).mark_rect().encode(
        x=alt.X("horizon_minutes:O", title="Horizon (min)", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("station_label:N", title="Station", sort=alt.SortField("distance_km", order="ascending")),
        color=alt.Color(
            "p_has_ebike:Q",
            scale=alt.Scale(scheme="blues", domain=[0, 1]),
            title="P(bike)",
        ),
        tooltip=[
            alt.Tooltip("station_label:N", title="Station"),
            alt.Tooltip("horizon_minutes:Q", title="Horizon"),
            alt.Tooltip("p_has_ebike:Q", format=".2f", title="P(bike)"),
            alt.Tooltip("distance_km:Q", format=".2f", title="Distance (km)"),
        ],
    )
    return chart.properties(title=title, height=alt.Step(22), width="container")


# ---------------------------------------------------------------------------
# Utility: convert leaderboard dicts into a DataFrame the table renderer wants.
# ---------------------------------------------------------------------------


def leaderboard_frame(leaderboard: list[dict]) -> pd.DataFrame:
    """Normalize the metric-row dicts produced by ``model_eval`` into a
    consistent column order for ``st.dataframe``."""
    if not leaderboard:
        return pd.DataFrame()
    column_order = [
        "rank", "model_label", "model_key", "n",
        "brier_score", "log_loss", "ece", "rank_loss",
        "decision_rank_loss", "skill_score",
        "observed_rate", "mean_prediction",
        "recommended_hit_rate", "distance_adjusted_regret",
        "count_log_loss", "crps",
        "mean_expected_ebikes", "mean_observed_ebikes",
    ]
    df = pd.DataFrame(leaderboard)
    ordered = [c for c in column_order if c in df.columns]
    extras = [c for c in df.columns if c not in column_order]
    return df[ordered + extras]


def format_probability(value: float | None) -> str:
    if value is None:
        return "—"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "—"
    if math.isnan(v):
        return "—"
    return f"{v * 100:.0f}%"
