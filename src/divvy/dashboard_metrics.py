"""Calibration math for the dashboard's diagnostic charts.

This module is the math counterpart to ``viz.py``: it returns pandas
DataFrames in the exact shape the chart constructors expect, with the
binning, confidence intervals, decompositions, and randomized PIT for
the count PMF computed once and reused.

Design choices and citations live in ``CALIBRATION_VIZ_DESIGN.md``.
Nothing here writes to the DB; callers pull resolved-forecast rows from
``model_eval._joined_forecasts`` (or equivalent) and pass them in.
"""

from __future__ import annotations

import json
import math
from typing import Iterable

import duckdb
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Binary calibration: reliability curve with Wilson confidence intervals
# ---------------------------------------------------------------------------


def _wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Two-sided Wilson 95% CI for a binomial proportion."""
    if n <= 0:
        return (0.0, 1.0)
    phat = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (phat + z2 / (2 * n)) / denom
    half = (z * math.sqrt(phat * (1 - phat) / n + z2 / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def reliability_curve(
    df: pd.DataFrame,
    *,
    prob_col: str = "p_has_ebike",
    outcome_col: str = "observed_has_ebike",
    group_cols: Iterable[str] = ("model_key", "model_label"),
    n_bins: int = 10,
    min_per_bin: int = 5,
) -> pd.DataFrame:
    """Bin predictions and return per-bin observed positive rate + Wilson CI.

    Columns returned:
      ``model_key``, ``model_label`` (or whatever group_cols supplies),
      ``bin_mid``, ``predicted_mean``, ``observed_rate``,
      ``observed_ci_low``, ``observed_ci_high``, ``n``.

    Empty input → empty DataFrame. Bins with fewer than ``min_per_bin``
    samples are dropped so the chart doesn't waste ink on noise.
    """
    if df.empty or prob_col not in df or outcome_col not in df:
        return pd.DataFrame(
            columns=list(group_cols) + [
                "bin_mid", "predicted_mean", "observed_rate",
                "observed_ci_low", "observed_ci_high", "n",
            ]
        )
    work = df[[*group_cols, prob_col, outcome_col]].copy()
    work[prob_col] = pd.to_numeric(work[prob_col], errors="coerce").clip(0.0, 1.0)
    work[outcome_col] = pd.to_numeric(work[outcome_col], errors="coerce")
    work = work.dropna(subset=[prob_col, outcome_col])
    if work.empty:
        return pd.DataFrame(
            columns=list(group_cols) + [
                "bin_mid", "predicted_mean", "observed_rate",
                "observed_ci_low", "observed_ci_high", "n",
            ]
        )

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    mids = (edges[:-1] + edges[1:]) / 2.0
    work["bin_idx"] = np.clip(
        np.digitize(work[prob_col].to_numpy(), edges, right=False) - 1,
        0, n_bins - 1,
    )

    rows = []
    for keys, group in work.groupby(list(group_cols), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        for bin_idx, bin_group in group.groupby("bin_idx"):
            n = len(bin_group)
            if n < min_per_bin:
                continue
            k = int(bin_group[outcome_col].sum())
            lo, hi = _wilson_interval(k, n)
            row = dict(zip(group_cols, keys))
            row.update({
                "bin_mid": float(mids[int(bin_idx)]),
                "predicted_mean": float(bin_group[prob_col].mean()),
                "observed_rate": float(k / n),
                "observed_ci_low": float(lo),
                "observed_ci_high": float(hi),
                "n": int(n),
            })
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Score-distribution histogram (binary discrimination)
# ---------------------------------------------------------------------------


def score_distribution(
    df: pd.DataFrame,
    *,
    prob_col: str = "p_has_ebike",
    outcome_col: str = "observed_has_ebike",
    group_cols: Iterable[str] = ("model_key", "model_label"),
    n_bins: int = 20,
) -> pd.DataFrame:
    """Histogram of predicted probabilities split by realised outcome.

    Returned columns: group_cols + ``outcome`` ("y=1" | "y=0"), ``bin_mid``,
    ``density`` (per-outcome relative frequency). Used to render the
    discrimination overlay: well-separated histograms = sharp + discriminating.
    """
    if df.empty or prob_col not in df or outcome_col not in df:
        return pd.DataFrame(columns=list(group_cols) + ["outcome", "bin_mid", "density"])
    work = df[[*group_cols, prob_col, outcome_col]].copy()
    work[prob_col] = pd.to_numeric(work[prob_col], errors="coerce").clip(0.0, 1.0)
    work[outcome_col] = pd.to_numeric(work[outcome_col], errors="coerce")
    work = work.dropna(subset=[prob_col, outcome_col])
    if work.empty:
        return pd.DataFrame(columns=list(group_cols) + ["outcome", "bin_mid", "density"])

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    mids = (edges[:-1] + edges[1:]) / 2.0

    rows = []
    for keys, group in work.groupby(list(group_cols), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        for outcome_value, label in ((1.0, "y=1 (had bike)"), (0.0, "y=0 (no bike)")):
            sub = group[group[outcome_col] == outcome_value]
            if sub.empty:
                continue
            counts, _ = np.histogram(sub[prob_col].to_numpy(), bins=edges)
            total = counts.sum()
            if total == 0:
                continue
            density = counts / total
            for mid, d in zip(mids, density):
                row = dict(zip(group_cols, keys))
                row.update({
                    "outcome": label,
                    "bin_mid": float(mid),
                    "density": float(d),
                    "n_outcome": int(total),
                })
                rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Sharpness for binary forecasts
# ---------------------------------------------------------------------------


def sharpness_variance(p: pd.Series | np.ndarray) -> float | None:
    """Mean predicted variance p(1-p). Lower = sharper.

    For a perfectly confident model that outputs 0/1, sharpness → 0.
    For a model that always says 0.5, sharpness = 0.25 (worst case).
    Reference: Gneiting et al. (2007) — sharpness principle.
    """
    arr = pd.to_numeric(pd.Series(p), errors="coerce").dropna()
    if arr.empty:
        return None
    arr = arr.clip(0.0, 1.0)
    return float((arr * (1.0 - arr)).mean())


def sharpness_ece_scatter(
    df: pd.DataFrame,
    *,
    prob_col: str = "p_has_ebike",
    outcome_col: str = "observed_has_ebike",
    model_col: str = "model_key",
    horizon_col: str = "horizon_minutes",
    bucket_cols: Iterable[str] = ("hour_band",),
    min_per_bucket: int = 30,
) -> pd.DataFrame:
    """Per-bucket sharpness vs ECE, one row per (model, horizon, bucket).

    Caller is expected to have already attached a ``hour_band`` (or whatever
    ``bucket_cols`` enumerates) column. Buckets with fewer than
    ``min_per_bucket`` resolved forecasts are dropped.
    """
    from . import decision_metrics

    needed = {prob_col, outcome_col, model_col, horizon_col, *bucket_cols}
    if df.empty or not needed.issubset(df.columns):
        return pd.DataFrame(
            columns=[model_col, horizon_col, *bucket_cols, "sharpness", "ece", "n"]
        )
    work = df[list(needed)].copy()
    work[prob_col] = pd.to_numeric(work[prob_col], errors="coerce").clip(0.0, 1.0)
    work[outcome_col] = pd.to_numeric(work[outcome_col], errors="coerce")
    work = work.dropna(subset=[prob_col, outcome_col])

    rows = []
    for keys, group in work.groupby([model_col, horizon_col, *bucket_cols], dropna=False):
        n = len(group)
        if n < min_per_bucket:
            continue
        ece = decision_metrics.ece_score(
            group[outcome_col].to_numpy(),
            group[prob_col].to_numpy(),
        )
        if ece is None:
            continue
        sharpness = sharpness_variance(group[prob_col])
        if sharpness is None:
            continue
        if not isinstance(keys, tuple):
            keys = (keys,)
        col_names = [model_col, horizon_col, *bucket_cols]
        row = dict(zip(col_names, keys))
        row.update({
            "sharpness": float(sharpness),
            "ece": float(ece),
            "n": int(n),
            "mean_prediction": float(group[prob_col].mean()),
            "observed_rate": float(group[outcome_col].mean()),
        })
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Murphy decomposition of the Brier score
# ---------------------------------------------------------------------------


def brier_decomposition(
    df: pd.DataFrame,
    *,
    prob_col: str = "p_has_ebike",
    outcome_col: str = "observed_has_ebike",
    n_bins: int = 10,
) -> dict:
    """Compute the Murphy three-component decomposition BS = REL - RES + UNC.

    Returns a dict with keys ``brier``, ``reliability``, ``resolution``,
    ``uncertainty``, ``n``. ``brier ≈ reliability - resolution + uncertainty``
    holds exactly when bin means are used; the chart layer rounds for display.

    Reference: Murphy (1973), Bröcker (2009). The three components answer:
      - reliability: how far per-bin observed rates are from per-bin predicted
        means (calibration error in MSE units). Lower is better.
      - resolution: how much per-bin observed rates differ from the marginal
        rate. Higher is better — measures discrimination.
      - uncertainty: marginal variance ȳ(1-ȳ). Fixed by data.
    """
    if df.empty or prob_col not in df or outcome_col not in df:
        return {"brier": None, "reliability": None, "resolution": None, "uncertainty": None, "n": 0}
    p = pd.to_numeric(df[prob_col], errors="coerce").clip(0.0, 1.0)
    y = pd.to_numeric(df[outcome_col], errors="coerce")
    mask = p.notna() & y.notna()
    p, y = p[mask].to_numpy(), y[mask].to_numpy()
    n = len(p)
    if n == 0:
        return {"brier": None, "reliability": None, "resolution": None, "uncertainty": None, "n": 0}

    y_bar = float(y.mean())
    brier = float(((p - y) ** 2).mean())
    uncertainty = float(y_bar * (1.0 - y_bar))

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(p, edges, right=False) - 1, 0, n_bins - 1)

    reliability = 0.0
    resolution = 0.0
    for k in range(n_bins):
        mask = bin_idx == k
        nk = int(mask.sum())
        if nk == 0:
            continue
        p_k = float(p[mask].mean())
        y_k = float(y[mask].mean())
        reliability += (nk / n) * (p_k - y_k) ** 2
        resolution += (nk / n) * (y_k - y_bar) ** 2

    return {
        "brier": brier,
        "reliability": float(reliability),
        "resolution": float(resolution),
        "uncertainty": uncertainty,
        "n": int(n),
        "marginal_rate": y_bar,
    }


def brier_decomposition_by_model(
    df: pd.DataFrame,
    *,
    prob_col: str = "p_has_ebike",
    outcome_col: str = "observed_has_ebike",
    group_cols: Iterable[str] = ("model_key", "model_label"),
    n_bins: int = 10,
) -> pd.DataFrame:
    """Per-model Brier decomposition + skill score.

    Returns one row per group, with ``brier``, ``reliability``, ``resolution``,
    ``uncertainty``, ``n``, ``marginal_rate``, and a placeholder ``skill_score``
    field that callers can fill in with a baseline comparison.
    """
    rows = []
    if df.empty:
        return pd.DataFrame(columns=list(group_cols) + [
            "brier", "reliability", "resolution", "uncertainty",
            "n", "marginal_rate",
        ])
    for keys, group in df.groupby(list(group_cols), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        decomp = brier_decomposition(
            group, prob_col=prob_col, outcome_col=outcome_col, n_bins=n_bins
        )
        row = dict(zip(group_cols, keys))
        row.update(decomp)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Randomized PIT for the count PMF (Czado, Gneiting, Held 2009)
# ---------------------------------------------------------------------------


_COUNT_PMF_KEYS = ("0", "1", "2", "3", "4", "5_plus")


def _parse_count_pmf(value) -> dict[str, float] | None:
    """Parse a stored count PMF JSON into a dict {key: prob}.

    Accepts a JSON string, a dict, or a pre-parsed list of {bin, prob}.
    Missing or malformed → None.
    """
    if value is None:
        return None
    if isinstance(value, dict):
        raw = value
    elif isinstance(value, str):
        try:
            raw = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return None
        if not isinstance(raw, dict):
            return None
    else:
        return None
    pmf: dict[str, float] = {}
    for key in _COUNT_PMF_KEYS:
        if key in raw:
            try:
                pmf[key] = float(raw[key])
            except (TypeError, ValueError):
                continue
    if not pmf:
        return None
    total = sum(pmf.values())
    if total <= 0:
        return None
    return {k: v / total for k, v in pmf.items()}


def _observed_bucket(observed: int) -> str:
    if observed <= 0:
        return "0"
    if observed >= 5:
        return "5_plus"
    return str(int(observed))


def randomized_pit(pmf: dict[str, float], observed: int, rng: np.random.Generator) -> float | None:
    """Randomized PIT for one (PMF, observed count) pair.

    Czado et al. 2009: ``u ~ Uniform(F(observed - 1), F(observed))`` so that
    under perfect calibration the marginal distribution of ``u`` is Uniform[0,1].
    """
    if pmf is None or observed is None:
        return None
    cumulative = 0.0
    f_lo = 0.0
    f_hi = 0.0
    target_bucket = _observed_bucket(int(observed))
    found = False
    for key in _COUNT_PMF_KEYS:
        prob = pmf.get(key, 0.0)
        if key == target_bucket:
            f_lo = cumulative
            cumulative += prob
            f_hi = cumulative
            found = True
            break
        cumulative += prob
    if not found:
        # Observed bucket wasn't in the PMF support — should not happen for
        # well-formed PMFs but be defensive.
        return None
    if f_hi - f_lo <= 1e-12:
        return float(f_hi)
    return float(rng.uniform(f_lo, f_hi))


def count_pit_values(
    df: pd.DataFrame,
    *,
    pmf_col: str = "p_count_ebikes_json",
    observed_col: str = "observed_ebikes",
    group_cols: Iterable[str] = ("model_key", "model_label"),
    seed: int = 7,
) -> pd.DataFrame:
    """Compute one randomized-PIT value per resolved count forecast.

    Returns columns: group_cols + ``pit``. Rows with missing PMF or observed
    count are dropped silently.
    """
    if df.empty or pmf_col not in df or observed_col not in df:
        return pd.DataFrame(columns=list(group_cols) + ["pit"])
    rng = np.random.default_rng(seed)
    out_rows = []
    for _, row in df.iterrows():
        pmf = _parse_count_pmf(row.get(pmf_col))
        observed = row.get(observed_col)
        if pmf is None or observed is None or pd.isna(observed):
            continue
        pit = randomized_pit(pmf, int(observed), rng)
        if pit is None:
            continue
        new = {col: row.get(col) for col in group_cols}
        new["pit"] = pit
        out_rows.append(new)
    return pd.DataFrame(out_rows)


def count_pit_histogram(
    df: pd.DataFrame,
    *,
    n_bins: int = 10,
    pmf_col: str = "p_count_ebikes_json",
    observed_col: str = "observed_ebikes",
    group_cols: Iterable[str] = ("model_key", "model_label"),
    seed: int = 7,
) -> pd.DataFrame:
    """Histogram of randomized PIT values per group.

    Returned columns: group_cols + ``bin_mid``, ``density``, ``n``.
    """
    pits = count_pit_values(
        df,
        pmf_col=pmf_col,
        observed_col=observed_col,
        group_cols=group_cols,
        seed=seed,
    )
    if pits.empty:
        return pd.DataFrame(columns=list(group_cols) + ["bin_mid", "density", "n"])
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    mids = (edges[:-1] + edges[1:]) / 2.0
    rows = []
    for keys, group in pits.groupby(list(group_cols), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        counts, _ = np.histogram(group["pit"].to_numpy(), bins=edges)
        total = int(counts.sum())
        if total == 0:
            continue
        densities = counts / total * n_bins  # density-normalized so flat = 1.0
        for mid, d in zip(mids, densities):
            row = dict(zip(group_cols, keys))
            row.update({"bin_mid": float(mid), "density": float(d), "n": total})
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Coverage / ECE heatmap by hour × day-of-week
# ---------------------------------------------------------------------------


_LOCAL_TZ = "America/Chicago"
_DOW_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def time_of_week_features(df: pd.DataFrame, *, ts_col: str = "forecasted_at") -> pd.DataFrame:
    """Attach ``local_hour`` and ``day_of_week`` columns based on a UTC timestamp."""
    if df.empty or ts_col not in df:
        return df
    ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    local = ts.dt.tz_convert(_LOCAL_TZ)
    out = df.copy()
    out["local_hour"] = local.dt.hour
    # Monday=0 in pandas, our label list starts at Mon.
    out["day_of_week"] = local.dt.dayofweek.map(
        lambda i: _DOW_LABELS[i] if isinstance(i, (int, np.integer)) and not pd.isna(i) else None
    )
    out["weekday_or_weekend"] = local.dt.dayofweek.map(
        lambda i: "weekend" if i in (5, 6) else "weekday" if i is not None and not pd.isna(i) else None
    )
    return out


def coverage_heatmap_data(
    df: pd.DataFrame,
    *,
    prob_col: str = "p_has_ebike",
    outcome_col: str = "observed_has_ebike",
    model_col: str = "model_key",
    model_label_col: str = "model_label",
    ts_col: str = "forecasted_at",
    min_per_cell: int = 20,
) -> pd.DataFrame:
    """Per (model, day, hour) cell: ECE, mean prediction, observed rate, n.

    Returned columns: ``model_key``, ``model_label``, ``day_of_week``,
    ``local_hour``, ``n``, ``mean_prediction``, ``observed_rate``,
    ``ece``, ``calibration_gap``. Cells with fewer than ``min_per_cell``
    samples are dropped.
    """
    if df.empty:
        return pd.DataFrame(columns=[
            model_col, model_label_col, "day_of_week", "local_hour",
            "n", "mean_prediction", "observed_rate", "ece", "calibration_gap",
        ])
    work = time_of_week_features(df, ts_col=ts_col)
    work = work.dropna(subset=["local_hour", "day_of_week"])
    work[prob_col] = pd.to_numeric(work[prob_col], errors="coerce").clip(0.0, 1.0)
    work[outcome_col] = pd.to_numeric(work[outcome_col], errors="coerce")
    work = work.dropna(subset=[prob_col, outcome_col])

    from . import decision_metrics

    rows = []
    grouped = work.groupby([model_col, model_label_col, "day_of_week", "local_hour"], dropna=False)
    for keys, group in grouped:
        n = len(group)
        if n < min_per_cell:
            continue
        model_key, model_label, dow, hour = keys
        ece = decision_metrics.ece_score(
            group[outcome_col].to_numpy(),
            group[prob_col].to_numpy(),
            n_bins=5,  # smaller binning at cell level since n is small
        )
        if ece is None:
            continue
        mean_pred = float(group[prob_col].mean())
        observed = float(group[outcome_col].mean())
        rows.append({
            model_col: model_key,
            model_label_col: model_label,
            "day_of_week": dow,
            "local_hour": int(hour),
            "n": n,
            "mean_prediction": mean_pred,
            "observed_rate": observed,
            "ece": float(ece),
            "calibration_gap": mean_pred - observed,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Skill score vs baseline
# ---------------------------------------------------------------------------


def skill_score(model_brier: float | None, baseline_brier: float | None) -> float | None:
    """Brier skill score vs a named baseline. >0 = better."""
    if model_brier is None or baseline_brier is None:
        return None
    if pd.isna(model_brier) or pd.isna(baseline_brier):
        return None
    if baseline_brier <= 0:
        return None
    return 1.0 - float(model_brier) / float(baseline_brier)


def attach_skill_scores(
    leaderboard: list[dict],
    *,
    baseline_key: str = "empirical",
    metric: str = "brier_score",
) -> list[dict]:
    """Add a ``skill_score`` field to each leaderboard row, computed against
    the row with ``model_key == baseline_key``."""
    baseline_row = next((r for r in leaderboard if r.get("model_key") == baseline_key), None)
    baseline_value = baseline_row.get(metric) if baseline_row else None
    out = []
    for row in leaderboard:
        new = dict(row)
        new["skill_score"] = skill_score(row.get(metric), baseline_value)
        out.append(new)
    return out


# ---------------------------------------------------------------------------
# Frequency-dot grid positions (for the rider-facing icon array)
# ---------------------------------------------------------------------------


def dot_grid_positions(
    probability: float,
    *,
    n: int = 100,
    cols: int = 10,
) -> pd.DataFrame:
    """Return ``n`` rows with ``x``, ``y``, ``filled`` indicating which dots
    are highlighted to represent ``probability``.

    Dots are filled left-to-right, bottom-to-top (so the eye reads the filled
    fraction as a rising stack). The fill count rounds half-up for stable
    visual semantics across small changes.
    """
    if probability is None or pd.isna(probability):
        probability = 0.0
    probability = max(0.0, min(1.0, float(probability)))
    rows = max(1, int(math.ceil(n / cols)))
    n = rows * cols
    filled_count = int(round(probability * n))
    out = []
    for idx in range(n):
        col = idx % cols
        row = rows - 1 - (idx // cols)  # bottom-up
        out.append({
            "x": col,
            "y": row,
            "idx": idx,
            "filled": idx < filled_count,
        })
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Trend extraction from the model_metrics table
# ---------------------------------------------------------------------------


def metric_trend(
    conn: duckdb.DuckDBPyConnection,
    *,
    metric_columns: tuple[str, ...] = ("brier_score", "log_loss", "rank_loss"),
    days: int = 14,
    by_model: bool = True,
) -> pd.DataFrame:
    """Pull rolling per-model metric snapshots for trend lines.

    Reads ``model_metrics`` (which is populated by ``poller.snapshot_metrics``).
    Returns one row per ``computed_at × model_key`` (or computed_at × 'overall'
    if ``by_model`` is False) with the requested metric columns.
    """
    safe_cols = [c for c in metric_columns if c.isidentifier()]
    if not safe_cols:
        return pd.DataFrame()
    col_sql = ", ".join(safe_cols)
    if by_model:
        sql = f"""
            SELECT computed_at, group_value AS model_key, n, {col_sql}
            FROM model_metrics
            WHERE group_key = 'model'
              AND computed_at > now() - (? * INTERVAL '1 day')
            ORDER BY computed_at
        """
    else:
        sql = f"""
            SELECT computed_at, 'overall' AS model_key, n, {col_sql}
            FROM model_metrics
            WHERE group_key = 'overall'
              AND computed_at > now() - (? * INTERVAL '1 day')
            ORDER BY computed_at
        """
    try:
        return conn.execute(sql, [days]).df()
    except duckdb.Error:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Worst-offender drill-in table
# ---------------------------------------------------------------------------


def worst_station_hours(
    df: pd.DataFrame,
    *,
    prob_col: str = "p_has_ebike",
    outcome_col: str = "observed_has_ebike",
    station_id_col: str = "station_id",
    station_name_col: str = "station_name",
    ts_col: str = "forecasted_at",
    model_col: str = "model_key",
    min_n: int = 5,
    top: int = 25,
) -> pd.DataFrame:
    """Top-N worst (station, hour-of-day, model) buckets by Brier.

    Returned columns: ``model_key``, ``station_id``, ``station_name``,
    ``local_hour``, ``n``, ``brier``, ``mean_prediction``, ``observed_rate``,
    ``calibration_gap``. Used as the drill-in companion to the coverage
    heatmap when the maintainer needs to find a specific bad cell.
    """
    if df.empty:
        return pd.DataFrame(columns=[
            model_col, station_id_col, station_name_col, "local_hour",
            "n", "brier", "mean_prediction", "observed_rate", "calibration_gap",
        ])
    work = time_of_week_features(df, ts_col=ts_col).dropna(subset=["local_hour"])
    work[prob_col] = pd.to_numeric(work[prob_col], errors="coerce").clip(0.0, 1.0)
    work[outcome_col] = pd.to_numeric(work[outcome_col], errors="coerce")
    work = work.dropna(subset=[prob_col, outcome_col])
    rows = []
    grouped = work.groupby(
        [model_col, station_id_col, station_name_col, "local_hour"], dropna=False
    )
    for keys, group in grouped:
        n = len(group)
        if n < min_n:
            continue
        model_key, sid, name, hour = keys
        p = group[prob_col].to_numpy()
        y = group[outcome_col].to_numpy()
        brier = float(((p - y) ** 2).mean())
        mean_pred = float(p.mean())
        observed = float(y.mean())
        rows.append({
            model_col: model_key,
            station_id_col: sid,
            station_name_col: name,
            "local_hour": int(hour),
            "n": int(n),
            "brier": brier,
            "mean_prediction": mean_pred,
            "observed_rate": observed,
            "calibration_gap": mean_pred - observed,
        })
    out = pd.DataFrame(rows).sort_values("brier", ascending=False)
    return out.head(top).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Resolved-forecast pull for the dashboard (joined frame)
# ---------------------------------------------------------------------------


def resolved_forecasts(
    conn: duckdb.DuckDBPyConnection,
    *,
    window_hours: int = 24,
    sources: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Pull the joined `(forecast, outcome)` frame for dashboard charts.

    Thin convenience wrapper around the same SQL pattern as
    ``model_eval._joined_forecasts``, returning only the columns the
    dashboard charts use. Splitting this out keeps charts decoupled from
    the metric-snapshot machinery in ``model_eval``.
    """
    where_clauses = ["f.forecasted_at >= now() - (? * INTERVAL '1 hour')"]
    params: list = [window_hours]
    if sources:
        placeholders = ",".join("?" for _ in sources)
        where_clauses.append(f"f.source IN ({placeholders})")
        params.extend(list(sources))
    where_sql = " AND ".join(where_clauses)
    try:
        return conn.execute(
            f"""
            SELECT
              f.forecast_id,
              COALESCE(f.model_key, 'logistic') AS model_key,
              COALESCE(f.model_label, f.model_version) AS model_label,
              f.station_id,
              f.station_name,
              f.forecasted_at,
              f.target_at,
              f.horizon_minutes,
              f.current_ebikes,
              f.p_has_ebike,
              f.p_zero,
              f.expected_ebikes,
              f.expected_total_bikes,
              f.p_count_ebikes_json,
              f.p_capacity_violation,
              f.p_dock_constrained_arrival,
              f.is_recommended,
              o.observed_at,
              o.observed_ebikes,
              o.observed_has_ebike,
              o.observed_total_bikes,
              o.count_log_prob,
              o.crps
            FROM model_forecasts f
            JOIN model_outcomes o USING (forecast_id)
            WHERE {where_sql}
            """,
            params,
        ).df()
    except duckdb.Error:
        return pd.DataFrame()
