"""Scoring utilities for DG-NISSM v2: CDG-NMIP evaluation.

Pure numpy/pandas — no torch dependency. Importable from tests and from
`scripts/evaluate_dg_nissm.py` without bringing in the model layer.
"""
from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


def brier_score(p, y) -> float:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    if p.size == 0:
        return float("nan")
    return float(np.mean((p - y) ** 2))


def log_loss(p, y, eps: float = 1e-12) -> float:
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    y = np.asarray(y, dtype=float)
    if p.size == 0:
        return float("nan")
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def reliability_curve(p, y, n_bins: int = 10):
    """Equal-width binning.

    Returns four arrays of length ``n_bins``: bin centers, mean predicted
    probability per bin, observed positive rate per bin, count per bin. Bins
    with zero count carry ``NaN`` in the predicted/observed slots.
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    pred = np.full(n_bins, np.nan, dtype=float)
    obs = np.full(n_bins, np.nan, dtype=float)
    cnt = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        mask = idx == b
        cnt[b] = int(mask.sum())
        if cnt[b] > 0:
            pred[b] = float(p[mask].mean())
            obs[b] = float(y[mask].mean())
    return centers, pred, obs, cnt


def ece(p, y, n_bins: int = 10) -> float:
    """Expected Calibration Error, equal-width bins, count-weighted."""
    _, pred, obs, cnt = reliability_curve(p, y, n_bins=n_bins)
    total = int(cnt.sum())
    if total == 0:
        return float("nan")
    mask = cnt > 0
    return float(np.sum(np.abs(pred[mask] - obs[mask]) * cnt[mask] / total))


def crps_count_pmf(
    pmf: Mapping,
    observed,
    *,
    max_int_key: int = 4,
    tail_key: str = "5_plus",
    tail_position: int = 5,
) -> float:
    """Discrete CRPS for the coarse count PMF schema used by DG-NISSM.

    The PMF maps integer-string keys ``"0".."4"`` to atom probabilities and a
    ``"5_plus"`` tail bucket; the tail bucket is treated as a single atom at
    ``tail_position`` for scoring purposes.
    """
    obs = max(0, int(round(float(observed))))
    pmf = dict(pmf or {})
    probs: dict[int, float] = {}
    for k in range(max_int_key + 1):
        probs[k] = float(pmf.get(str(k), 0.0)) + float(pmf.get(k, 0.0))
    probs[tail_position] = float(pmf.get(tail_key, 0.0))
    grid_max = max(max(probs.keys(), default=0), obs)
    cdf = 0.0
    crps = 0.0
    for k in range(grid_max + 1):
        cdf += probs.get(k, 0.0)
        ind = 1.0 if k >= obs else 0.0
        crps += (cdf - ind) ** 2
    return float(crps)


def crps_count_pmf_mean(pmfs: Iterable, observed: Iterable[int], **kwargs) -> float:
    """Mean CRPS over a batch of (pmf, observation) pairs."""
    values: list[float] = []
    for pmf, y in zip(pmfs, observed):
        if not isinstance(pmf, Mapping):
            continue
        try:
            values.append(crps_count_pmf(pmf, y, **kwargs))
        except Exception:
            continue
    if not values:
        return float("nan")
    return float(np.mean(values))


def stratified_metrics(
    preds: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    by: Sequence[str],
    p_col: str = "p_has_ebike",
    y_col: str = "has_ebike",
    pmf_col: str | None = None,
    obs_count_col: str | None = None,
) -> pd.DataFrame:
    """Compute Brier/log-loss/ECE (and optionally CRPS) per slice.

    ``preds`` and ``labels`` must share the same index. ``by`` lists the column
    names to group on; those columns are looked up in ``labels`` first, then
    ``preds`` as a fallback.
    """
    if preds.empty:
        return pd.DataFrame(columns=[*by, "n", "brier", "log_loss", "ece"])
    df = preds.copy()
    for col in by:
        if col in labels.columns:
            df[col] = labels[col].to_numpy()
        elif col not in df.columns:
            df[col] = np.nan
    df["__y__"] = pd.to_numeric(labels[y_col], errors="coerce").fillna(0).to_numpy()
    df["__p__"] = pd.to_numeric(df[p_col], errors="coerce").fillna(0.0).clip(1e-12, 1.0 - 1e-12).to_numpy()
    if pmf_col is not None and obs_count_col is not None:
        df["__pmf__"] = df[pmf_col].tolist()
        df["__obs__"] = pd.to_numeric(labels[obs_count_col], errors="coerce").fillna(0).astype(int).to_numpy()

    rows: list[dict] = []
    if not list(by):
        groups = [((), df)]
    else:
        groups = [(key if isinstance(key, tuple) else (key,), grp) for key, grp in df.groupby(list(by), dropna=False)]
    for key, grp in groups:
        record = {col: val for col, val in zip(by, key)}
        record["n"] = int(len(grp))
        record["brier"] = brier_score(grp["__p__"].to_numpy(), grp["__y__"].to_numpy())
        record["log_loss"] = log_loss(grp["__p__"].to_numpy(), grp["__y__"].to_numpy())
        record["ece"] = ece(grp["__p__"].to_numpy(), grp["__y__"].to_numpy())
        record["p_mean"] = float(grp["__p__"].mean())
        record["y_mean"] = float(grp["__y__"].mean())
        if pmf_col is not None and obs_count_col is not None:
            record["crps"] = crps_count_pmf_mean(grp["__pmf__"].tolist(), grp["__obs__"].tolist())
        rows.append(record)
    return pd.DataFrame(rows)


def uniform_count_pmf(max_int_key: int = 4, tail_key: str = "5_plus") -> dict:
    """Reference: a flat PMF over the coarse count schema (worst-case baseline)."""
    n = max_int_key + 2
    p = 1.0 / n
    out = {str(k): p for k in range(max_int_key + 1)}
    out[tail_key] = p
    return out


def empirical_count_pmf(observed: Iterable[int], max_int_key: int = 4, tail_key: str = "5_plus") -> dict:
    """Reference: the empirical PMF of observed counts on the coarse schema."""
    counts = np.zeros(max_int_key + 2, dtype=float)
    n = 0
    for y in observed:
        try:
            yi = max(0, int(round(float(y))))
        except (TypeError, ValueError):
            continue
        if yi >= max_int_key + 1:
            counts[-1] += 1.0
        else:
            counts[yi] += 1.0
        n += 1
    if n == 0:
        return uniform_count_pmf(max_int_key=max_int_key, tail_key=tail_key)
    counts = counts / n
    out = {str(k): float(counts[k]) for k in range(max_int_key + 1)}
    out[tail_key] = float(counts[-1])
    return out
