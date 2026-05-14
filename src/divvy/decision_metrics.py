from __future__ import annotations

import math

import numpy as np
import pandas as pd


def expected_utility(
    p_arrival,
    distance_km,
    search_radius_km: float = 1.5,
    distance_weight: float = 0.35,
) -> np.ndarray:
    p = np.asarray(p_arrival, dtype=float)
    d = np.asarray(distance_km, dtype=float)
    proximity = np.clip(1.0 - d / max(0.1, float(search_radius_km)), 0.0, 1.0)
    return np.clip(p, 0.0, 1.0) + distance_weight * proximity


def distance_adjusted_regret(
    chosen_utility: float | None,
    oracle_utility: float | None,
) -> float | None:
    if chosen_utility is None or oracle_utility is None:
        return None
    if not math.isfinite(float(chosen_utility)) or not math.isfinite(float(oracle_utility)):
        return None
    return float(max(0.0, oracle_utility - chosen_utility))


def ece_score(y_true, y_prob, n_bins: int = 10) -> float | None:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    if not mask.any():
        return None
    y = y[mask]
    p = np.clip(p[mask], 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    total = len(p)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        if hi == 1.0:
            m = (p >= lo) & (p <= hi)
        else:
            m = (p >= lo) & (p < hi)
        if not m.any():
            continue
        ece += float(m.mean()) * abs(float(p[m].mean()) - float(y[m].mean()))
    return float(ece)


def decision_rank_loss(metrics: dict) -> float | None:
    brier = metrics.get("brier_score")
    log_loss = metrics.get("log_loss")
    ece = metrics.get("ece")
    regret = metrics.get("distance_adjusted_regret")
    if regret is not None:
        return float(
            float(regret)
            + 0.25 * float(brier or 0.0)
            + 0.05 * float(log_loss or 0.0)
            + 0.10 * float(ece or 0.0)
        )
    if brier is None or log_loss is None:
        return None
    return float(float(brier) + 0.05 * float(log_loss))


def recommended_precision(df: pd.DataFrame) -> dict | None:
    if df.empty or "observed_has_ebike" not in df:
        return None
    recommended = df[
        (df.get("is_recommended", False) == True)  # noqa: E712
        | df.get("decision_role", pd.Series("", index=df.index)).astype(str).str.contains("best", na=False)
    ]
    if recommended.empty:
        return None
    return {
        "n": int(len(recommended)),
        "hit_rate": float(recommended["observed_has_ebike"].astype(float).mean()),
        "mean_prediction": float(pd.to_numeric(recommended.get("p_arrival", recommended["p_has_ebike"]), errors="coerce").mean()),
    }
