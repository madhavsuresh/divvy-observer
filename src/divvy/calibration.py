from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def _clip_prob(values) -> np.ndarray:
    return np.asarray(values, dtype=float).clip(0.001, 0.999)


@dataclass
class HorizonCalibrator:
    method_by_horizon: dict[int, str] = field(default_factory=dict)
    model_by_horizon: dict[int, object] = field(default_factory=dict)
    temperature_by_horizon: dict[int, float] = field(default_factory=dict)

    def fit(self, df: pd.DataFrame, prob_col: str = "p", y_col: str = "y", horizon_col: str = "horizon_minutes") -> "HorizonCalibrator":
        for horizon, group in df.groupby(horizon_col):
            if len(group) < 20 or group[y_col].nunique() < 2:
                self.method_by_horizon[int(horizon)] = "identity"
                continue
            p = _clip_prob(group[prob_col])
            y = group[y_col].astype(int).to_numpy()
            try:
                if len(group) >= 80:
                    from sklearn.isotonic import IsotonicRegression

                    model = IsotonicRegression(out_of_bounds="clip")
                    model.fit(p, y)
                    self.model_by_horizon[int(horizon)] = model
                    self.method_by_horizon[int(horizon)] = "isotonic"
                else:
                    from sklearn.linear_model import LogisticRegression

                    logits = np.log(p / (1.0 - p)).reshape(-1, 1)
                    model = LogisticRegression(max_iter=1000)
                    model.fit(logits, y)
                    self.model_by_horizon[int(horizon)] = model
                    self.method_by_horizon[int(horizon)] = "platt"
            except Exception:
                self.method_by_horizon[int(horizon)] = "identity"
        return self

    def predict(self, df: pd.DataFrame, prob_col: str = "p", horizon_col: str = "horizon_minutes") -> np.ndarray:
        probs = _clip_prob(df[prob_col])
        out = probs.copy()
        horizons = df[horizon_col].astype(int).to_numpy() if horizon_col in df else np.zeros(len(df), dtype=int)
        for horizon in np.unique(horizons):
            mask = horizons == horizon
            method = self.method_by_horizon.get(int(horizon), "identity")
            model = self.model_by_horizon.get(int(horizon))
            if model is None or method == "identity":
                continue
            if method == "isotonic":
                out[mask] = model.predict(probs[mask])
            elif method == "platt":
                logits = np.log(probs[mask] / (1.0 - probs[mask])).reshape(-1, 1)
                out[mask] = model.predict_proba(logits)[:, 1]
        return _clip_prob(out)


def fit_platt_by_horizon(df: pd.DataFrame, prob_col: str = "p", y_col: str = "y") -> HorizonCalibrator:
    return HorizonCalibrator().fit(df, prob_col=prob_col, y_col=y_col)


def fit_isotonic_by_horizon(df: pd.DataFrame, prob_col: str = "p", y_col: str = "y") -> HorizonCalibrator:
    calibrator = HorizonCalibrator()
    for horizon, group in df.groupby("horizon_minutes"):
        if len(group) < 30 or group[y_col].nunique() < 2:
            calibrator.method_by_horizon[int(horizon)] = "identity"
            continue
        try:
            from sklearn.isotonic import IsotonicRegression

            model = IsotonicRegression(out_of_bounds="clip")
            model.fit(_clip_prob(group[prob_col]), group[y_col].astype(int))
            calibrator.model_by_horizon[int(horizon)] = model
            calibrator.method_by_horizon[int(horizon)] = "isotonic"
        except Exception:
            calibrator.method_by_horizon[int(horizon)] = "identity"
    return calibrator


def temperature_scale_probabilities(probs, temperature: float = 1.0) -> np.ndarray:
    p = _clip_prob(probs)
    temp = max(0.05, float(temperature) if math.isfinite(float(temperature)) else 1.0)
    logits = np.log(p / (1.0 - p)) / temp
    return _clip_prob(1.0 / (1.0 + np.exp(-np.clip(logits, -35.0, 35.0))))


def conformal_lower_confidence_bound(
    df: pd.DataFrame,
    prob_col: str = "p",
    y_col: str = "y",
    group_cols: tuple[str, ...] = ("horizon_minutes",),
    alpha: float = 0.10,
) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    out = pd.Series(_clip_prob(df[prob_col]), index=df.index, dtype=float)
    alpha = max(0.01, min(0.5, float(alpha)))
    for _, group in df.groupby(list(group_cols), dropna=False):
        if len(group) < 10:
            out.loc[group.index] = np.maximum(0.0, out.loc[group.index] - 0.15)
            continue
        residual = (_clip_prob(group[prob_col]) - group[y_col].astype(float).to_numpy())
        q = float(np.quantile(np.maximum(0.0, residual), 1.0 - alpha))
        out.loc[group.index] = np.maximum(0.0, _clip_prob(group[prob_col]) - q)
    return out.clip(0.0, 1.0)
