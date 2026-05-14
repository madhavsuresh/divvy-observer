from __future__ import annotations

import pickle
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import inventory_dp


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _series(rows: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in rows.columns:
        return pd.Series(default, index=rows.index, dtype=float)
    return pd.to_numeric(rows[column], errors="coerce").fillna(default).astype(float)


class CCNISSMModel:
    """Censored constrained neural inventory state-space model interface.

    This implementation is intentionally lightweight for request-path safety:
    it estimates constrained inventory-flow intensities from the existing
    leak-free feature set and decodes those intensities through inventory DP.
    Persisted neural artifacts can keep the same ``predict_distribution`` API.
    """

    model_key = "cc_nissm"
    model_family = "cc_nissm"
    model_version = "cc-nissm-bootstrap-v1"
    method = "cc_nissm_bootstrap_inventory_flow"

    def __init__(self, flow_scale: float = 1.0, trained_at: datetime | None = None) -> None:
        self.flow_scale = float(flow_scale)
        self.trained_at = trained_at or _utc_now()
        self.feature_columns: list[str] = []
        self.training_examples = 0
        self.training_positive = 0
        self.model_warning: str | None = "No trained artifact found; using lightweight fallback."

    def fit(self, train_df: pd.DataFrame, valid_df: pd.DataFrame | None = None) -> "CCNISSMModel":
        del valid_df
        self.training_examples = int(len(train_df))
        if not train_df.empty and "y_has_ebike" in train_df:
            self.training_positive = int(train_df["y_has_ebike"].sum())
        elif not train_df.empty and "has_ebike" in train_df:
            self.training_positive = int(train_df["has_ebike"].sum())
        flow_cols = [
            "trip_departures_same_hour_10m",
            "trip_arrivals_same_hour_10m",
            "trip_recent_departures_30m",
            "trip_recent_arrivals_30m",
        ]
        available = [c for c in flow_cols if c in train_df.columns]
        if available:
            level = pd.to_numeric(train_df[available].stack(), errors="coerce").dropna()
            if not level.empty:
                self.flow_scale = float(np.clip(level.mean() / 2.0, 0.35, 2.5))
        self.trained_at = _utc_now()
        self.method = f"{self.model_key}_trained_v1"
        self.model_version = str(self.model_key).replace("_", "-") + "-trained-v1"
        self.model_warning = None
        return self

    def _flow_means(self, rows: pd.DataFrame) -> pd.DataFrame:
        horizon_scale = (_series(rows, "horizon_minutes", 10.0).clip(1.0, 30.0) / 10.0)
        current_ebikes = _series(rows, "num_ebikes_available", np.nan).fillna(_series(rows, "current_ebikes_clipped")).clip(0.0, 80.0)
        total_bikes = _series(rows, "num_bikes_available", np.nan).fillna(_series(rows, "current_total_bikes_clipped")).clip(lower=current_ebikes)
        capacity = _series(rows, "capacity", np.nan).fillna(_series(rows, "capacity_clipped", 15.0)).clip(1.0, 80.0)
        docks = _series(rows, "num_docks_available", np.nan).fillna((capacity - total_bikes).clip(lower=0.0)).clip(0.0, 80.0)

        ebike_share = np.where(total_bikes > 0, current_ebikes / total_bikes.clip(lower=1.0), 0.0)
        trip_ebike_share = _series(rows, "trip_ebike_arrival_share_same_hour", 0.25).clip(0.05, 0.95)
        ebike_share = np.maximum(ebike_share, 0.5 * trip_ebike_share)

        trip_depart = (
            _series(rows, "trip_departures_same_hour_10m")
            + _series(rows, "trip_recent_departures_30m") / 3.0
        ).clip(lower=0.0)
        trip_arrive = (
            _series(rows, "trip_arrivals_same_hour_10m")
            + _series(rows, "trip_recent_arrivals_30m") / 3.0
            + _series(rows, "route_inbound_due_horizon")
        ).clip(lower=0.0)
        horizon = _series(rows, "horizon_minutes", 10.0).round().astype(int)
        inflight_ebike = pd.Series(0.0, index=rows.index)
        inflight_classic = pd.Series(0.0, index=rows.index)
        for minutes in (5, 10, 15, 20):
            mask = horizon == minutes
            if mask.any():
                inflight_ebike.loc[mask] = _series(rows.loc[mask], f"live_inflight_ebike_due_{minutes}m")
                inflight_classic.loc[mask] = _series(rows.loc[mask], f"live_inflight_classic_due_{minutes}m")
        trend = (
            0.50 * _series(rows, "trend_5m")
            + 0.30 * _series(rows, "trend_10m")
            + 0.20 * _series(rows, "trend_15m")
        )
        neighbor_support = _series(rows, "station_neighbor_recent_ebikes") * 0.06
        station_prior = _series(rows, "station_same_hour_rate", 0.35).clip(0.01, 0.99)
        neighbor_prior = _series(rows, "station_neighbor_same_hour_rate", 0.35).clip(0.01, 0.99)
        appearance_prior = np.where(
            current_ebikes <= 0,
            -np.log1p(-(0.12 * station_prior + 0.08 * neighbor_prior).clip(0.0, 0.60)),
            0.0,
        )
        weather_factor = (
            1.0
            - 0.20 * _series(rows, "weather_bad_conditions").clip(0.0, 1.0)
            - 0.025 * _series(rows, "weather_precipitation").clip(0.0, 4.0)
        ).clip(0.55, 1.05)
        commute_factor = 1.0 + 0.08 * _series(rows, "is_commute_hour").clip(0.0, 1.0)
        stale_factor = (1.0 - 0.01 * _series(rows, "data_age_minutes", 0.0).clip(0.0, 30.0)).clip(0.70, 1.0)
        return pd.DataFrame({
            "ebike_departure_mean": (
                (trip_depart * np.maximum(0.08, ebike_share) + np.maximum(-trend, 0.0))
                * horizon_scale * weather_factor * commute_factor * self.flow_scale
            ).clip(lower=0.0),
            "classic_departure_mean": (
                trip_depart * np.maximum(0.05, 1.0 - ebike_share)
                * horizon_scale * weather_factor * commute_factor * self.flow_scale
            ).clip(lower=0.0),
            "ebike_arrival_mean": (
                (trip_arrive * trip_ebike_share + inflight_ebike + np.maximum(trend, 0.0) + neighbor_support + appearance_prior)
                * horizon_scale * weather_factor * stale_factor * self.flow_scale
            ).clip(lower=0.0),
            "classic_arrival_mean": (
                (trip_arrive * (1.0 - trip_ebike_share) + inflight_classic)
                * horizon_scale * weather_factor * stale_factor * self.flow_scale
            ).clip(lower=0.0),
        }, index=rows.index)

    def predict_distribution(self, rows: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
        means = self._flow_means(rows)
        results = []
        for idx, row in rows.iterrows():
            rollout = inventory_dp.rollout_inventory_distribution(
                capacity=row.get("capacity_clipped", row.get("capacity")),
                current_ebikes=row.get("num_ebikes_available", row.get("current_ebikes_clipped")),
                current_total_bikes=row.get("num_bikes_available", row.get("current_total_bikes_clipped")),
                ebike_departure_mean=float(means.at[idx, "ebike_departure_mean"]),
                classic_departure_mean=float(means.at[idx, "classic_departure_mean"]),
                ebike_arrival_mean=float(means.at[idx, "ebike_arrival_mean"]),
                classic_arrival_mean=float(means.at[idx, "classic_arrival_mean"]),
            )
            results.append({
                "p_has_ebike": float(np.clip(rollout.p_has_ebike, 0.001, 0.999)),
                "p_zero": float(np.clip(rollout.p_zero, 0.001, 0.999)),
                "expected_ebikes": rollout.expected_ebikes,
                "expected_total_bikes": rollout.expected_total_bikes,
                "p_count_ebikes": rollout.p_count_ebikes,
                "p_count_total": rollout.p_count_total,
                "p_count_ebikes_json": json.dumps(rollout.p_count_ebikes, sort_keys=True),
                "p_count_total_json": json.dumps(rollout.p_count_total, sort_keys=True),
                "p_capacity_violation": rollout.p_capacity_violation,
                "p_dock_constrained_arrival": rollout.p_dock_constrained_arrival,
                "expected_ebike_departures": rollout.expected_ebike_departures,
                "expected_classic_departures": rollout.expected_classic_departures,
                "expected_ebike_arrivals": rollout.expected_ebike_arrivals,
                "expected_classic_arrivals": rollout.expected_classic_arrivals,
            })
            if debug:
                results[-1].update({
                    "mu_e_depart": float(means.at[idx, "ebike_departure_mean"]),
                    "mu_e_arrive": float(means.at[idx, "ebike_arrival_mean"]),
                    "mu_c_depart": float(means.at[idx, "classic_departure_mean"]),
                    "mu_c_arrive": float(means.at[idx, "classic_arrival_mean"]),
                    "theta_e_depart": 20.0,
                    "theta_e_arrive": 20.0,
                    "zero_inflation_e_depart": 0.0,
                    "zero_inflation_e_arrive": 0.0,
                    "dock_constraint_probability": rollout.p_dock_constrained_arrival,
                    "stockout_probability": rollout.p_zero,
                })
        return pd.DataFrame(results, index=rows.index)

    def save(self, path: str | Path) -> None:
        with Path(path).open("wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path: str | Path) -> "CCNISSMModel":
        with Path(path).open("rb") as handle:
            obj = pickle.load(handle)
        if not isinstance(obj, cls):
            raise TypeError(f"Artifact is not {cls.__name__}")
        return obj
