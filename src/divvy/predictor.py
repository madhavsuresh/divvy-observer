from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable

import duckdb
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from . import inventory_dp

try:
    import torch.nn as _torch_nn
except Exception:
    _torch_nn = None

LOCAL_TZ = "America/Chicago"
HORIZONS = (5, 10, 15, 20)
MODEL_VERSION = "inventory-world-v2"
BASELINE_VERSION = "empirical-bayes-v1"
SOTA_PRIMARY_MODEL_KEYS = (
    "dg_nissm",
    "cc_nissm",
    "stg_ncde_inventory",
    "tft_inventory",
)
BASELINE_MODEL_KEYS = (
    "empirical",
    "logistic",
    "random_forest",
    "gradient_boosting",
    "inventory_world",
)
MODEL_KEYS = (
    "dg_nissm",
    "cc_nissm",
    "stg_ncde_inventory",
    "tft_inventory",
    "inventory_world",
    "logistic",
    "random_forest",
    "gradient_boosting",
    "empirical",
)
ACTIVE_MODEL_POLICY = os.getenv("DIVVY_ACTIVE_MODEL_POLICY", "best_sota")
FORCED_ACTIVE_MODEL_KEY = os.getenv("DIVVY_ACTIVE_MODEL_KEY")
ACTIVE_MODEL_KEY = FORCED_ACTIVE_MODEL_KEY or "cc_nissm"
MODEL_SPECS = {
    "dg_nissm": {
        "label": "DG-NISSM dynamic graph inventory flow",
        "version": "dg-nissm-cdg-nmip-v1",
    },
    "cc_nissm": {
        "label": "CC-NISSM constrained inventory flow",
        "version": "cc-nissm-bootstrap-v1",
    },
    "stg_ncde_inventory": {
        "label": "STG-NCDE inventory flow",
        "version": "stg-ncde-inventory-bootstrap-v1",
    },
    "tft_inventory": {
        "label": "TFT inventory flow",
        "version": "tft-inventory-bootstrap-v1",
    },
    "inventory_world": {
        "label": "Distributional inventory world",
        "version": "inventory-world-v2",
    },
    "logistic": {
        "label": "Baseline calibrated logistic",
        "version": "temporal-logistic-v1",
    },
    "random_forest": {
        "label": "Flow/weather random forest",
        "version": "flow-weather-random-forest-v1",
    },
    "gradient_boosting": {
        "label": "Flow/weather gradient boosting",
        "version": "flow-weather-gradient-boosting-v1",
    },
    "empirical": {
        "label": "Empirical Bayes baseline",
        "version": "empirical-bayes-v1",
    },
}
RANK_PROBABILITY_WEIGHT = 0.60
RANK_DISTANCE_WEIGHT = 0.35
RANK_CURRENT_WEIGHT = 0.05
STG_NCDE_MIN_EXAMPLES = int(os.getenv("DIVVY_STG_NCDE_MIN_EXAMPLES", "2000"))
STG_NCDE_MAX_EXAMPLES = int(os.getenv("DIVVY_STG_NCDE_MAX_EXAMPLES", "8000"))
STG_NCDE_EPOCHS = int(os.getenv("DIVVY_STG_NCDE_EPOCHS", "4"))
STG_NCDE_BATCH_SIZE = int(os.getenv("DIVVY_STG_NCDE_BATCH_SIZE", "512"))
STG_NCDE_DEVICE = os.getenv("DIVVY_STG_NCDE_DEVICE", "auto").lower()
REQUEST_BOOTSTRAP_HISTORY_HOURS = int(os.getenv("DIVVY_REQUEST_BOOTSTRAP_HISTORY_HOURS", "2"))
REQUEST_BOOTSTRAP_MAX_SOURCE_ROWS = int(os.getenv("DIVVY_REQUEST_BOOTSTRAP_MAX_SOURCE_ROWS", "50000"))

BASELINE_FEATURE_COLUMNS = [
    "current_ebikes_clipped",
    "current_bucket",
    "horizon_minutes",
    "hour_sin",
    "hour_cos",
    "dow",
    "is_weekend",
    "trend_5m",
    "trend_10m",
    "trend_15m",
    "churn_rate",
    "capacity_clipped",
    "current_total_bikes_clipped",
    "docks_available_clipped",
    "ebike_share_of_bikes",
    "dock_availability_fraction",
    "station_same_hour_rate",
    "nearby_same_hour_rate",
    "station_neighbor_count_500m",
    "station_neighbor_capacity_500m",
    "station_neighbor_same_hour_rate",
    "station_neighbor_recent_ebikes",
    "station_neighbor_recent_zero_rate",
]
EXPERIMENTAL_FEATURE_COLUMNS = [
    *BASELINE_FEATURE_COLUMNS,
    "month_sin",
    "month_cos",
    "day_of_year_sin",
    "day_of_year_cos",
    "is_commute_hour",
    "is_federal_holiday",
    "trip_departures_same_hour_10m",
    "trip_arrivals_same_hour_10m",
    "trip_net_arrivals_same_hour_10m",
    "trip_ebike_arrival_share_same_hour",
    "trip_recent_departures_30m",
    "trip_recent_arrivals_30m",
    "trip_recent_net_arrivals_30m",
    "route_inbound_trips_same_hour",
    "route_inbound_ebike_share_same_hour",
    "route_inbound_median_duration_minutes",
    "route_inbound_due_horizon",
    "weather_temperature_2m",
    "weather_relative_humidity_2m",
    "weather_apparent_temperature",
    "weather_precipitation",
    "weather_rain",
    "weather_snowfall",
    "weather_snow_depth",
    "weather_cloud_cover",
    "weather_wind_speed_10m",
    "weather_wind_gusts_10m",
    "weather_bad_conditions",
]
FEATURE_COLUMNS = EXPERIMENTAL_FEATURE_COLUMNS
TRIP_FEATURE_COLUMNS = [
    "trip_departures_same_hour_10m",
    "trip_arrivals_same_hour_10m",
    "trip_net_arrivals_same_hour_10m",
    "trip_ebike_arrival_share_same_hour",
    "trip_recent_departures_30m",
    "trip_recent_arrivals_30m",
    "trip_recent_net_arrivals_30m",
    "route_inbound_trips_same_hour",
    "route_inbound_ebike_share_same_hour",
    "route_inbound_median_duration_minutes",
    "route_inbound_due_horizon",
]
WEATHER_FEATURE_COLUMNS = [
    "weather_temperature_2m",
    "weather_relative_humidity_2m",
    "weather_apparent_temperature",
    "weather_precipitation",
    "weather_rain",
    "weather_snowfall",
    "weather_snow_depth",
    "weather_cloud_cover",
    "weather_wind_speed_10m",
    "weather_wind_gusts_10m",
    "weather_bad_conditions",
]
LIVE_INFLIGHT_FEATURE_COLUMNS = [
    "live_inflight_ebike_due_5m",
    "live_inflight_ebike_due_10m",
    "live_inflight_ebike_due_15m",
    "live_inflight_ebike_due_20m",
    "live_inflight_classic_due_5m",
    "live_inflight_classic_due_10m",
    "live_inflight_classic_due_15m",
    "live_inflight_classic_due_20m",
]
FREE_FLOATING_FEATURE_COLUMNS = [
    "free_floating_density_300m",
    "free_floating_density_500m",
    "free_floating_density_1000m",
]
STATUS_QUALITY_FEATURE_COLUMNS = [
    "status_age_minutes",
    "station_closed_penalty_flag",
    "stale_status_penalty_flag",
]
CALENDAR_FEATURE_COLUMNS = [
    "month_sin",
    "month_cos",
    "day_of_year_sin",
    "day_of_year_cos",
    "is_commute_hour",
    "is_federal_holiday",
]
EXPERIMENTAL_FEATURE_COLUMNS = [
    *EXPERIMENTAL_FEATURE_COLUMNS,
    *LIVE_INFLIGHT_FEATURE_COLUMNS,
    *FREE_FLOATING_FEATURE_COLUMNS,
    *STATUS_QUALITY_FEATURE_COLUMNS,
]
FEATURE_COLUMNS = EXPERIMENTAL_FEATURE_COLUMNS
DISTRIBUTION_OUTPUT_COLUMNS = [
    "expected_ebikes",
    "expected_total_bikes",
    "p_count_ebikes",
    "p_count_total",
    "p_capacity_violation",
    "p_dock_constrained_arrival",
    "expected_ebike_departures",
    "expected_classic_departures",
    "expected_ebike_arrivals",
    "expected_classic_arrivals",
]
DEBUG_OUTPUT_COLUMNS = [
    "mu_e_depart",
    "mu_e_arrive",
    "mu_c_depart",
    "mu_c_arrive",
    "theta_e_depart",
    "theta_e_arrive",
    "theta_c_depart",
    "theta_c_arrive",
    "zero_inflation_e_depart",
    "zero_inflation_e_arrive",
    "zero_inflation_c_depart",
    "zero_inflation_c_arrive",
    "dock_constraint_probability",
    "stockout_probability",
    "encoder_norm",
    "graph_message_norm",
    "temporal_message_norm",
    "calibration_delta_zero",
]

_MODEL_CACHE: dict[str, object] = {}


@dataclass
class FittedAvailabilityModel:
    model: object | None
    trained_at: datetime
    n_examples: int
    n_positive: int
    n_negative: int
    method: str
    model_key: str = ACTIVE_MODEL_KEY
    label: str = MODEL_SPECS[ACTIVE_MODEL_KEY]["label"]
    model_version: str = MODEL_VERSION
    artifact_id: str | None = None
    model_warning: str | None = None

    @property
    def usable(self) -> bool:
        if self.model is None:
            return False
        if self.model_key == "dg_nissm":
            return bool(
                getattr(self.model, "trained", False)
                and getattr(self.model, "net", None) is not None
                and getattr(self.model, "tabular_scaler", None) is not None
            )
        return True


@dataclass
class FittedModelSuite:
    models: dict[str, FittedAvailabilityModel]
    active_key: str = ACTIVE_MODEL_KEY
    active_source: str = "default_cc_nissm"
    best_evaluated_model_key: str | None = None
    best_usable_model_key: str | None = None
    best_sota_model_key: str | None = None
    best_trained_sota_model_key: str | None = None
    best_baseline_model_key: str | None = None
    selection_metric: str = "decision_rank_loss"
    selection_window_hours: int = 24

    @property
    def active(self) -> FittedAvailabilityModel:
        return self.models.get(self.active_key) or next(iter(self.models.values()))

    @property
    def model_version(self) -> str:
        return self.active.model_version

    @property
    def method(self) -> str:
        return self.active.method

    @property
    def trained_at(self) -> datetime:
        return self.active.trained_at

    @property
    def n_examples(self) -> int:
        return self.active.n_examples

    @property
    def n_positive(self) -> int:
        return self.active.n_positive

    @property
    def n_negative(self) -> int:
        return self.active.n_negative

    def summary(self) -> list[dict]:
        return [
            model_meta_from_fitted(key, model) | {
                "model_key": key,
                "label": model.label,
                "version": model.model_version,
                "artifact_id": model.artifact_id,
                "method": model.method,
                "trained_at": model.trained_at.isoformat(),
                "training_examples": model.n_examples,
                "training_positive": model.n_positive,
                "training_negative": model.n_negative,
                "usable": model.usable,
                "warning": model.model_warning,
                "active": key == self.active_key,
                "sota_primary": key in SOTA_PRIMARY_MODEL_KEYS,
                "baseline": key in BASELINE_MODEL_KEYS,
            }
            for key, model in self.models.items()
        ]


def is_bootstrap_or_fallback(model_meta: dict) -> bool:
    method = str(model_meta.get("method") or "").lower()
    return (
        model_meta.get("artifact_id") is None
        or "bootstrap" in method
        or "fallback" in method
    )


def model_display_label(model_meta: dict) -> str:
    label = model_meta.get("label") or model_meta.get("model_key") or "model"
    if is_bootstrap_or_fallback(model_meta):
        return f"{label} — provisional bootstrap"
    return str(label)


def is_trained_artifact(model_meta: dict) -> bool:
    method = str(model_meta.get("method") or "").lower()
    return (
        model_meta.get("artifact_id") is not None
        and "bootstrap" not in method
        and "fallback" not in method
        and bool(model_meta.get("usable", True))
    )


def is_primary_eligible_trained_sota(model_meta: dict) -> bool:
    return (
        model_meta.get("model_key") in SOTA_PRIMARY_MODEL_KEYS
        and is_trained_artifact(model_meta)
    )


def model_meta_from_fitted(model_key: str, model: FittedAvailabilityModel) -> dict:
    meta = {
        "model_key": model_key,
        "label": model.label,
        "method": model.method,
        "artifact_id": model.artifact_id,
        "usable": model.usable,
    }
    meta["bootstrap_or_fallback"] = is_bootstrap_or_fallback(meta)
    meta["trained_artifact"] = is_trained_artifact(meta)
    meta["display_label"] = model_display_label(meta)
    return meta


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _ids_clause(ids: Iterable[str]) -> tuple[str, list[str]]:
    values = list(ids)
    if not values:
        return "", []
    return ",".join(["?"] * len(values)), values


def _clip_probability(value: float | int | None, default: float = 0.5) -> float:
    if value is None or not np.isfinite(value):
        return default
    return float(min(0.99, max(0.01, value)))


def _finite_float(value, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def cold_start_cap(current_ebikes: int, horizon_minutes: int) -> float:
    if current_ebikes <= 0:
        return {5: 0.45, 10: 0.55, 15: 0.65, 20: 0.72}.get(horizon_minutes, 0.60)
    if current_ebikes == 1:
        return {5: 0.88, 10: 0.84, 15: 0.80, 20: 0.76}.get(horizon_minutes, 0.82)
    if current_ebikes == 2:
        return {5: 0.95, 10: 0.92, 15: 0.89, 20: 0.86}.get(horizon_minutes, 0.90)
    return {5: 0.985, 10: 0.970, 15: 0.955, 20: 0.940}.get(horizon_minutes, 0.955)


def apply_cold_start_probability_guard(
    p: float,
    *,
    current_ebikes: int,
    horizon_minutes: int,
    model_meta: dict,
    n_resolved: int,
) -> float:
    value = float(np.clip(p, 0.001, 0.999))
    if not is_bootstrap_or_fallback(model_meta) and n_resolved >= 1000:
        return value
    return float(min(value, cold_start_cap(int(current_ebikes), int(horizon_minutes))))


def cap_zero_current_without_inbound_support(row: pd.Series, p: float) -> float:
    current_ebikes = int(_finite_float(row.get("num_ebikes_available"), 0.0))
    if current_ebikes > 0:
        return float(p)

    inbound_due = _finite_float(row.get("route_inbound_due_horizon"), 0.0)
    recent_net = _finite_float(row.get("trip_recent_net_arrivals_30m"), 0.0)
    neighbor_ebikes = _finite_float(row.get("station_neighbor_recent_ebikes"), 0.0)
    trend_5m = _finite_float(row.get("trend_5m"), 0.0)

    strong_support = (
        inbound_due >= 1.0
        or recent_net >= 1.0
        or neighbor_ebikes >= 2.0
        or trend_5m >= 1.0
    )
    return float(min(p, 0.80 if strong_support else 0.55))


def _align_count_distribution_to_p_zero(value, p_zero: float):
    if not isinstance(value, dict):
        return value
    target_zero = float(np.clip(_finite_float(p_zero, 0.0), 0.0, 1.0))
    positive_keys = [key for key in value.keys() if str(key) != "0"]
    positive_total = sum(max(0.0, _finite_float(value.get(key), 0.0)) for key in positive_keys)
    out = {str(key): max(0.0, _finite_float(val, 0.0)) for key, val in value.items()}
    out["0"] = target_zero
    if positive_total > 0:
        scale = (1.0 - target_zero) / positive_total
        for key in positive_keys:
            out[str(key)] = max(0.0, _finite_float(value.get(key), 0.0)) * scale
    elif target_zero < 1.0:
        out["1"] = out.get("1", 0.0) + (1.0 - target_zero)
    total = sum(out.values())
    if total > 0:
        out = {key: float(val / total) for key, val in out.items()}
    return out


def _confidence(sample_size: int, data_age_minutes: float | None = None) -> str:
    if data_age_minutes is not None and data_age_minutes > 20:
        return "low"
    if sample_size >= 50:
        return "high"
    if sample_size >= 15:
        return "medium"
    return "low"


def ranking_formula(search_radius_km: float = 1.5) -> dict:
    return {
        "score": (
            "0.60 * rank_probability_at_arrival + 0.35 * proximity_score "
            "+ 0.05 * current_count_score"
        ),
        "probability_weight": RANK_PROBABILITY_WEIGHT,
        "distance_weight": RANK_DISTANCE_WEIGHT,
        "current_count_weight": RANK_CURRENT_WEIGHT,
        "distance_reference_km": search_radius_km,
        "proximity_score": "max(0, 1 - distance_km / distance_reference_km)",
        "current_count_score": "min(current_ebikes, 3) / 3",
    }


def apply_walk_adjusted_scores(
    scored: pd.DataFrame,
    search_radius_km: float | None = None,
) -> pd.DataFrame:
    if "p_arrival" in scored.columns:
        return apply_arrival_time_scores(
            scored,
            active_model_key=str(scored.get("active_model_key", pd.Series(ACTIVE_MODEL_KEY, index=scored.index)).iloc[0]),
            search_radius_km=search_radius_km,
        )
    out = scored.copy()
    if out.empty:
        return out
    reference = search_radius_km
    if reference is None or reference <= 0:
        max_distance = out["distance_km"].dropna().max() if "distance_km" in out else np.nan
        reference = float(max(max_distance, 1.5)) if pd.notna(max_distance) else 1.5
    reference = max(float(reference), 0.1)
    probability = out["p_has_ebike_10m"].fillna(out["p_has_ebike_5m"]).fillna(0.0).clip(0.0, 1.0)
    distance = out["distance_km"].fillna(reference).clip(lower=0.0)
    proximity = (1.0 - (distance / reference)).clip(0.0, 1.0)
    current = out["num_ebikes_available"].fillna(0.0).clip(0.0, 3.0) / 3.0

    out["rank_probability"] = probability
    out["distance_reference_km"] = reference
    out["distance_score"] = proximity
    out["current_count_score"] = current
    out["walk_adjusted_score"] = (
        RANK_PROBABILITY_WEIGHT * probability
        + RANK_DISTANCE_WEIGHT * proximity
        + RANK_CURRENT_WEIGHT * current
    )
    out["rank_score"] = out["walk_adjusted_score"]
    for model_key in MODEL_KEYS:
        p10_col = f"p_has_ebike_10m_{model_key}"
        p5_col = f"p_has_ebike_5m_{model_key}"
        if p10_col not in out.columns and p5_col not in out.columns:
            continue
        model_probability = out.get(p10_col, pd.Series(index=out.index, dtype=float)).fillna(
            out.get(p5_col, pd.Series(index=out.index, dtype=float))
        ).fillna(0.0).clip(0.0, 1.0)
        out[f"rank_probability_{model_key}"] = model_probability
        out[f"walk_adjusted_score_{model_key}"] = (
            RANK_PROBABILITY_WEIGHT * model_probability
            + RANK_DISTANCE_WEIGHT * proximity
            + RANK_CURRENT_WEIGHT * current
        )
    return out


def _walking_minutes(distance_km: pd.Series, walking_speed_kmh: float, buffer_minutes: int) -> pd.Series:
    speed = max(0.5, float(walking_speed_kmh))
    return np.ceil(60.0 * distance_km.astype(float).clip(lower=0.0) / speed).astype(int) + int(buffer_minutes)


def interpolate_horizon_probability(row: pd.Series, model_key: str, target_horizon) -> float:
    try:
        target = float(target_horizon)
    except (TypeError, ValueError):
        target = 10.0
    target = float(np.clip(target, min(HORIZONS), max(HORIZONS)))
    points: list[tuple[float, float]] = []
    for horizon in HORIZONS:
        col = f"p_has_ebike_{horizon}m_{model_key}"
        value = row.get(col)
        if value is None or pd.isna(value):
            if model_key == row.get("active_model_key"):
                value = row.get(f"p_has_ebike_{horizon}m")
        if value is not None and not pd.isna(value):
            points.append((float(horizon), float(value)))
    if not points:
        value = row.get("p_has_ebike_10m")
        return _clip_probability(float(value) if value is not None and not pd.isna(value) else 0.35)
    points = sorted(points)
    xs = np.array([p[0] for p in points], dtype=float)
    ys = np.array([p[1] for p in points], dtype=float)
    return _clip_probability(float(np.interp(target, xs, ys)))


def apply_arrival_time_scores(
    scored: pd.DataFrame,
    active_model_key: str,
    search_radius_km: float | None = None,
    walking_speed_kmh: float = 4.8,
    buffer_minutes: int = 1,
) -> pd.DataFrame:
    out = scored.copy()
    if out.empty:
        return out
    reference = search_radius_km
    if reference is None or reference <= 0:
        max_distance = out["distance_km"].dropna().max() if "distance_km" in out else np.nan
        reference = float(max(max_distance, 1.5)) if pd.notna(max_distance) else 1.5
    reference = max(float(reference), 0.1)
    distance = out["distance_km"].fillna(reference).clip(lower=0.0)
    arrival_time = _walking_minutes(distance, walking_speed_kmh, buffer_minutes).clip(lower=min(HORIZONS), upper=max(HORIZONS))
    proximity = (1.0 - (distance / reference)).clip(0.0, 1.0)
    current = out["num_ebikes_available"].fillna(0.0).clip(0.0, 3.0) / 3.0
    stale_penalty = np.where(out.get("data_age_minutes", pd.Series(0.0, index=out.index)).fillna(0.0) > 20.0, 0.08, 0.0)
    closed_penalty = np.where(out.get("is_renting", pd.Series(True, index=out.index)).fillna(True).astype(bool), 0.0, 0.25)

    out["arrival_time_minutes"] = arrival_time.astype(int)
    out["target_horizon_minutes"] = arrival_time.astype(int)
    out["distance_reference_km"] = reference
    out["distance_score"] = proximity
    out["current_count_score"] = current
    out["active_model_key"] = active_model_key

    for model_key in MODEL_KEYS:
        probabilities = [
            interpolate_horizon_probability(row, model_key, row["target_horizon_minutes"])
            for _, row in out.iterrows()
        ]
        p_arrival = pd.Series(probabilities, index=out.index, dtype=float).clip(0.0, 1.0)
        lcb_col = f"reliable_probability_lcb_{model_key}"
        if lcb_col in out.columns:
            rank_probability = pd.to_numeric(out[lcb_col], errors="coerce").fillna(p_arrival).clip(0.0, 1.0)
        else:
            rank_probability = p_arrival
            out[lcb_col] = p_arrival
        out[f"p_arrival_{model_key}"] = p_arrival
        out[f"rank_probability_{model_key}"] = rank_probability
        out[f"walk_adjusted_score_{model_key}"] = (
            RANK_PROBABILITY_WEIGHT * rank_probability
            + RANK_DISTANCE_WEIGHT * proximity
            + RANK_CURRENT_WEIGHT * current
            - stale_penalty
            - closed_penalty
        ).clip(0.0, 1.0)
        ordered = out[f"walk_adjusted_score_{model_key}"].rank(method="first", ascending=False)
        out[f"recommended_rank_{model_key}"] = ordered.astype(int)

    active = active_model_key if active_model_key in MODEL_KEYS else "cc_nissm"
    out["p_arrival"] = out.get(f"p_arrival_{active}", pd.Series(0.0, index=out.index))
    out["reliable_probability_lcb"] = out.get(f"reliable_probability_lcb_{active}", out["p_arrival"])
    out["rank_probability"] = out.get(f"rank_probability_{active}", out["p_arrival"])
    out["walk_adjusted_score"] = out.get(f"walk_adjusted_score_{active}", out["rank_probability"])
    out["rank_score"] = out["walk_adjusted_score"]
    return out


def current_bucket(count: int | float | None) -> int:
    if count is None or pd.isna(count):
        return 0
    if count <= 0:
        return 0
    if count == 1:
        return 1
    if count == 2:
        return 2
    return 3


def add_temporal_features(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    out = df.copy()
    ts = pd.to_datetime(out[ts_col], utc=True)
    local = ts.dt.tz_convert(LOCAL_TZ)
    out["local_hour"] = local.dt.hour.astype(int)
    out["dow"] = local.dt.dayofweek.astype(int)
    out["is_weekend"] = out["dow"].isin([5, 6]).astype(int)
    radians = 2.0 * math.pi * out["local_hour"] / 24.0
    out["hour_sin"] = np.sin(radians)
    out["hour_cos"] = np.cos(radians)
    return out


def _feature_columns_for_model(model_key: str) -> list[str]:
    if model_key == "logistic":
        return BASELINE_FEATURE_COLUMNS
    return EXPERIMENTAL_FEATURE_COLUMNS


def add_calendar_features(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    out = df.copy()
    ts = pd.to_datetime(out[ts_col], utc=True)
    local = ts.dt.tz_convert(LOCAL_TZ)
    month = local.dt.month.astype(int)
    day_of_year = local.dt.dayofyear.astype(int)
    out["month_sin"] = np.sin(2.0 * math.pi * month / 12.0)
    out["month_cos"] = np.cos(2.0 * math.pi * month / 12.0)
    out["day_of_year_sin"] = np.sin(2.0 * math.pi * day_of_year / 366.0)
    out["day_of_year_cos"] = np.cos(2.0 * math.pi * day_of_year / 366.0)
    out["is_commute_hour"] = local.dt.hour.isin([7, 8, 9, 16, 17, 18]).astype(int)

    dates = local.dt.date
    if len(dates) and dates.notna().any():
        start = pd.Timestamp(min(d for d in dates.dropna())).tz_localize(LOCAL_TZ)
        end = pd.Timestamp(max(d for d in dates.dropna())).tz_localize(LOCAL_TZ)
        holiday_dates = set(USFederalHolidayCalendar().holidays(start=start, end=end).date)
        out["is_federal_holiday"] = dates.isin(holiday_dates).astype(int)
    else:
        out["is_federal_holiday"] = 0
    return out


def _table_has_rows(conn: duckdb.DuckDBPyConnection, table_name: str) -> bool:
    try:
        row = conn.execute(
            """
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_name = ?
            """,
            [table_name],
        ).fetchone()
        if not row or row[0] == 0:
            return False
        count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        return bool(count)
    except Exception:
        return False


def _fill_feature_defaults(df: pd.DataFrame, columns: Iterable[str], default: float = 0.0) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = default
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(default)
    return out


def _haversine_np(lat: float, lon: float, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    lat1 = np.radians(lat)
    lat2 = np.radians(lats.astype(float))
    dlat = lat2 - lat1
    dlon = np.radians(lons.astype(float) - lon)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * 6371.0 * np.arcsin(np.sqrt(a))


def _station_metadata(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    cached = _MODEL_CACHE.get("station_metadata")
    if isinstance(cached, pd.DataFrame):
        return cached.copy()
    rows = conn.execute(
        """
        SELECT station_id, capacity, lat, lon
        FROM stations
        WHERE lat IS NOT NULL AND lon IS NOT NULL
        """
    ).df()
    rows["capacity"] = pd.to_numeric(rows["capacity"], errors="coerce").fillna(0.0)
    _MODEL_CACHE["station_metadata"] = rows.copy()
    return rows


def _station_graph_edges(conn: duckdb.DuckDBPyConnection, radius_km: float = 0.5) -> pd.DataFrame:
    cache_key = f"station_graph_edges_{radius_km:.2f}"
    cached = _MODEL_CACHE.get(cache_key)
    if isinstance(cached, pd.DataFrame):
        return cached.copy()

    stations = _station_metadata(conn)
    if stations.empty:
        return pd.DataFrame(columns=["station_id", "neighbor_id", "distance_km"])

    lats = stations["lat"].to_numpy(dtype=float)
    lons = stations["lon"].to_numpy(dtype=float)
    station_ids = stations["station_id"].astype(str).to_numpy()
    rows: list[pd.DataFrame] = []
    for idx, station_id in enumerate(station_ids):
        distances = _haversine_np(float(lats[idx]), float(lons[idx]), lats, lons)
        mask = (distances > 0.0) & (distances <= radius_km)
        if not mask.any():
            continue
        rows.append(pd.DataFrame({
            "station_id": station_id,
            "neighbor_id": station_ids[mask],
            "distance_km": distances[mask],
        }))

    edges = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["station_id", "neighbor_id", "distance_km"]
    )
    _MODEL_CACHE[cache_key] = edges.copy()
    return edges


def _station_graph_static_features(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    cached = _MODEL_CACHE.get("station_graph_static_features")
    if isinstance(cached, pd.DataFrame):
        return cached.copy()

    stations = _station_metadata(conn)
    edges = _station_graph_edges(conn)
    if edges.empty:
        out = stations[["station_id"]].copy()
        out["station_neighbor_count_500m"] = 0
        out["station_neighbor_capacity_500m"] = 0.0
        _MODEL_CACHE["station_graph_static_features"] = out.copy()
        return out

    neighbor_capacity = stations[["station_id", "capacity"]].rename(
        columns={"station_id": "neighbor_id", "capacity": "neighbor_capacity"}
    )
    enriched = edges.merge(neighbor_capacity, on="neighbor_id", how="left")
    out = (
        enriched.groupby("station_id", as_index=False)
        .agg(
            station_neighbor_count_500m=("neighbor_id", "nunique"),
            station_neighbor_capacity_500m=("neighbor_capacity", "sum"),
        )
    )
    out = stations[["station_id"]].merge(out, on="station_id", how="left")
    out["station_neighbor_count_500m"] = out["station_neighbor_count_500m"].fillna(0).astype(int)
    out["station_neighbor_capacity_500m"] = out["station_neighbor_capacity_500m"].fillna(0.0)
    _MODEL_CACHE["station_graph_static_features"] = out.copy()
    return out


def _add_inventory_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in ["capacity", "num_bikes_available", "num_ebikes_available", "num_docks_available"]:
        if column not in out.columns:
            out[column] = 0.0
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)

    capacity = out["capacity"].clip(lower=1.0)
    total = out["num_bikes_available"].clip(lower=0.0)
    ebikes = out["num_ebikes_available"].clip(lower=0.0)
    docks = out["num_docks_available"].clip(lower=0.0)

    out["capacity_clipped"] = out["capacity"].clip(0, 80)
    out["current_total_bikes_clipped"] = total.clip(0, 80)
    out["docks_available_clipped"] = docks.clip(0, 80)
    out["ebike_share_of_bikes"] = np.where(total > 0, ebikes / total.clip(lower=1.0), 0.0)
    out["dock_availability_fraction"] = (docks / capacity).clip(0.0, 1.0)
    return out


def _neighbor_same_hour_rates(
    conn: duckdb.DuckDBPyConnection,
    station_hour: pd.DataFrame,
    global_hour: pd.DataFrame,
) -> pd.DataFrame:
    edges = _station_graph_edges(conn)
    if edges.empty or station_hour.empty:
        return pd.DataFrame(
            columns=["station_id", "local_hour", "dow", "station_neighbor_same_hour_rate"]
        )

    neighbor_rates = station_hour[
        ["station_id", "local_hour", "dow", "station_same_hour_rate"]
    ].rename(
        columns={
            "station_id": "neighbor_id",
            "station_same_hour_rate": "neighbor_rate",
        }
    )
    joined = edges.merge(neighbor_rates, on="neighbor_id", how="inner")
    if joined.empty:
        return pd.DataFrame(
            columns=["station_id", "local_hour", "dow", "station_neighbor_same_hour_rate"]
        )
    out = (
        joined.groupby(["station_id", "local_hour", "dow"], as_index=False)["neighbor_rate"]
        .mean()
        .rename(columns={"neighbor_rate": "station_neighbor_same_hour_rate"})
    )
    return out.merge(global_hour, on=["local_hour", "dow"], how="left")


def _live_neighbor_features(
    conn: duckdb.DuckDBPyConnection,
    station_ids: list[str],
    now: datetime,
    freshness_minutes: int = 30,
) -> pd.DataFrame:
    if not station_ids:
        return pd.DataFrame()
    edges = _station_graph_edges(conn)
    if edges.empty:
        return pd.DataFrame({
            "station_id": station_ids,
            "station_neighbor_recent_ebikes": 0.0,
            "station_neighbor_recent_zero_rate": 1.0,
        })

    rows = conn.execute(
        """
        WITH latest AS (
          SELECT station_id, num_ebikes_available, last_reported
          FROM (
            SELECT *, ROW_NUMBER() OVER (
              PARTITION BY station_id ORDER BY last_reported DESC
            ) AS rn
            FROM station_status
            WHERE last_reported >= ? - (? * INTERVAL '1 minute')
          )
          WHERE rn = 1
        )
        SELECT station_id, num_ebikes_available
        FROM latest
        """,
        [now, freshness_minutes],
    ).df()
    if rows.empty:
        return pd.DataFrame({
            "station_id": station_ids,
            "station_neighbor_recent_ebikes": 0.0,
            "station_neighbor_recent_zero_rate": 1.0,
        })

    neighbor_state = rows.rename(columns={"station_id": "neighbor_id"})
    joined = edges[edges["station_id"].isin(station_ids)].merge(neighbor_state, on="neighbor_id", how="left")
    if joined.empty:
        return pd.DataFrame({
            "station_id": station_ids,
            "station_neighbor_recent_ebikes": 0.0,
            "station_neighbor_recent_zero_rate": 1.0,
        })
    joined["num_ebikes_available"] = pd.to_numeric(joined["num_ebikes_available"], errors="coerce")
    out = (
        joined.groupby("station_id", as_index=False)
        .agg(
            station_neighbor_recent_ebikes=("num_ebikes_available", "mean"),
            station_neighbor_recent_zero_rate=(
                "num_ebikes_available",
                lambda s: float((s.fillna(0) <= 0).mean()),
            ),
        )
    )
    out = pd.DataFrame({"station_id": station_ids}).merge(out, on="station_id", how="left")
    out["station_neighbor_recent_ebikes"] = out["station_neighbor_recent_ebikes"].fillna(0.0)
    out["station_neighbor_recent_zero_rate"] = out["station_neighbor_recent_zero_rate"].fillna(1.0)
    return out


def _live_inflight_features(
    conn: duckdb.DuckDBPyConnection,
    station_ids: list[str],
    now: datetime,
) -> pd.DataFrame:
    try:
        from .live_inflight import get_live_inflight_features

        return get_live_inflight_features(conn, station_ids, now, HORIZONS)
    except Exception:
        out = pd.DataFrame({"station_id": station_ids})
        for column in LIVE_INFLIGHT_FEATURE_COLUMNS:
            out[column] = 0.0
        return out


def _free_floating_density_features(
    conn: duckdb.DuckDBPyConnection,
    stations: pd.DataFrame,
    now: datetime,
    freshness_minutes: int = 10,
) -> pd.DataFrame:
    station_ids = stations["station_id"].astype(str).tolist() if "station_id" in stations else []
    out = pd.DataFrame({"station_id": station_ids})
    for column in FREE_FLOATING_FEATURE_COLUMNS:
        out[column] = 0.0
    if not station_ids or "lat" not in stations or "lon" not in stations:
        return out
    try:
        free = conn.execute(
            """
            SELECT lat, lon
            FROM (
              SELECT *, ROW_NUMBER() OVER (
                PARTITION BY bike_id ORDER BY fetched_at DESC
              ) AS rn
              FROM free_bike_status
              WHERE fetched_at >= ? - (? * INTERVAL '1 minute')
                AND COALESCE(is_reserved, false) = false
                AND COALESCE(is_disabled, false) = false
                AND lat IS NOT NULL
                AND lon IS NOT NULL
            )
            WHERE rn = 1
            """,
            [now, int(freshness_minutes)],
        ).df()
    except Exception:
        return out
    if free.empty:
        return out
    free_lats = free["lat"].to_numpy(dtype=float)
    free_lons = free["lon"].to_numpy(dtype=float)
    rows = []
    for row in stations[["station_id", "lat", "lon"]].itertuples(index=False):
        if pd.isna(row.lat) or pd.isna(row.lon):
            rows.append((str(row.station_id), 0.0, 0.0, 0.0))
            continue
        distances = _haversine_np(float(row.lat), float(row.lon), free_lats, free_lons)
        rows.append((
            str(row.station_id),
            float((distances <= 0.3).sum() / (math.pi * 0.3 * 0.3)),
            float((distances <= 0.5).sum() / (math.pi * 0.5 * 0.5)),
            float((distances <= 1.0).sum() / math.pi),
        ))
    return pd.DataFrame(
        rows,
        columns=[
            "station_id",
            "free_floating_density_300m",
            "free_floating_density_500m",
            "free_floating_density_1000m",
        ],
    )


def _trip_alias_values(conn: duckdb.DuckDBPyConnection, station_ids: list[str]) -> tuple[str, list[str]]:
    if not station_ids:
        return "", []
    unique_ids = sorted(set(str(station_id) for station_id in station_ids if station_id))
    aliases = {(station_id, station_id) for station_id in unique_ids}
    placeholders, params = _ids_clause(unique_ids)
    rows = conn.execute(
        f"""
        SELECT station_id, legacy_id, short_name
        FROM stations
        WHERE station_id IN ({placeholders})
        """,
        params,
    ).fetchall()
    for station_id, legacy_id, short_name in rows:
        for trip_id in [station_id, legacy_id, short_name]:
            if trip_id:
                aliases.add((str(trip_id), str(station_id)))
    values = ",".join(["(?, ?)"] * len(aliases))
    flat_params: list[str] = []
    for trip_id, station_id in sorted(aliases):
        flat_params.extend([trip_id, station_id])
    return values, flat_params


def _trip_hourly_profile(conn: duckdb.DuckDBPyConnection, station_ids: list[str]) -> pd.DataFrame:
    if not station_ids or not _table_has_rows(conn, "station_trip_flows"):
        return pd.DataFrame()
    alias_values, params = _trip_alias_values(conn, station_ids)
    if not alias_values:
        return pd.DataFrame()
    return conn.execute(
        f"""
        WITH alias(trip_station_id, station_id) AS (
          VALUES {alias_values}
        ),
        hourly AS (
          SELECT
            a.station_id,
            DATE_TRUNC('hour', f.bucket_start) AS hour_start,
            CAST(EXTRACT(HOUR FROM (
              CAST(f.bucket_start AS TIMESTAMP) AT TIME ZONE 'UTC' AT TIME ZONE '{LOCAL_TZ}'
            )) AS INTEGER) AS local_hour,
            CAST((
              CAST(EXTRACT(DOW FROM (
                CAST(f.bucket_start AS TIMESTAMP) AT TIME ZONE 'UTC' AT TIME ZONE '{LOCAL_TZ}'
              )) AS INTEGER) + 6
            ) % 7 AS INTEGER) AS dow,
            SUM(f.departures) AS departures,
            SUM(f.arrivals) AS arrivals,
            SUM(f.ebike_arrivals) AS ebike_arrivals
          FROM station_trip_flows f
          JOIN alias a ON f.station_id = a.trip_station_id
          GROUP BY a.station_id, hour_start, local_hour, dow
        )
        SELECT
          station_id,
          local_hour,
          dow,
          AVG(departures) / 6.0 AS trip_departures_same_hour_10m,
          AVG(arrivals) / 6.0 AS trip_arrivals_same_hour_10m,
          (AVG(arrivals) - AVG(departures)) / 6.0 AS trip_net_arrivals_same_hour_10m,
          CASE WHEN SUM(arrivals) > 0
            THEN SUM(ebike_arrivals) / SUM(arrivals)
            ELSE 0.0
          END AS trip_ebike_arrival_share_same_hour
        FROM hourly
        GROUP BY station_id, local_hour, dow
        """,
        params,
    ).df()


def _trip_route_profile(conn: duckdb.DuckDBPyConnection, station_ids: list[str]) -> pd.DataFrame:
    if not station_ids or not _table_has_rows(conn, "station_trip_routes"):
        return pd.DataFrame()
    alias_values, params = _trip_alias_values(conn, station_ids)
    if not alias_values:
        return pd.DataFrame()
    return conn.execute(
        f"""
        WITH alias(trip_station_id, station_id) AS (
          VALUES {alias_values}
        )
        SELECT
          a.station_id,
          r.local_hour,
          r.dow,
          SUM(r.trips) AS route_inbound_trips_same_hour,
          CASE WHEN SUM(r.trips) > 0 THEN SUM(r.ebike_trips) / SUM(r.trips) ELSE 0.0 END
            AS route_inbound_ebike_share_same_hour,
          AVG(r.median_duration_minutes) AS route_inbound_median_duration_minutes
        FROM station_trip_routes r
        JOIN alias a ON r.end_station_id = a.trip_station_id
        GROUP BY a.station_id, r.local_hour, r.dow
        """,
        params,
    ).df()


def _trip_recent_features(
    conn: duckdb.DuckDBPyConnection,
    station_ids: list[str],
    now: datetime,
    minutes: int = 30,
) -> pd.DataFrame:
    base = pd.DataFrame({"station_id": station_ids})
    if not station_ids or not _table_has_rows(conn, "station_trip_flows"):
        return _fill_feature_defaults(
            base,
            ["trip_recent_departures_30m", "trip_recent_arrivals_30m", "trip_recent_net_arrivals_30m"],
        )
    alias_values, params = _trip_alias_values(conn, station_ids)
    if not alias_values:
        return _fill_feature_defaults(
            base,
            ["trip_recent_departures_30m", "trip_recent_arrivals_30m", "trip_recent_net_arrivals_30m"],
        )
    rows = conn.execute(
        f"""
        WITH alias(trip_station_id, station_id) AS (
          VALUES {alias_values}
        )
        SELECT
          a.station_id,
          SUM(f.departures) AS trip_recent_departures_30m,
          SUM(f.arrivals) AS trip_recent_arrivals_30m,
          SUM(f.arrivals) - SUM(f.departures) AS trip_recent_net_arrivals_30m
        FROM station_trip_flows f
        JOIN alias a ON f.station_id = a.trip_station_id
        WHERE f.bucket_start >= ? - (? * INTERVAL '1 minute')
          AND f.bucket_start <= ?
        GROUP BY a.station_id
        """,
        [*params, now, minutes, now],
    ).df()
    out = base.merge(rows, on="station_id", how="left")
    return _fill_feature_defaults(
        out,
        ["trip_recent_departures_30m", "trip_recent_arrivals_30m", "trip_recent_net_arrivals_30m"],
    )


def _add_trip_features(
    conn: duckdb.DuckDBPyConnection,
    df: pd.DataFrame,
    now: datetime | None = None,
) -> pd.DataFrame:
    if df.empty:
        return _fill_feature_defaults(df, TRIP_FEATURE_COLUMNS)
    out = df.copy()
    station_ids = out["station_id"].astype(str).dropna().unique().tolist()
    if "local_hour" not in out.columns or "dow" not in out.columns:
        ts_col = "forecasted_at" if "forecasted_at" in out.columns else "last_reported"
        out = add_temporal_features(out, ts_col)

    hourly = _trip_hourly_profile(conn, station_ids)
    if not hourly.empty:
        out = out.merge(hourly, on=["station_id", "local_hour", "dow"], how="left")
    routes = _trip_route_profile(conn, station_ids)
    if not routes.empty:
        out = out.merge(routes, on=["station_id", "local_hour", "dow"], how="left")
    if now is not None:
        recent = _trip_recent_features(conn, station_ids, now)
        out = out.merge(recent, on="station_id", how="left")

    out = _fill_feature_defaults(out, TRIP_FEATURE_COLUMNS)
    duration = out["route_inbound_median_duration_minutes"].replace(0.0, np.nan)
    horizon = out["horizon_minutes"] if "horizon_minutes" in out.columns else 10.0
    horizon = pd.to_numeric(horizon, errors="coerce").fillna(10.0)
    route_volume = out["route_inbound_trips_same_hour"].clip(lower=0.0)
    duration_distance = (duration - horizon).abs().fillna(60.0)
    out["route_inbound_due_horizon"] = route_volume * np.exp(-duration_distance / 10.0)
    out = _fill_feature_defaults(out, TRIP_FEATURE_COLUMNS)
    return out


def _weather_defaults(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    defaults = {
        "weather_temperature_2m": 12.0,
        "weather_relative_humidity_2m": 60.0,
        "weather_apparent_temperature": 12.0,
        "weather_precipitation": 0.0,
        "weather_rain": 0.0,
        "weather_snowfall": 0.0,
        "weather_snow_depth": 0.0,
        "weather_cloud_cover": 50.0,
        "weather_wind_speed_10m": 12.0,
        "weather_wind_gusts_10m": 20.0,
        "weather_bad_conditions": 0.0,
    }
    for column, default in defaults.items():
        if column not in out.columns:
            out[column] = default
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(default)
    return out


def _add_weather_features(
    conn: duckdb.DuckDBPyConnection,
    df: pd.DataFrame,
    ts_col: str,
) -> pd.DataFrame:
    if df.empty:
        return _weather_defaults(df)
    out = df.copy()
    out["_weather_hour"] = (
        pd.to_datetime(out[ts_col], utc=True)
        .dt.floor("h")
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
    )
    if _table_has_rows(conn, "weather_hourly"):
        min_ts = out["_weather_hour"].min() - pd.Timedelta(hours=1)
        max_ts = out["_weather_hour"].max() + pd.Timedelta(hours=1)
        weather = conn.execute(
            """
            SELECT
              observed_at AS _weather_hour,
              temperature_2m AS weather_temperature_2m,
              relative_humidity_2m AS weather_relative_humidity_2m,
              apparent_temperature AS weather_apparent_temperature,
              precipitation AS weather_precipitation,
              rain AS weather_rain,
              snowfall AS weather_snowfall,
              snow_depth AS weather_snow_depth,
              cloud_cover AS weather_cloud_cover,
              wind_speed_10m AS weather_wind_speed_10m,
              wind_gusts_10m AS weather_wind_gusts_10m
            FROM weather_hourly
            WHERE observed_at BETWEEN ? AND ?
            """,
            [min_ts.to_pydatetime(), max_ts.to_pydatetime()],
        ).df()
        if not weather.empty:
            weather["_weather_hour"] = pd.to_datetime(weather["_weather_hour"])
            out["_weather_row_id"] = np.arange(len(out))
            out = pd.merge_asof(
                out.sort_values("_weather_hour"),
                weather.sort_values("_weather_hour"),
                on="_weather_hour",
                direction="nearest",
                tolerance=pd.Timedelta(hours=1),
            ).sort_values("_weather_row_id")
            out = out.drop(columns=["_weather_row_id"])
    out = out.drop(columns=["_weather_hour"])
    out = _weather_defaults(out)
    out["weather_bad_conditions"] = (
        (out["weather_precipitation"] > 0.25)
        | (out["weather_snowfall"] > 0.0)
        | (out["weather_snow_depth"] > 0.0)
        | (out["weather_wind_gusts_10m"] >= 35.0)
        | (out["weather_apparent_temperature"] <= -5.0)
    ).astype(int)
    return out


def reconstruct_minute_series(status_df: pd.DataFrame, freq: str = "1min") -> pd.DataFrame:
    """Return piecewise-constant minute availability per station.

    This is primarily used by tests and backtesting diagnostics. Runtime scoring
    uses as-of joins to avoid materializing large multi-month minute grids.
    """
    if status_df.empty:
        return pd.DataFrame(columns=["station_id", "ts", "num_ebikes_available"])

    frames: list[pd.DataFrame] = []
    for station_id, group in status_df.sort_values("last_reported").groupby("station_id"):
        g = group[["last_reported", "num_ebikes_available"]].dropna().copy()
        if g.empty:
            continue
        g["last_reported"] = pd.to_datetime(g["last_reported"])
        g = g.drop_duplicates("last_reported", keep="last").set_index("last_reported")
        idx = pd.date_range(g.index.min().floor(freq), g.index.max().ceil(freq), freq=freq)
        minute = g.reindex(g.index.union(idx)).sort_index().ffill().reindex(idx)
        minute = minute.rename_axis("ts").reset_index()
        minute["station_id"] = station_id
        frames.append(minute[["station_id", "ts", "num_ebikes_available"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def station_candidates(
    conn: duckdb.DuckDBPyConnection,
    lat: float,
    lon: float,
    radius_km: float = 1.5,
    limit: int = 120,
) -> pd.DataFrame:
    return conn.execute(
        """
        WITH latest AS (
          SELECT station_id, num_bikes_available, num_ebikes_available,
                 num_docks_available, last_reported, is_renting
          FROM (
            SELECT *, ROW_NUMBER() OVER (
              PARTITION BY station_id ORDER BY last_reported DESC
            ) AS rn
            FROM station_status
          )
          WHERE rn = 1
        ),
        joined AS (
          SELECT
            s.station_id,
            s.name,
            s.short_name,
            s.capacity,
            s.lat,
            s.lon,
            l.num_bikes_available,
            l.num_ebikes_available,
            l.num_docks_available,
            l.last_reported,
            l.is_renting,
            6371.0 * 2.0 * ASIN(
              SQRT(
                POWER(SIN(RADIANS(s.lat - ?) / 2.0), 2)
                + COS(RADIANS(?)) * COS(RADIANS(s.lat))
                  * POWER(SIN(RADIANS(s.lon - ?) / 2.0), 2)
              )
            ) AS distance_km
          FROM stations s
          JOIN latest l USING (station_id)
          WHERE s.lat IS NOT NULL AND s.lon IS NOT NULL
        )
        SELECT *
        FROM joined
        WHERE distance_km <= ?
        ORDER BY distance_km
        LIMIT ?
        """,
        [lat, lat, lon, radius_km, limit],
    ).df()


def _fetch_status_rows(
    conn: duckdb.DuckDBPyConnection,
    station_ids: list[str] | None = None,
    history_hours: int = 24 * 14,
    max_rows: int = 200_000,
) -> pd.DataFrame:
    params: list[object] = [history_hours]
    station_filter = ""
    if station_ids:
        placeholders, ids = _ids_clause(station_ids)
        station_filter = f"AND station_id IN ({placeholders})"
        params.extend(ids)
    params.append(max_rows)
    return conn.execute(
        f"""
        SELECT station_id, last_reported, num_bikes_available, num_ebikes_available,
               num_docks_available, is_renting
        FROM station_status
        WHERE last_reported >= now() - (? * INTERVAL '1 hour')
          AND num_ebikes_available IS NOT NULL
          {station_filter}
        ORDER BY last_reported DESC
        LIMIT ?
        """,
        params,
    ).df()


def _add_trend_features(status: pd.DataFrame) -> pd.DataFrame:
    if status.empty:
        return status
    frames: list[pd.DataFrame] = []
    for _, group in status.sort_values("last_reported").groupby("station_id"):
        g = group.copy().reset_index(drop=True)
        past = g[["last_reported", "num_ebikes_available"]].rename(
            columns={
                "last_reported": "past_reported",
                "num_ebikes_available": "past_ebikes",
            }
        )
        for minutes in (5, 10, 15):
            targets = pd.DataFrame({
                "lookup_ts": g["last_reported"] - pd.Timedelta(minutes=minutes),
                "row_id": g.index,
            }).sort_values("lookup_ts")
            matched = pd.merge_asof(
                targets,
                past.sort_values("past_reported"),
                left_on="lookup_ts",
                right_on="past_reported",
                direction="backward",
                tolerance=pd.Timedelta(minutes=10),
            ).set_index("row_id")
            g[f"trend_{minutes}m"] = (
                g["num_ebikes_available"] - matched["past_ebikes"].reindex(g.index)
            ).fillna(0.0)
        g["churn_rate"] = (
            g["num_ebikes_available"].diff().abs().rolling(5, min_periods=1).mean()
        ).fillna(0.0)
        frames.append(g)
    return pd.concat(frames, ignore_index=True)


def build_training_examples(
    conn: duckdb.DuckDBPyConnection,
    station_ids: list[str] | None = None,
    horizons: tuple[int, ...] = HORIZONS,
    history_hours: int = 24 * 14,
    tolerance_minutes: int = 3,
    max_source_rows: int = 200_000,
) -> pd.DataFrame:
    del tolerance_minutes
    from . import label_builder

    return label_builder.build_leak_free_examples(
        conn,
        station_ids=station_ids,
        horizons=horizons,
        history_hours=history_hours,
        max_source_rows=max_source_rows,
    )


def _empty_model(model_key: str, n: int, positives: int, negatives: int) -> FittedAvailabilityModel:
    spec = MODEL_SPECS[model_key]
    return FittedAvailabilityModel(
        model=None,
        trained_at=_utc_now(),
        n_examples=n,
        n_positive=positives,
        n_negative=negatives,
        method="empirical_only",
        model_key=model_key,
        label=spec["label"],
        model_version=spec["version"],
    )


def _series(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(default).astype(float)


def _sigmoid(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(arr, -35.0, 35.0)))


class InventoryWorldRolloutModel:
    """Distributional constrained rollout over station inventory.

    This model treats trip history as demand/supply pressure and explicitly
    rolls docked eBike and total-bike inventory under capacity and dock
    constraints. It returns a compact count distribution for downstream
    logging/evaluation while keeping sklearn-style ``predict_proba`` support.
    """

    def _flow_means(self, rows: pd.DataFrame) -> pd.DataFrame:
        horizon_scale = (_series(rows, "horizon_minutes", 10.0).clip(1.0, 30.0) / 10.0)
        current_ebikes = _series(rows, "num_ebikes_available", np.nan).fillna(
            _series(rows, "current_ebikes_clipped")
        ).clip(0.0, 80.0)
        total_bikes = _series(rows, "num_bikes_available", np.nan).fillna(
            _series(rows, "current_total_bikes_clipped")
        ).clip(lower=current_ebikes)
        ebike_share = _series(rows, "ebike_share_of_bikes").clip(0.0, 1.0)
        ebike_share = np.maximum(ebike_share, _series(rows, "trip_ebike_arrival_share_same_hour").clip(0.0, 1.0) * 0.5)
        ebike_share = np.maximum(ebike_share, np.where(current_ebikes > 0, current_ebikes / total_bikes.clip(lower=1.0), 0.0))

        weather_penalty = (
            1.0
            - 0.18 * _series(rows, "weather_bad_conditions").clip(0.0, 1.0)
            - 0.02 * _series(rows, "weather_precipitation").clip(0.0, 4.0)
            - 0.01 * (_series(rows, "weather_wind_gusts_10m", 20.0).clip(lower=20.0) - 20.0) / 10.0
        ).clip(0.55, 1.05)
        commute_boost = 1.0 + 0.08 * _series(rows, "is_commute_hour").clip(0.0, 1.0)

        departure_total = (
            _series(rows, "trip_departures_same_hour_10m")
            + _series(rows, "trip_recent_departures_30m") / 3.0
        ).clip(lower=0.0) * horizon_scale * weather_penalty * commute_boost
        arrival_total = (
            _series(rows, "trip_arrivals_same_hour_10m")
            + _series(rows, "trip_recent_arrivals_30m") / 3.0
            + _series(rows, "route_inbound_due_horizon")
        ).clip(lower=0.0) * horizon_scale * weather_penalty
        neighbor_support = _series(rows, "station_neighbor_recent_ebikes") * 0.08 * horizon_scale
        trend_support = (
            0.45 * _series(rows, "trend_5m")
            + 0.30 * _series(rows, "trend_10m")
            + 0.15 * _series(rows, "trend_15m")
        ) * horizon_scale

        history_p = _series(rows, "station_same_hour_rate", 0.35).clip(0.01, 0.99)
        neighbor_p = _series(rows, "station_neighbor_same_hour_rate", 0.35).clip(0.01, 0.99)
        appearance_prior = np.where(
            current_ebikes <= 0,
            -np.log1p(-(0.16 * history_p + 0.08 * neighbor_p).clip(0.0, 0.65)),
            0.0,
        )

        arrival_ebike_share = np.maximum(
            0.20,
            _series(rows, "trip_ebike_arrival_share_same_hour").clip(0.0, 1.0),
        )
        positive_trend = np.maximum(trend_support, 0.0)
        negative_trend = np.maximum(-trend_support, 0.0)

        ebike_departures = departure_total * np.maximum(0.12, ebike_share) + negative_trend
        classic_departures = departure_total * np.maximum(0.05, 1.0 - ebike_share)
        ebike_arrivals = arrival_total * arrival_ebike_share + neighbor_support + positive_trend + appearance_prior
        classic_arrivals = arrival_total * (1.0 - arrival_ebike_share)

        return pd.DataFrame({
            "ebike_departure_mean": np.maximum(0.0, ebike_departures),
            "classic_departure_mean": np.maximum(0.0, classic_departures),
            "ebike_arrival_mean": np.maximum(0.0, ebike_arrivals),
            "classic_arrival_mean": np.maximum(0.0, classic_arrivals),
        }, index=rows.index)

    def predict_distribution(self, rows: pd.DataFrame) -> pd.DataFrame:
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
                "p_has_ebike": float(min(1.0, max(0.0, rollout.p_has_ebike))),
                "p_zero": float(min(1.0, max(0.0, rollout.p_zero))),
                "expected_ebikes": rollout.expected_ebikes,
                "expected_total_bikes": rollout.expected_total_bikes,
                "p_count_ebikes": rollout.p_count_ebikes,
                "p_count_total": rollout.p_count_total,
                "p_capacity_violation": rollout.p_capacity_violation,
                "p_dock_constrained_arrival": rollout.p_dock_constrained_arrival,
                "expected_ebike_departures": rollout.expected_ebike_departures,
                "expected_classic_departures": rollout.expected_classic_departures,
                "expected_ebike_arrivals": rollout.expected_ebike_arrivals,
                "expected_classic_arrivals": rollout.expected_classic_arrivals,
            })
        return pd.DataFrame(results, index=rows.index)

    def predict_proba(self, rows: pd.DataFrame) -> np.ndarray:
        dist = self.predict_distribution(rows)
        p = dist["p_has_ebike"].to_numpy(dtype=float)
        history_p = _series(rows, "station_same_hour_rate", 0.35).clip(0.01, 0.99)
        neighbor_p = _series(rows, "station_neighbor_same_hour_rate", 0.35).clip(0.01, 0.99)
        p = 0.92 * p + 0.05 * history_p + 0.03 * neighbor_p
        return np.column_stack([1.0 - np.clip(p, 0.01, 0.99), np.clip(p, 0.01, 0.99)])


if _torch_nn is not None:

    class TorchCDEFunc(_torch_nn.Module):
        def __init__(self, hidden_channels: int, input_channels: int) -> None:
            super().__init__()
            self.hidden_channels = hidden_channels
            self.input_channels = input_channels
            self.net = _torch_nn.Sequential(
                _torch_nn.Linear(hidden_channels, 64),
                _torch_nn.Tanh(),
                _torch_nn.Linear(64, hidden_channels * input_channels),
            )

        def forward(self, t, z):  # type: ignore[no-untyped-def]
            del t
            out = self.net(z)
            return out.view(z.size(0), self.hidden_channels, self.input_channels)


    class TorchCDEClassifier(_torch_nn.Module):
        def __init__(self, input_channels: int, hidden_channels: int = 32) -> None:
            super().__init__()
            self.initial = _torch_nn.Linear(input_channels, hidden_channels)
            self.func = TorchCDEFunc(hidden_channels, input_channels)
            self.readout = _torch_nn.Linear(hidden_channels, 1)

        def forward(self, coeffs):  # type: ignore[no-untyped-def]
            import torch
            import torchcde

            control = torchcde.LinearInterpolation(coeffs)
            t = torch.tensor(
                [control.interval[0], control.interval[1]],
                device=coeffs.device,
                dtype=coeffs.dtype,
            )
            z0 = self.initial(control.evaluate(control.interval[0]))
            zt = torchcde.cdeint(X=control, z0=z0, func=self.func, t=t, method="rk4")
            return self.readout(zt[:, -1]).squeeze(-1)

else:
    TorchCDEClassifier = None  # type: ignore[assignment]


class GraphNCDEAvailabilitySurrogate:
    """STG-NCDE-style graph temporal model with a deterministic fallback.

    On larger datasets this trains a small torchcde neural controlled
    differential equation over temporal, spatial, trip-flow, weather, and
    calendar controls. For tiny fixtures or missing torch dependencies it keeps
    the explicit Euler state evolution plus logistic calibration fallback.
    """

    _CONTROL_COLUMNS = [
        "ncde_z0",
        "ncde_z1",
        "ncde_temporal",
        "ncde_spatial",
        "ncde_semantic",
        "ncde_flow",
        "ncde_weather",
        "ncde_calendar",
        "ncde_history",
        "ncde_neighbor_history",
        "ncde_horizon",
    ]

    def __init__(self) -> None:
        self.calibrator: object | None = None
        self.torch_model: object | None = None
        self.torch_mean: np.ndarray | None = None
        self.torch_scale: np.ndarray | None = None
        self.method = "stg_ncde_euler_surrogate"
        self.version = "stg-ncde-euler-surrogate-v1"
        self.torch_training_error: str | None = None
        self.torch_device: str | None = None

    def _latent_features(self, rows: pd.DataFrame) -> pd.DataFrame:
        horizon_scale = (_series(rows, "horizon_minutes", 10.0).clip(1.0, 30.0) / 10.0)
        current = _series(rows, "current_ebikes_clipped")
        temporal_control = (
            0.55 * _series(rows, "trend_5m")
            + 0.30 * _series(rows, "trend_10m")
            + 0.15 * _series(rows, "trend_15m")
        )
        spatial_control = (
            _series(rows, "station_neighbor_recent_ebikes")
            - current * (1.0 - _series(rows, "station_neighbor_recent_zero_rate", 1.0))
        )
        semantic_control = (
            _series(rows, "station_neighbor_same_hour_rate", 0.35)
            - _series(rows, "station_same_hour_rate", 0.35)
        )
        flow_control = (
            _series(rows, "trip_arrivals_same_hour_10m")
            - _series(rows, "trip_departures_same_hour_10m")
            + 0.5 * _series(rows, "route_inbound_due_horizon")
        )
        weather_control = (
            -0.8 * _series(rows, "weather_bad_conditions")
            -0.08 * _series(rows, "weather_precipitation")
            -0.02 * (_series(rows, "weather_wind_gusts_10m", 20.0) - 20.0)
        )
        calendar_control = (
            0.15 * _series(rows, "is_commute_hour")
            -0.10 * _series(rows, "is_federal_holiday")
            +0.08 * _series(rows, "is_weekend")
        )
        z0 = np.log1p(current)
        z1 = z0 + horizon_scale * (
            0.35 * temporal_control
            + 0.25 * spatial_control
            + 0.30 * flow_control
            + 0.18 * semantic_control
            + weather_control
            + calendar_control
        )
        return pd.DataFrame({
            "ncde_z0": z0,
            "ncde_z1": z1,
            "ncde_temporal": temporal_control,
            "ncde_spatial": spatial_control,
            "ncde_semantic": semantic_control,
            "ncde_flow": flow_control,
            "ncde_weather": weather_control,
            "ncde_calendar": calendar_control,
            "ncde_history": _series(rows, "station_same_hour_rate", 0.35),
            "ncde_neighbor_history": _series(rows, "station_neighbor_same_hour_rate", 0.35),
            "ncde_horizon": horizon_scale,
        }, index=rows.index).fillna(0.0)

    def _raw_probability(self, rows: pd.DataFrame) -> np.ndarray:
        latent = self._latent_features(rows)
        logits = (
            -0.35
            + 0.75 * latent["ncde_z1"]
            + 0.70 * latent["ncde_history"]
            + 0.35 * latent["ncde_neighbor_history"]
            + 0.12 * latent["ncde_flow"]
            + 0.08 * latent["ncde_spatial"]
            - 0.08 * latent["ncde_horizon"]
        )
        return np.clip(_sigmoid(logits), 0.01, 0.99)

    def _control_path(self, rows: pd.DataFrame, fit: bool = False) -> np.ndarray:
        latent = self._latent_features(rows)[self._CONTROL_COLUMNS].astype(float)
        start = latent.copy()
        start["ncde_z1"] = latent["ncde_z0"]
        for column in [
            "ncde_temporal",
            "ncde_spatial",
            "ncde_semantic",
            "ncde_flow",
            "ncde_weather",
            "ncde_calendar",
        ]:
            start[column] = 0.0
        middle = start + 0.5 * (latent - start)
        path = np.stack(
            [
                start.to_numpy(dtype=np.float32),
                middle.to_numpy(dtype=np.float32),
                latent.to_numpy(dtype=np.float32),
            ],
            axis=1,
        )
        if fit:
            flat = path.reshape(-1, path.shape[-1])
            self.torch_mean = np.nanmean(flat, axis=0).astype(np.float32)
            self.torch_scale = np.nanstd(flat, axis=0).astype(np.float32)
            self.torch_scale = np.where(self.torch_scale < 1e-3, 1.0, self.torch_scale)
        mean = self.torch_mean if self.torch_mean is not None else np.zeros(path.shape[-1], dtype=np.float32)
        scale = self.torch_scale if self.torch_scale is not None else np.ones(path.shape[-1], dtype=np.float32)
        return np.nan_to_num((path - mean) / scale, nan=0.0, posinf=6.0, neginf=-6.0).astype(np.float32)

    def _fit_torch_cde(self, examples: pd.DataFrame) -> None:
        if os.getenv("DIVVY_DISABLE_TORCH_STG_NCDE") == "1":
            return
        if len(examples) < STG_NCDE_MIN_EXAMPLES or examples["has_ebike"].nunique() < 2:
            return
        try:
            import torch
            import torch.nn as nn
            import torchcde
            from torch.utils.data import DataLoader, TensorDataset
        except Exception as exc:
            self.torch_training_error = f"torch import failed: {exc}"
            return

        def candidate_devices() -> list[str]:
            if STG_NCDE_DEVICE == "cpu":
                return ["cpu"]
            if STG_NCDE_DEVICE == "mps":
                return ["mps", "cpu"] if torch.backends.mps.is_available() else ["cpu"]
            if STG_NCDE_DEVICE == "cuda":
                return ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
            devices: list[str] = []
            if torch.backends.mps.is_available():
                devices.append("mps")
            if torch.cuda.is_available():
                devices.append("cuda")
            devices.append("cpu")
            return devices

        train = examples
        if len(train) > STG_NCDE_MAX_EXAMPLES:
            train = train.sample(n=STG_NCDE_MAX_EXAMPLES, random_state=42)
        train = train.reset_index(drop=True)
        controls = self._control_path(train, fit=True)
        y = train["has_ebike"].astype(float).to_numpy(dtype=np.float32)
        if not np.isfinite(controls).all() or len(np.unique(y)) < 2:
            return

        for device_name in candidate_devices():
            try:
                if TorchCDEClassifier is None:
                    self.torch_training_error = "torch module class unavailable"
                    return
                torch.manual_seed(42)
                device = torch.device(device_name)
                x_tensor = torch.tensor(controls, dtype=torch.float32, device=device)
                y_tensor = torch.tensor(y, dtype=torch.float32, device=device)
                coeffs = torchcde.linear_interpolation_coeffs(x_tensor)
                dataset = TensorDataset(coeffs, y_tensor)
                loader = DataLoader(
                    dataset,
                    batch_size=max(64, STG_NCDE_BATCH_SIZE),
                    shuffle=True,
                )
                model = TorchCDEClassifier(input_channels=controls.shape[-1]).to(device)
                positives = float(y_tensor.sum().item())
                negatives = float(len(y_tensor) - positives)
                pos_weight = torch.tensor(
                    [max(0.25, min(4.0, negatives / max(1.0, positives)))],
                    device=device,
                )
                loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
                model.train()
                for _ in range(max(1, STG_NCDE_EPOCHS)):
                    for batch_coeffs, batch_y in loader:
                        batch_coeffs = batch_coeffs.to(device)
                        batch_y = batch_y.to(device)
                        optimizer.zero_grad(set_to_none=True)
                        logits = model(batch_coeffs)
                        loss = loss_fn(logits, batch_y)
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                        optimizer.step()
                model.eval()
                self.torch_model = model
                self.torch_device = device_name
                self.method = f"stg_ncde_torchcde_{device_name}"
                self.version = MODEL_SPECS["stg_ncde_inventory"]["version"]
                self.torch_training_error = None
                return
            except Exception as exc:
                self.torch_model = None
                self.torch_training_error = f"torch training failed on {device_name}: {exc}"

    def _torch_probability(self, rows: pd.DataFrame) -> np.ndarray | None:
        if self.torch_model is None:
            return None
        try:
            import torch
            import torchcde

            controls = self._control_path(rows, fit=False)
            device = next(self.torch_model.parameters()).device
            with torch.no_grad():
                x_tensor = torch.tensor(controls, dtype=torch.float32, device=device)
                coeffs = torchcde.linear_interpolation_coeffs(x_tensor)
                logits = self.torch_model(coeffs)
                p = torch.sigmoid(logits).cpu().numpy()
            return np.clip(p.astype(float), 0.01, 0.99)
        except Exception as exc:
            self.torch_training_error = f"torch prediction failed: {exc}"
            return None

    def fit(self, examples: pd.DataFrame) -> "GraphNCDEAvailabilitySurrogate":
        if len(examples) < 50 or examples["has_ebike"].nunique() < 2:
            return self
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler

            latent = self._latent_features(examples)
            self.calibrator = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, class_weight="balanced", C=0.9),
            )
            self.calibrator.fit(latent, examples["has_ebike"])
        except Exception:
            self.calibrator = None
        self._fit_torch_cde(examples)
        return self

    def predict_proba(self, rows: pd.DataFrame) -> np.ndarray:
        raw = self._raw_probability(rows)
        if self.calibrator is None:
            p = raw
        else:
            calibrated = self.calibrator.predict_proba(self._latent_features(rows))[:, 1]
            p = 0.70 * calibrated + 0.30 * raw
        torch_p = self._torch_probability(rows)
        if torch_p is not None:
            p = 0.80 * torch_p + 0.20 * p
        return np.column_stack([1.0 - np.clip(p, 0.01, 0.99), np.clip(p, 0.01, 0.99)])


class EmpiricalBayesAvailabilityModel:
    method = "empirical_bayes_trained"
    model_version = "empirical-bayes-v1"

    def predict_proba(self, rows: pd.DataFrame) -> np.ndarray:
        prior = _series(rows, "nearby_same_hour_rate", 0.35).clip(0.01, 0.99)
        station_rate = _series(rows, "station_same_hour_rate", 0.35).clip(0.01, 0.99)
        n = _series(rows, "station_same_hour_n", 0.0).clip(lower=0.0)
        smoothed = (station_rate * n + prior * 20.0) / (n + 20.0)
        current = _series(rows, "num_ebikes_available", np.nan).fillna(_series(rows, "current_ebikes_clipped"))
        horizon = _series(rows, "horizon_minutes", 10.0)
        trend = _series(rows, "trend_10m", 0.0)
        p = np.where(
            current <= 0,
            0.75 * smoothed + 0.25 * prior + np.maximum(0.0, trend).clip(0.0, 3.0) * 0.05,
            np.where(
                current == 1,
                0.55 + 0.35 * smoothed,
                np.where(current == 2, 0.72 + 0.22 * smoothed, 0.84 + 0.12 * smoothed),
            ),
        )
        p = np.asarray(p, dtype=float) - np.maximum(0.0, horizon - 5.0) * 0.01
        p = np.clip(p, 0.01, 0.99)
        return np.column_stack([1.0 - p, p])


def _fit_inventory_world_model(examples: pd.DataFrame) -> FittedAvailabilityModel:
    n = len(examples)
    positives = int(examples["has_ebike"].sum()) if n else 0
    spec = MODEL_SPECS["inventory_world"]
    return FittedAvailabilityModel(
        model=InventoryWorldRolloutModel(),
        trained_at=_utc_now(),
        n_examples=n,
        n_positive=positives,
        n_negative=n - positives,
        method="distributional_inventory_world_rollout",
        model_key="inventory_world",
        label=spec["label"],
        model_version=spec["version"],
    )


def _fit_stg_ncde_model(examples: pd.DataFrame) -> FittedAvailabilityModel:
    n = len(examples)
    positives = int(examples["has_ebike"].sum()) if n else 0
    model = GraphNCDEAvailabilitySurrogate().fit(examples)
    spec = MODEL_SPECS["stg_ncde_inventory"]
    return FittedAvailabilityModel(
        model=model,
        trained_at=_utc_now(),
        n_examples=n,
        n_positive=positives,
        n_negative=n - positives,
        method=model.method,
        model_key="stg_ncde_inventory",
        label=spec["label"],
        model_version=spec["version"],
    )


def _fit_sklearn_model(examples: pd.DataFrame, model_key: str = ACTIVE_MODEL_KEY) -> FittedAvailabilityModel:
    n = len(examples)
    positives = int(examples["has_ebike"].sum()) if n else 0
    negatives = n - positives
    if model_key == "empirical":
        spec = MODEL_SPECS[model_key]
        return FittedAvailabilityModel(
            model=EmpiricalBayesAvailabilityModel(),
            trained_at=_utc_now(),
            n_examples=n,
            n_positive=positives,
            n_negative=negatives,
            method="empirical_bayes_trained",
            model_key=model_key,
            label=spec["label"],
            model_version=spec["version"],
        )
    if n < 50 or positives < 10 or negatives < 10:
        return _empty_model(model_key, n, positives, negatives)
    try:
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        if model_key == "logistic":
            base = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, class_weight="balanced", C=0.8),
            )
            method = "calibrated_logistic"
        elif model_key == "random_forest":
            base = RandomForestClassifier(
                n_estimators=160,
                max_depth=10,
                min_samples_leaf=4,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            )
            method = "calibrated_random_forest"
        elif model_key == "gradient_boosting":
            base = HistGradientBoostingClassifier(
                max_iter=220,
                learning_rate=0.05,
                max_leaf_nodes=31,
                min_samples_leaf=12,
                l2_regularization=0.01,
                random_state=42,
            )
            method = "calibrated_hist_gradient_boosting"
        else:
            raise ValueError(f"Unknown model_key: {model_key}")

        cv = 3 if positives >= 15 and negatives >= 15 else 2
        model = CalibratedClassifierCV(base, method="sigmoid", cv=cv)
        feature_columns = _feature_columns_for_model(model_key)
        model.fit(examples[feature_columns], examples["has_ebike"])
        spec = MODEL_SPECS[model_key]
        return FittedAvailabilityModel(
            model=model,
            trained_at=_utc_now(),
            n_examples=n,
            n_positive=positives,
            n_negative=negatives,
            method=method,
            model_key=model_key,
            label=spec["label"],
            model_version=spec["version"],
        )
    except Exception:
        return _empty_model(model_key, n, positives, negatives)


def _sota_model_class(model_key: str):
    if model_key == "dg_nissm":
        from .dg_nissm import DGNISSMModel

        return DGNISSMModel
    if model_key == "cc_nissm":
        from .cc_nissm import CCNISSMModel

        return CCNISSMModel
    if model_key == "stg_ncde_inventory":
        from .stg_ncde_inventory import STGNCDEInventoryModel

        return STGNCDEInventoryModel
    if model_key == "tft_inventory":
        from .tft_inventory import TFTInventoryModel

        return TFTInventoryModel
    raise KeyError(model_key)


def _fit_sota_bootstrap_model(examples: pd.DataFrame, model_key: str) -> FittedAvailabilityModel:
    n = len(examples)
    positives = int(examples["has_ebike"].sum()) if n and "has_ebike" in examples else 0
    spec = MODEL_SPECS[model_key]
    cls = _sota_model_class(model_key)
    model = cls()
    try:
        model.fit(examples)
    except Exception:
        model = cls()
    return FittedAvailabilityModel(
        model=model,
        trained_at=getattr(model, "trained_at", _utc_now()),
        n_examples=n,
        n_positive=positives,
        n_negative=n - positives,
        method=getattr(model, "method", f"{model_key}_bootstrap"),
        model_key=model_key,
        label=spec["label"],
        model_version=getattr(model, "model_version", spec["version"]),
    )


def _load_registry_model(
    conn: duckdb.DuckDBPyConnection,
    model_key: str,
    *,
    trained_only: bool = False,
) -> FittedAvailabilityModel | None:
    try:
        from . import model_registry

        artifact = (
            model_registry.load_latest_trained_artifact(conn, model_key)
            if trained_only
            else model_registry.load_latest_artifact(conn, model_key)
        )
    except Exception:
        return None
    if not artifact or artifact.get("model") is None:
        return None
    model = artifact["model"]
    if model_key == "dg_nissm" and not (
        getattr(model, "trained", False)
        and getattr(model, "net", None) is not None
        and getattr(model, "tabular_scaler", None) is not None
    ):
        return None
    try:
        from . import model_registry

        if trained_only and not model_registry.is_trained_artifact(artifact):
            return None
    except Exception:
        if trained_only:
            return None
    spec = MODEL_SPECS[model_key]
    metrics = artifact.get("metrics_json") or {}
    method = (
        getattr(artifact["model"], "method", None)
        or metrics.get("method")
        or f"{model_key}_artifact"
    )
    return FittedAvailabilityModel(
        model=artifact["model"],
        trained_at=pd.Timestamp(artifact.get("trained_at") or _utc_now()).to_pydatetime(),
        n_examples=int(metrics.get("n_train") or metrics.get("training_examples") or 0),
        n_positive=int(metrics.get("training_positive") or 0),
        n_negative=int(metrics.get("training_negative") or 0),
        method=str(method),
        model_key=model_key,
        label=spec["label"],
        model_version=str(artifact.get("model_version") or spec["version"]),
        artifact_id=str(artifact.get("artifact_id")) if artifact.get("artifact_id") else None,
    )


def _runtime_fallback_model(
    model_key: str,
    n_examples: int = 0,
    n_positive: int = 0,
) -> FittedAvailabilityModel:
    spec = MODEL_SPECS[model_key]
    if model_key in SOTA_PRIMARY_MODEL_KEYS:
        if model_key == "dg_nissm":
            return FittedAvailabilityModel(
                model=None,
                trained_at=_utc_now(),
                n_examples=int(n_examples),
                n_positive=int(n_positive),
                n_negative=max(0, int(n_examples) - int(n_positive)),
                method="dg_nissm_unavailable_no_trained_artifact",
                model_key=model_key,
                label=spec["label"],
                model_version=spec["version"],
                artifact_id=None,
                model_warning="No trained DG-NISSM artifact found; using non-DG fallback models.",
            )
        try:
            model = _sota_model_class(model_key)()
        except Exception:
            model = None
        method = f"{model_key}_bootstrap_no_artifact"
        if model_key == "stg_ncde_inventory":
            method = "stg_ncde_inventory_flow_fallback"
        return FittedAvailabilityModel(
            model=model,
            trained_at=_utc_now(),
            n_examples=int(n_examples),
            n_positive=int(n_positive),
            n_negative=max(0, int(n_examples) - int(n_positive)),
            method=method,
            model_key=model_key,
            label=spec["label"],
            model_version=spec["version"],
            artifact_id=None,
            model_warning="No trained artifact found; using lightweight fallback.",
        )
    if model_key == "inventory_world":
        model = _fit_inventory_world_model(pd.DataFrame({"has_ebike": [1] * int(n_positive) + [0] * max(0, int(n_examples) - int(n_positive))}))
        model.model_warning = "No trained SOTA artifact found; using inventory fallback."
        return model
    model = _empty_model(model_key, int(n_examples), int(n_positive), max(0, int(n_examples) - int(n_positive)))
    model.method = f"{model_key}_empirical_fallback"
    model.model_warning = "No trained baseline artifact found; using empirical fallback."
    return model


def _load_runtime_model_suite(conn: duckdb.DuckDBPyConnection) -> FittedModelSuite:
    models: dict[str, FittedAvailabilityModel] = {}
    try:
        n_examples, n_positive = conn.execute(
            """
            SELECT COUNT(*), SUM(CASE WHEN COALESCE(num_ebikes_available, 0) >= 1 THEN 1 ELSE 0 END)
            FROM station_status
            WHERE last_reported >= now() - INTERVAL '2 hours'
            """
        ).fetchone()
    except Exception:
        n_examples, n_positive = 0, 0
    for model_key in MODEL_KEYS:
        loaded = _load_registry_model(
            conn,
            model_key,
            trained_only=model_key in SOTA_PRIMARY_MODEL_KEYS,
        )
        models[model_key] = loaded if loaded is not None else _runtime_fallback_model(
            model_key,
            int(n_examples or 0),
            int(n_positive or 0),
        )
    suite = FittedModelSuite(models)
    return suite


def _fit_model_suite(examples: pd.DataFrame, conn: duckdb.DuckDBPyConnection | None = None) -> FittedModelSuite:
    models: dict[str, FittedAvailabilityModel] = {}
    for model_key in MODEL_KEYS:
        loaded = _load_registry_model(conn, model_key, trained_only=True) if conn is not None and model_key in SOTA_PRIMARY_MODEL_KEYS else None
        if loaded is not None:
            models[model_key] = loaded
        elif model_key == "dg_nissm":
            n = len(examples)
            positives = int(examples["has_ebike"].sum()) if n and "has_ebike" in examples else 0
            models[model_key] = _runtime_fallback_model(model_key, n, positives)
        elif model_key in SOTA_PRIMARY_MODEL_KEYS:
            models[model_key] = _fit_sota_bootstrap_model(examples, model_key)
        elif model_key == "inventory_world":
            models[model_key] = _fit_inventory_world_model(examples)
        else:
            models[model_key] = _fit_sklearn_model(examples, model_key)
    return FittedModelSuite(models)


def _best_key_from_summary(summary: dict, eligible: tuple[str, ...]) -> str | None:
    leaderboard = summary.get("model_leaderboard") or summary.get("by_model") or []
    eligible_set = set(eligible)
    for row in leaderboard:
        key = row.get("model_key")
        if key in eligible_set and int(row.get("n") or 0) > 0:
            return str(key)
    return None


def _active_bootstrap_warning(suite: FittedModelSuite) -> str | None:
    active_meta = model_meta_from_fitted(suite.active_key, suite.active)
    if suite.active_key in SOTA_PRIMARY_MODEL_KEYS and is_bootstrap_or_fallback(active_meta):
        return "No trained SOTA artifact registered; using capped provisional probabilities."
    return suite.active.model_warning


def select_primary_driver(
    conn: duckdb.DuckDBPyConnection,
    candidate_model_keys: tuple[str, ...] = SOTA_PRIMARY_MODEL_KEYS,
    window_hours: int = 24,
    min_resolved: int = 30,
    fallback_order: tuple[str, ...] = ("cc_nissm", "dg_nissm", "stg_ncde_inventory", "tft_inventory", "inventory_world"),
    suite: FittedModelSuite | None = None,
) -> dict:
    usable = {
        key
        for key, model in (suite.models.items() if suite is not None else [])
        if model.usable
    }
    trained_sota = {
        key
        for key, model in (suite.models.items() if suite is not None else [])
        if is_primary_eligible_trained_sota(model_meta_from_fitted(key, model))
    }
    if FORCED_ACTIVE_MODEL_KEY:
        forced = str(FORCED_ACTIVE_MODEL_KEY)
        if forced in MODEL_KEYS and (not usable or forced in usable):
            forced_model = suite.models.get(forced) if suite is not None else None
            forced_meta = model_meta_from_fitted(forced, forced_model) if forced_model is not None else {"artifact_id": None, "method": "fallback"}
            return {
                "active_model_key": forced,
                "active_model_source": "cold_start_sota_bootstrap"
                if forced in SOTA_PRIMARY_MODEL_KEYS and is_bootstrap_or_fallback(forced_meta)
                else "forced_env",
                "selection_metric": "forced_env",
                "window_hours": window_hours,
            }

    if suite is not None:
        registry_module = None
        try:
            from . import model_registry

            registry_module = model_registry
            active_artifact = model_registry.load_active_artifact(conn)
        except Exception:
            active_artifact = None
        active_artifact_key = str((active_artifact or {}).get("model_key") or "")
        if (
            registry_module is not None
            and
            active_artifact_key in candidate_model_keys
            and active_artifact_key in suite.models
            and registry_module.is_trained_artifact(active_artifact)
        ):
            return {
                "active_model_key": active_artifact_key,
                "active_model_source": "trained_artifact",
                "selection_metric": "registry_active",
                "window_hours": window_hours,
            }

    if ACTIVE_MODEL_POLICY == "best_sota":
        try:
            from . import model_eval

            best = model_eval.best_performing_model(
                conn,
                window_hours=window_hours,
                min_n=min_resolved,
                eligible_model_keys=tuple(candidate_model_keys),
                metric="decision_rank_loss",
            )
        except Exception:
            best = {}
        key = best.get("best_model_key")
        if key in candidate_model_keys and key in trained_sota:
            return {
                "active_model_key": str(key),
                "active_model_source": "best_sota_recent_performance",
                "selection_metric": best.get("metric") or "decision_rank_loss",
                "window_hours": window_hours,
                "selection": best,
            }

    if suite is not None:
        cc_model = suite.models.get("cc_nissm")
        if cc_model is not None and is_primary_eligible_trained_sota(model_meta_from_fitted("cc_nissm", cc_model)):
            return {
                "active_model_key": "cc_nissm",
                "active_model_source": "latest_trained_cc_nissm",
                "selection_metric": "latest_trained_cc_nissm",
                "window_hours": window_hours,
            }

    for key in fallback_order:
        if key in candidate_model_keys and (not usable or key in usable):
            model = suite.models.get(key) if suite is not None else None
            meta = model_meta_from_fitted(key, model) if model is not None else {"artifact_id": None, "method": "fallback"}
            source = "cold_start_sota_bootstrap" if is_bootstrap_or_fallback(meta) else "artifact_available"
            return {
                "active_model_key": key,
                "active_model_source": source,
                "selection_metric": "artifact_available",
                "window_hours": window_hours,
            }
    return {
        "active_model_key": "inventory_world",
        "active_model_source": "emergency_legacy_fallback",
        "selection_metric": "fallback",
        "window_hours": window_hours,
    }


def resolve_active_model_key(
    conn: duckdb.DuckDBPyConnection,
    suite: FittedModelSuite,
    performance_window_hours: int = 24,
) -> str:
    selection = select_primary_driver(conn, window_hours=performance_window_hours, suite=suite)
    active_key = str(selection["active_model_key"])
    if active_key not in suite.models or not suite.models[active_key].usable:
        active_key = "inventory_world" if suite.models.get("inventory_world", None) and suite.models["inventory_world"].usable else next(iter(suite.models))
        suite.active_source = "emergency_legacy_fallback"
    else:
        suite.active_source = str(selection.get("active_model_source") or "default_cc_nissm")
    suite.active_key = active_key
    active_meta = model_meta_from_fitted(active_key, suite.active)
    if active_key in SOTA_PRIMARY_MODEL_KEYS and is_bootstrap_or_fallback(active_meta):
        suite.active_source = "cold_start_sota_bootstrap"
        suite.active.model_warning = _active_bootstrap_warning(suite)
    suite.selection_metric = str(selection.get("selection_metric") or "decision_rank_loss")
    suite.selection_window_hours = int(selection.get("window_hours") or performance_window_hours)
    try:
        from . import model_eval

        perf = model_eval.performance_summary(
            conn,
            window_hours=performance_window_hours,
            resolve=False,
            initialize_schema=False,
        )
        suite.best_evaluated_model_key = (perf.get("best_current_model") or {}).get("best_model_key") or _best_key_from_summary(perf, MODEL_KEYS)
        suite.best_sota_model_key = (perf.get("best_sota_model") or {}).get("best_model_key") or _best_key_from_summary(perf, SOTA_PRIMARY_MODEL_KEYS)
        suite.best_baseline_model_key = _best_key_from_summary(perf, BASELINE_MODEL_KEYS)
        usable_keys = tuple(key for key, model in suite.models.items() if model.usable)
        trained_sota_keys = tuple(
            key
            for key, model in suite.models.items()
            if is_primary_eligible_trained_sota(model_meta_from_fitted(key, model))
        )
        suite.best_usable_model_key = _best_key_from_summary(perf, usable_keys) or (active_key if suite.active.usable else None)
        suite.best_trained_sota_model_key = _best_key_from_summary(perf, trained_sota_keys) or (trained_sota_keys[0] if trained_sota_keys else None)
    except Exception:
        suite.best_evaluated_model_key = suite.best_evaluated_model_key or active_key
        suite.best_sota_model_key = suite.best_sota_model_key or (active_key if active_key in SOTA_PRIMARY_MODEL_KEYS else None)
        suite.best_usable_model_key = suite.best_usable_model_key or (active_key if suite.active.usable else None)
        suite.best_trained_sota_model_key = suite.best_trained_sota_model_key or None
    return suite.active_key


def get_availability_model(
    conn: duckdb.DuckDBPyConnection,
    force: bool = False,
    ttl_seconds: int = 600,
) -> FittedAvailabilityModel:
    suite = get_availability_model_suite(conn, force=force, ttl_seconds=ttl_seconds)
    return suite.active


def get_availability_model_suite(
    conn: duckdb.DuckDBPyConnection,
    force: bool = False,
    ttl_seconds: int = 600,
) -> FittedModelSuite:
    cached = _MODEL_CACHE.get("availability_suite")
    if (
        not force
        and isinstance(cached, FittedModelSuite)
        and (time.time() - _MODEL_CACHE.get("availability_suite_ts", 0) < ttl_seconds)
    ):
        return cached
    if os.getenv("DIVVY_DISABLE_REQUEST_TRAINING", "1") != "0":
        suite = _load_runtime_model_suite(conn)
        resolve_active_model_key(conn, suite)
        _MODEL_CACHE["availability_suite"] = suite
        _MODEL_CACHE["availability_suite_ts"] = time.time()
        _MODEL_CACHE["availability"] = suite.active
        _MODEL_CACHE["availability_ts"] = time.time()
        return suite
    examples = build_training_examples(
        conn,
        history_hours=REQUEST_BOOTSTRAP_HISTORY_HOURS,
        max_source_rows=REQUEST_BOOTSTRAP_MAX_SOURCE_ROWS,
    )
    suite = _fit_model_suite(examples, conn=conn)
    resolve_active_model_key(conn, suite)
    _MODEL_CACHE["availability_suite"] = suite
    _MODEL_CACHE["availability_suite_ts"] = time.time()
    _MODEL_CACHE["availability"] = suite.active
    _MODEL_CACHE["availability_ts"] = time.time()
    return suite


def _history_rates_for_candidates(
    conn: duckdb.DuckDBPyConnection,
    station_ids: list[str],
    now: datetime,
    history_hours: int = 24 * 14,
) -> pd.DataFrame:
    if not station_ids:
        return pd.DataFrame()
    local = pd.Timestamp(now, tz="UTC").tz_convert(LOCAL_TZ)
    hour = int(local.hour)
    dow = int(local.dayofweek)
    placeholders, params = _ids_clause(station_ids)
    rows = conn.execute(
        f"""
        SELECT
          station_id,
          AVG(CASE WHEN num_ebikes_available >= 1 THEN 1.0 ELSE 0.0 END) AS station_same_hour_rate,
          COUNT(*) AS station_same_hour_n
        FROM station_status
        WHERE station_id IN ({placeholders})
          AND last_reported >= ? - (? * INTERVAL '1 hour')
          AND CAST(EXTRACT(HOUR FROM (CAST(last_reported AS TIMESTAMP) AT TIME ZONE 'UTC' AT TIME ZONE '{LOCAL_TZ}')) AS INTEGER) = ?
          AND CAST(EXTRACT(DOW FROM (CAST(last_reported AS TIMESTAMP) AT TIME ZONE 'UTC' AT TIME ZONE '{LOCAL_TZ}')) AS INTEGER) = ?
        GROUP BY station_id
        """,
        [*params, now, history_hours, hour, (dow + 1) % 7],
    ).df()
    global_row = conn.execute(
        f"""
        SELECT AVG(CASE WHEN num_ebikes_available >= 1 THEN 1.0 ELSE 0.0 END) AS global_rate
        FROM station_status
        WHERE last_reported >= ? - (? * INTERVAL '1 hour')
          AND CAST(EXTRACT(HOUR FROM (CAST(last_reported AS TIMESTAMP) AT TIME ZONE 'UTC' AT TIME ZONE '{LOCAL_TZ}')) AS INTEGER) = ?
        """,
        [now, history_hours, hour],
    ).fetchone()
    global_rate = float(global_row[0]) if global_row and global_row[0] is not None else 0.35
    out = pd.DataFrame({"station_id": station_ids})
    out = out.merge(rows, on="station_id", how="left")
    out["station_same_hour_n"] = out["station_same_hour_n"].fillna(0).astype(int)
    out["station_same_hour_rate"] = out["station_same_hour_rate"].fillna(global_rate)
    out["nearby_same_hour_rate"] = float(out["station_same_hour_rate"].mean()) if len(out) else global_rate
    edges = _station_graph_edges(conn)
    if not edges.empty:
        neighbor_rates = out[["station_id", "station_same_hour_rate"]].rename(
            columns={"station_id": "neighbor_id", "station_same_hour_rate": "neighbor_rate"}
        )
        neighbor = (
            edges[edges["station_id"].isin(station_ids)]
            .merge(neighbor_rates, on="neighbor_id", how="left")
            .groupby("station_id", as_index=False)["neighbor_rate"]
            .mean()
            .rename(columns={"neighbor_rate": "station_neighbor_same_hour_rate"})
        )
        out = out.merge(neighbor, on="station_id", how="left")
    else:
        out["station_neighbor_same_hour_rate"] = np.nan
    out["station_neighbor_same_hour_rate"] = out["station_neighbor_same_hour_rate"].fillna(
        out["nearby_same_hour_rate"]
    )
    return out


def _latest_trends(
    conn: duckdb.DuckDBPyConnection,
    station_ids: list[str],
    now: datetime,
    history_minutes: int = 30,
) -> pd.DataFrame:
    if not station_ids:
        return pd.DataFrame()
    empty = pd.DataFrame({"station_id": station_ids})
    for column in ["trend_5m", "trend_10m", "trend_15m", "churn_rate"]:
        empty[column] = 0.0
    placeholders, params = _ids_clause(station_ids)
    rows = conn.execute(
        f"""
        SELECT station_id, last_reported, num_ebikes_available
        FROM station_status
        WHERE station_id IN ({placeholders})
          AND last_reported >= ? - (? * INTERVAL '1 minute')
        ORDER BY station_id, last_reported
        """,
        [*params, now, history_minutes],
    ).df()
    if rows.empty:
        return empty
    rows["last_reported"] = pd.to_datetime(rows["last_reported"])
    rows = _add_trend_features(rows)
    latest = rows.sort_values("last_reported").groupby("station_id").tail(1)
    return empty[["station_id"]].merge(
        latest[["station_id", "trend_5m", "trend_10m", "trend_15m", "churn_rate"]],
        on="station_id",
        how="left",
    ).fillna(0.0)


def _empirical_probability(row: pd.Series) -> tuple[float, int]:
    prior = _clip_probability(row.get("nearby_same_hour_rate"), 0.35)
    station_rate = _clip_probability(row.get("station_same_hour_rate"), prior)
    n = int(row.get("station_same_hour_n") or 0)
    smoothed = (station_rate * n + prior * 20.0) / (n + 20.0)
    current = int(row.get("num_ebikes_available") or 0)
    horizon = int(row.get("horizon_minutes") or 10)
    trend = float(row.get("trend_10m") or 0.0)

    if current <= 0:
        p = 0.75 * smoothed + 0.25 * prior
        if trend > 0:
            p += min(0.15, trend * 0.05)
    elif current == 1:
        p = 0.55 + 0.35 * smoothed
    elif current == 2:
        p = 0.72 + 0.22 * smoothed
    else:
        p = 0.84 + 0.12 * smoothed

    p -= max(0.0, horizon - 5) * 0.01
    return _clip_probability(p), n


def _score_feature_rows(
    rows: pd.DataFrame,
    fitted: FittedAvailabilityModel,
    *,
    n_resolved: int = 0,
    debug: bool = False,
) -> pd.DataFrame:
    scored = rows.copy()
    model_meta = model_meta_from_fitted(fitted.model_key, fitted)
    empirical: list[float] = []
    samples: list[int] = []
    for _, row in scored.iterrows():
        p, n = _empirical_probability(row)
        empirical.append(p)
        samples.append(n)
    scored["p_empirical"] = empirical
    scored["sample_size"] = samples

    distribution_error: str | None = None
    if fitted.usable and fitted.model is not None and hasattr(fitted.model, "predict_distribution"):
        try:
            distribution = fitted.model.predict_distribution(scored, debug=debug)
        except TypeError:
            try:
                distribution = fitted.model.predict_distribution(scored)
            except Exception as exc:
                distribution = None
                distribution_error = str(exc)
        except Exception as exc:
            distribution = None
            distribution_error = str(exc)
        if distribution is not None:
            for column in distribution.columns:
                scored[column] = distribution[column]
            scored["p_learned"] = scored["p_has_ebike"]
            scored["model_method"] = fitted.method
        else:
            scored["p_learned"] = np.nan
            scored["p_has_ebike"] = scored["p_empirical"]
            scored["model_method"] = f"{fitted.method}_unavailable"
            scored["model_error"] = distribution_error or "distribution_unavailable"
    elif fitted.usable and fitted.model is not None:
        feature_columns = _feature_columns_for_model(fitted.model_key)
        learned = fitted.model.predict_proba(scored[feature_columns])[:, 1]
        scored["p_learned"] = learned
        scored["p_has_ebike"] = [
            _clip_probability(0.7 * l + 0.3 * e)
            for l, e in zip(scored["p_learned"], scored["p_empirical"])
        ]
        scored["model_method"] = fitted.method
    else:
        scored["p_learned"] = np.nan
        scored["p_has_ebike"] = scored["p_empirical"]
        scored["model_method"] = fitted.method
        if fitted.model_warning:
            scored["model_error"] = fitted.model_warning

    if fitted.model_key in SOTA_PRIMARY_MODEL_KEYS:
        guarded = []
        for _, row in scored.iterrows():
            p = apply_cold_start_probability_guard(
                _finite_float(row.get("p_has_ebike"), 0.5),
                current_ebikes=int(_finite_float(row.get("num_ebikes_available"), 0.0)),
                horizon_minutes=int(_finite_float(row.get("horizon_minutes"), 10.0)),
                model_meta=model_meta,
                n_resolved=int(n_resolved),
            )
            if is_bootstrap_or_fallback(model_meta) or int(n_resolved) < 1000:
                p = cap_zero_current_without_inbound_support(row, p)
            guarded.append(p)
        scored["p_has_ebike"] = pd.Series(guarded, index=scored.index, dtype=float)
        scored["p_learned"] = pd.to_numeric(scored["p_learned"], errors="coerce").clip(0.001, 0.999)

    if "p_zero" not in scored.columns:
        scored["p_zero"] = 1.0 - scored["p_has_ebike"]
    else:
        scored["p_zero"] = 1.0 - pd.to_numeric(scored["p_has_ebike"], errors="coerce")
    if "p_count_ebikes" in scored.columns:
        scored["p_count_ebikes"] = [
            _align_count_distribution_to_p_zero(dist, p_zero)
            for dist, p_zero in zip(scored["p_count_ebikes"], scored["p_zero"])
        ]
    scored["p_appears"] = np.where(scored["num_ebikes_available"] <= 0, scored["p_has_ebike"], np.nan)
    scored["p_survives"] = np.where(scored["num_ebikes_available"] > 0, scored["p_has_ebike"], np.nan)
    scored["model_key"] = fitted.model_key
    scored["model_label"] = fitted.label
    scored["model_version"] = fitted.model_version
    return scored


def _resolved_outcome_counts_by_model(
    conn: duckdb.DuckDBPyConnection,
    window_hours: int = 24,
) -> dict[str, int]:
    try:
        rows = conn.execute(
            """
            SELECT COALESCE(f.model_key, 'logistic') AS model_key, COUNT(*) AS n
            FROM model_forecasts f
            JOIN model_outcomes o USING (forecast_id)
            WHERE o.resolved_at >= now() - (? * INTERVAL '1 hour')
            GROUP BY COALESCE(f.model_key, 'logistic')
            """,
            [int(window_hours)],
        ).fetchall()
    except Exception:
        return {}
    return {str(model_key): int(n or 0) for model_key, n in rows}


def score_candidates(
    conn: duckdb.DuckDBPyConnection,
    candidates: pd.DataFrame,
    horizons: tuple[int, ...] = HORIZONS,
    search_radius_km: float | None = None,
    model_keys: tuple[str, ...] | list[str] | None = None,
    debug: bool = False,
) -> tuple[pd.DataFrame, FittedModelSuite]:
    if candidates.empty:
        return candidates.copy(), get_availability_model_suite(conn)

    now = _utc_now()
    station_ids = candidates["station_id"].astype(str).tolist()
    rates = _history_rates_for_candidates(conn, station_ids, now)
    trends = _latest_trends(conn, station_ids, now)
    graph_static = _station_graph_static_features(conn)
    live_neighbor = _live_neighbor_features(conn, station_ids, now)
    live_inflight = _live_inflight_features(conn, station_ids, now)
    free_density = _free_floating_density_features(conn, candidates, now)
    base = (
        candidates
        .merge(rates, on="station_id", how="left")
        .merge(trends, on="station_id", how="left")
        .merge(graph_static, on="station_id", how="left")
        .merge(live_neighbor, on="station_id", how="left")
        .merge(live_inflight, on="station_id", how="left")
        .merge(free_density, on="station_id", how="left")
    )
    base = _add_inventory_features(base)
    base["station_same_hour_rate"] = base["station_same_hour_rate"].fillna(0.35)
    base["nearby_same_hour_rate"] = base["nearby_same_hour_rate"].fillna(
        base["station_same_hour_rate"].mean()
    )
    base["station_neighbor_same_hour_rate"] = base["station_neighbor_same_hour_rate"].fillna(
        base["nearby_same_hour_rate"]
    )
    base[["trend_5m", "trend_10m", "trend_15m", "churn_rate"]] = base[
        ["trend_5m", "trend_10m", "trend_15m", "churn_rate"]
    ].fillna(0.0)
    base[[
        "station_neighbor_count_500m",
        "station_neighbor_capacity_500m",
        "station_neighbor_recent_ebikes",
        "station_neighbor_recent_zero_rate",
    ]] = base[[
        "station_neighbor_count_500m",
        "station_neighbor_capacity_500m",
        "station_neighbor_recent_ebikes",
        "station_neighbor_recent_zero_rate",
    ]].fillna(0.0)
    for column in LIVE_INFLIGHT_FEATURE_COLUMNS + FREE_FLOATING_FEATURE_COLUMNS:
        if column not in base.columns:
            base[column] = 0.0
        base[column] = pd.to_numeric(base[column], errors="coerce").fillna(0.0)
    latest_report_base = pd.to_datetime(base["last_reported"], errors="coerce")
    base["status_age_minutes"] = (pd.Timestamp(now) - latest_report_base).dt.total_seconds() / 60.0
    base["station_closed_penalty_flag"] = (~base.get("is_renting", pd.Series(True, index=base.index)).fillna(True).astype(bool)).astype(int)
    base["stale_status_penalty_flag"] = (base["status_age_minutes"].fillna(999.0) > 10.0).astype(int)

    feature_rows: list[pd.DataFrame] = []
    for horizon in horizons:
        f = base.copy()
        f["forecasted_at"] = now
        f["target_at"] = now + timedelta(minutes=horizon)
        f["horizon_minutes"] = horizon
        f["current_ebikes_clipped"] = f["num_ebikes_available"].fillna(0).clip(0, 6)
        f["current_bucket"] = f["num_ebikes_available"].fillna(0).map(current_bucket)
        feature_rows.append(f)
    rows = pd.concat(feature_rows, ignore_index=True)
    rows = add_temporal_features(rows, "forecasted_at")
    rows = add_calendar_features(rows, "forecasted_at")
    rows = _add_trip_features(conn, rows, now=now)
    rows = _add_weather_features(conn, rows, "target_at")
    rows = _fill_feature_defaults(rows, CALENDAR_FEATURE_COLUMNS)
    rows = _fill_feature_defaults(rows, TRIP_FEATURE_COLUMNS)
    rows = _fill_feature_defaults(rows, LIVE_INFLIGHT_FEATURE_COLUMNS)
    rows = _fill_feature_defaults(rows, FREE_FLOATING_FEATURE_COLUMNS)
    rows = _fill_feature_defaults(rows, STATUS_QUALITY_FEATURE_COLUMNS)
    rows = _weather_defaults(rows)
    rows[FEATURE_COLUMNS] = rows[FEATURE_COLUMNS].fillna(0.0)

    suite = get_availability_model_suite(conn)
    active_model_key = suite.active_key if suite.active_key in suite.models else "cc_nissm"
    selected_model_keys = [str(key) for key in (model_keys or list(suite.models.keys())) if str(key) in suite.models]
    if active_model_key not in selected_model_keys:
        selected_model_keys.insert(0, active_model_key)
    resolved_counts = _resolved_outcome_counts_by_model(conn)
    scored_by_model = {
        model_key: _score_feature_rows(
            rows,
            fitted,
            n_resolved=resolved_counts.get(model_key, 0),
            debug=debug,
        )
        for model_key, fitted in suite.models.items()
        if model_key in selected_model_keys
    }
    scored_long = scored_by_model[active_model_key]

    wide = base.copy()
    wide["forecasted_at"] = now
    feature_snapshot = rows[rows["horizon_minutes"] == max(horizons)][
        ["station_id", *CALENDAR_FEATURE_COLUMNS, *TRIP_FEATURE_COLUMNS, *WEATHER_FEATURE_COLUMNS]
    ].copy()
    wide = wide.merge(feature_snapshot, on="station_id", how="left")
    for horizon in horizons:
        active_columns = [
            "station_id",
            "p_has_ebike",
            "p_zero",
            "p_appears",
            "p_survives",
            "p_empirical",
            "p_learned",
            "sample_size",
            "model_method",
        ]
        active_columns.extend(
            column for column in DISTRIBUTION_OUTPUT_COLUMNS if column in scored_long.columns
        )
        active_columns.extend(
            column for column in DEBUG_OUTPUT_COLUMNS if column in scored_long.columns
        )
        h = scored_long[scored_long["horizon_minutes"] == horizon][
            active_columns
        ].copy()
        suffix = f"_{horizon}m"
        h = h.rename(columns={c: f"{c}{suffix}" for c in h.columns if c != "station_id"})
        wide = wide.merge(h, on="station_id", how="left")
        for model_key, model_scored in scored_by_model.items():
            model_columns = [
                "station_id",
                "p_has_ebike",
                "p_zero",
                "p_appears",
                "p_survives",
                "p_empirical",
                "p_learned",
                "sample_size",
                "model_method",
                "model_label",
                "model_version",
            ]
            model_columns.extend(
                column for column in DISTRIBUTION_OUTPUT_COLUMNS if column in model_scored.columns
            )
            model_columns.extend(
                column for column in DEBUG_OUTPUT_COLUMNS if column in model_scored.columns
            )
            model_h = model_scored[model_scored["horizon_minutes"] == horizon][
                model_columns
            ].copy()
            model_h = model_h.rename(
                columns={
                    c: f"{c}_{horizon}m_{model_key}"
                    for c in model_h.columns
                    if c != "station_id"
                }
            )
            wide = wide.merge(model_h, on="station_id", how="left")

    latest_report = pd.to_datetime(wide["last_reported"], errors="coerce")
    age_minutes = (pd.Timestamp(now) - latest_report).dt.total_seconds() / 60.0
    wide["data_age_minutes"] = age_minutes
    wide["confidence"] = [
        _confidence(int(max(row.get("sample_size_5m") or 0, row.get("sample_size_10m") or 0)), age)
        for (_, row), age in zip(wide.iterrows(), age_minutes)
    ]
    wide["model_version"] = suite.active.model_version
    wide["baseline_version"] = BASELINE_VERSION
    wide["active_model_key"] = active_model_key
    wide["active_model_source"] = suite.active_source
    wide["best_evaluated_model_key"] = suite.best_evaluated_model_key
    wide["best_usable_model_key"] = suite.best_usable_model_key
    wide["best_sota_model_key"] = suite.best_sota_model_key
    wide["best_trained_sota_model_key"] = suite.best_trained_sota_model_key
    wide = apply_arrival_time_scores(
        wide,
        active_model_key=active_model_key,
        search_radius_km=search_radius_km,
    )
    return wide.sort_values(
        ["walk_adjusted_score", "rank_probability", "distance_km", "num_ebikes_available"],
        ascending=[False, False, True, False],
    ), suite


def model_health(conn: duckdb.DuckDBPyConnection) -> dict:
    suite = get_availability_model_suite(conn)
    fitted = suite.active
    def count_rows(table_name: str) -> int:
        try:
            return int(conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0] or 0)
        except Exception:
            return 0

    row = conn.execute(
        """
        SELECT
          COUNT(*) AS station_status_rows,
          COUNT(DISTINCT station_id) AS stations_with_status,
          MIN(last_reported) AS first_reported,
          MAX(last_reported) AS last_reported
        FROM station_status
        """
    ).fetchone()
    active_meta = model_meta_from_fitted(suite.active_key, suite.active)
    return {
        "active_model_key": suite.active_key,
        "active_model_label": suite.active.label,
        "active_model_display_label": model_display_label(active_meta),
        "active_artifact_id": suite.active.artifact_id,
        "active_model_source": suite.active_source,
        "active_model_warning": _active_bootstrap_warning(suite),
        "model_warning": _active_bootstrap_warning(suite),
        "best_evaluated_model_key": suite.best_evaluated_model_key,
        "best_usable_model_key": suite.best_usable_model_key,
        "best_sota_model_key": suite.best_sota_model_key,
        "best_trained_sota_model_key": suite.best_trained_sota_model_key,
        "best_baseline_model_key": suite.best_baseline_model_key,
        "active_equals_best": bool(suite.active_key == suite.best_evaluated_model_key) if suite.best_evaluated_model_key else None,
        "active_equals_best_evaluated": bool(suite.active_key == suite.best_evaluated_model_key) if suite.best_evaluated_model_key else None,
        "active_equals_best_usable": bool(suite.active_key == suite.best_usable_model_key) if suite.best_usable_model_key else None,
        "selection_metric": suite.selection_metric,
        "selection_window_hours": suite.selection_window_hours,
        "model_version": fitted.model_version,
        "method": fitted.method,
        "trained_at": fitted.trained_at.isoformat(),
        "training_examples": fitted.n_examples,
        "training_positive": fitted.n_positive,
        "training_negative": fitted.n_negative,
        "models": suite.summary(),
        "station_status_rows": int(row[0] or 0),
        "stations_with_status": int(row[1] or 0),
        "first_reported": row[2].isoformat() if row and row[2] else None,
        "last_reported": row[3].isoformat() if row and row[3] else None,
        "trip_rows": count_rows("divvy_trips"),
        "trip_flow_rows": count_rows("station_trip_flows"),
        "trip_route_rows": count_rows("station_trip_routes"),
        "weather_hourly_rows": count_rows("weather_hourly"),
    }


def backtest(
    conn: duckdb.DuckDBPyConnection,
    history_hours: int = 24 * 7,
) -> dict:
    examples = build_training_examples(conn, history_hours=history_hours)
    if len(examples) < 80 or examples["has_ebike"].nunique() < 2:
        return {
            "status": "insufficient_data",
            "n_examples": int(len(examples)),
            "message": "Need at least 80 labeled examples with both outcomes.",
        }

    examples = examples.sort_values("last_reported").reset_index(drop=True)
    split = max(1, int(len(examples) * 0.7))
    train = examples.iloc[:split]
    test = examples.iloc[split:]
    suite = _fit_model_suite(train, conn=conn)
    resolve_active_model_key(conn, suite)

    model_results = []
    by_horizon = []
    active_result = None
    for model_key, fitted in suite.models.items():
        scored = _score_feature_rows(test.copy(), fitted)
        y = scored["has_ebike"].astype(float).to_numpy()
        p = scored["p_has_ebike"].astype(float).clip(0.001, 0.999).to_numpy()
        brier = float(np.mean((p - y) ** 2))
        log_loss = float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
        result = {
            "model_key": model_key,
            "model_label": fitted.label,
            "model_method": fitted.method,
            "n": int(len(scored)),
            "brier_score": brier,
            "log_loss": log_loss,
            "rank_loss": float(brier + 0.05 * log_loss),
            "observed_rate": float(np.mean(y)),
            "mean_prediction": float(np.mean(p)),
        }
        model_results.append(result)
        if model_key == suite.active_key:
            active_result = result
        for horizon, group in scored.groupby("horizon_minutes"):
            gy = group["has_ebike"].astype(float).to_numpy()
            gp = group["p_has_ebike"].astype(float).clip(0.001, 0.999).to_numpy()
            h_brier = float(np.mean((gp - gy) ** 2))
            h_log_loss = float(-np.mean(gy * np.log(gp) + (1.0 - gy) * np.log(1.0 - gp)))
            by_horizon.append({
                "model_key": model_key,
                "model_label": fitted.label,
                "horizon_minutes": int(horizon),
                "n": int(len(group)),
                "brier_score": h_brier,
                "log_loss": h_log_loss,
                "rank_loss": float(h_brier + 0.05 * h_log_loss),
                "observed_rate": float(np.mean(gy)),
                "mean_prediction": float(np.mean(gp)),
            })
    model_results = sorted(model_results, key=lambda r: r["rank_loss"])
    for rank, result in enumerate(model_results, start=1):
        result["rank"] = rank
    active_result = active_result or model_results[0]
    return {
        "status": "ok",
        "model_method": active_result["model_method"],
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "brier_score": active_result["brier_score"],
        "log_loss": active_result["log_loss"],
        "rank_loss": active_result["rank_loss"],
        "observed_rate": active_result["observed_rate"],
        "mean_prediction": active_result["mean_prediction"],
        "models": model_results,
        "by_horizon": by_horizon,
    }
