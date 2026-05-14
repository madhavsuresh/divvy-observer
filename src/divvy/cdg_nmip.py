from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from . import inventory_dp
from .dg_nissm_features import (
    FLOW_LABEL_COLUMNS,
    SHIFTED_PRIOR_COLUMNS,
    SequenceSpec,
    add_shifted_empirical_priors,
    observed_flow_labels_from_states,
    sequence_array_from_rows,
)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover - import availability is checked by trainer.
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    TensorDataset = None  # type: ignore[assignment]


CDG_REQUIRED_OUTPUT_COLUMNS = [
    "p_has_ebike",
    "p_zero",
    "p_appears",
    "p_survives",
    "expected_ebikes",
    "expected_total_bikes",
    "p_count_ebikes",
    "p_count_total",
    "p_count_ebikes_json",
    "p_count_total_json",
]

CDG_DIAGNOSTIC_COLUMNS = [
    "p_capacity_violation",
    "p_dock_constrained_arrival",
    "expected_ebike_departures",
    "expected_classic_departures",
    "expected_ebike_arrivals",
    "expected_classic_arrivals",
]

CDG_DEBUG_COLUMNS = [
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

CANONICAL_FEATURE_COLUMNS = [
    "capacity",
    "capacity_clipped",
    "num_ebikes_available",
    "num_bikes_available",
    "num_docks_available",
    "current_ebikes_clipped",
    "current_total_bikes_clipped",
    "docks_available_clipped",
    "ebike_share_of_bikes",
    "dock_availability_fraction",
    "current_bucket",
    "horizon_minutes",
    "hour_sin",
    "hour_cos",
    "dow",
    "is_weekend",
    "month_sin",
    "month_cos",
    "day_of_year_sin",
    "day_of_year_cos",
    "is_commute_hour",
    "is_federal_holiday",
    "trend_5m",
    "trend_10m",
    "trend_15m",
    "churn_rate",
    "station_same_hour_rate",
    "nearby_same_hour_rate",
    "station_neighbor_same_hour_rate",
    "station_neighbor_count_500m",
    "station_neighbor_capacity_500m",
    "station_neighbor_recent_ebikes",
    "station_neighbor_recent_zero_rate",
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
    "live_inflight_ebike_due_5m",
    "live_inflight_ebike_due_10m",
    "live_inflight_ebike_due_15m",
    "live_inflight_ebike_due_20m",
    "live_inflight_classic_due_5m",
    "live_inflight_classic_due_10m",
    "live_inflight_classic_due_15m",
    "live_inflight_classic_due_20m",
    "free_floating_density_300m",
    "free_floating_density_500m",
    "free_floating_density_1000m",
    "status_age_minutes",
    "data_age_minutes",
    "station_closed_penalty_flag",
    "stale_status_penalty_flag",
    *SHIFTED_PRIOR_COLUMNS,
]


@dataclass
class CDGNMIPConfig:
    hidden_dim: int = 128
    sequence_hidden_dim: int = 64
    station_embedding_dim: int = 32
    horizon_embedding_dim: int = 16
    dropout: float = 0.08
    seq_len: int = 24
    seq_step_minutes: int = 2
    top_k: int = 16
    epochs: int = 8
    batch_size: int = 4096
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    device: str = "auto"
    runtime_device: str = "auto"
    max_examples: int = 600_000
    min_train_examples: int = 10_000
    min_valid_examples: int = 1_000
    min_positive_examples: int = 100
    min_zero_future_examples: int = 100
    early_stopping_patience: int = 2
    empirical_bayes_alpha: float = 50.0
    use_sequence: bool = True
    use_graph: bool = True
    calibrate: bool = True
    max_capacity: int = 80
    max_rollout_steps: int = 5
    exact_inventory_dp: bool = False
    loss_count_weight: float = 1.0
    loss_bce_weight: float = 0.5
    loss_flow_weight: float = 0.2
    loss_cal_weight: float = 0.05
    graph_relation_types: tuple[str, ...] = ("distance", "semantic")
    model_version: str = "dg-nissm-cdg-nmip-v1"

    def sequence_spec(self) -> SequenceSpec:
        return SequenceSpec(seq_len=int(self.seq_len), seq_step_minutes=int(self.seq_step_minutes))

    def rollout_steps(self, horizon_minutes: int | float | None) -> int:
        try:
            horizon = int(round(float(horizon_minutes)))
        except (TypeError, ValueError):
            horizon = 10
        if horizon <= 5:
            return max(1, horizon)
        return max(1, min(int(self.max_rollout_steps), int(math.ceil(horizon / 5.0))))


@dataclass
class TabularScaler:
    mean_: np.ndarray
    scale_: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "TabularScaler":
        arr = np.asarray(values, dtype=np.float32)
        mean = np.nanmean(arr, axis=0).astype(np.float32)
        scale = np.nanstd(arr, axis=0).astype(np.float32)
        mean = np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
        scale = np.where(np.isfinite(scale) & (scale >= 1e-4), scale, 1.0).astype(np.float32)
        return cls(mean_=mean, scale_=scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32)
        return np.nan_to_num((arr - self.mean_) / self.scale_, nan=0.0, posinf=6.0, neginf=-6.0).astype(np.float32)


def choose_device(requested: str = "auto"):
    if torch is None:
        raise RuntimeError("PyTorch is required for DG-NISSM CDG-NMIP")
    requested = str(requested or "auto").lower()
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def config_from_dict(value: dict[str, Any] | CDGNMIPConfig | None) -> CDGNMIPConfig:
    if isinstance(value, CDGNMIPConfig):
        return value
    data = dict(value or {})
    allowed = {field.name for field in CDGNMIPConfig.__dataclass_fields__.values()}
    return CDGNMIPConfig(**{key: data[key] for key in data if key in allowed})


def _finite_series(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default).astype(float)


def ensure_cdg_features(rows: pd.DataFrame, *, alpha: float = 50.0, add_priors: bool = False) -> pd.DataFrame:
    out = rows.copy()
    if out.empty:
        return out
    if "anchor_ts" not in out.columns:
        if "forecasted_at" in out.columns:
            out["anchor_ts"] = out["forecasted_at"]
        elif "last_reported" in out.columns:
            out["anchor_ts"] = out["last_reported"]
        else:
            out["anchor_ts"] = pd.Timestamp.utcnow().tz_localize(None)
    ts = pd.to_datetime(out["anchor_ts"], utc=True, errors="coerce")
    local = ts.dt.tz_convert("America/Chicago")
    if "local_hour" not in out.columns:
        out["local_hour"] = local.dt.hour.fillna(0).astype(int)
    if "dow" not in out.columns:
        out["dow"] = local.dt.dayofweek.fillna(0).astype(int)
    if "hour_sin" not in out.columns:
        out["hour_sin"] = np.sin(2.0 * math.pi * out["local_hour"].astype(float) / 24.0)
        out["hour_cos"] = np.cos(2.0 * math.pi * out["local_hour"].astype(float) / 24.0)
    if "is_weekend" not in out.columns:
        out["is_weekend"] = out["dow"].isin([5, 6]).astype(int)
    if "month_sin" not in out.columns:
        month = local.dt.month.fillna(1).astype(int)
        doy = local.dt.dayofyear.fillna(1).astype(int)
        out["month_sin"] = np.sin(2.0 * math.pi * month / 12.0)
        out["month_cos"] = np.cos(2.0 * math.pi * month / 12.0)
        out["day_of_year_sin"] = np.sin(2.0 * math.pi * doy / 366.0)
        out["day_of_year_cos"] = np.cos(2.0 * math.pi * doy / 366.0)
        out["is_commute_hour"] = out["local_hour"].isin([7, 8, 9, 16, 17, 18]).astype(int)
        out["is_federal_holiday"] = 0

    capacity = _finite_series(out, "capacity", np.nan).fillna(_finite_series(out, "capacity_clipped", 15.0)).clip(1.0, 80.0)
    ebikes = _finite_series(out, "num_ebikes_available", np.nan).fillna(_finite_series(out, "current_ebikes_clipped", 0.0)).clip(0.0, 80.0)
    total = _finite_series(out, "num_bikes_available", np.nan).fillna(_finite_series(out, "current_total_bikes_clipped", 0.0)).clip(lower=ebikes, upper=80.0)
    docks = _finite_series(out, "num_docks_available", np.nan).fillna((capacity - total).clip(lower=0.0)).clip(0.0, 80.0)
    out["capacity"] = capacity
    out["capacity_clipped"] = capacity.clip(0.0, 80.0)
    out["num_ebikes_available"] = ebikes
    out["num_bikes_available"] = total
    out["num_docks_available"] = docks
    out["current_ebikes_clipped"] = ebikes.clip(0.0, 6.0)
    out["current_total_bikes_clipped"] = total.clip(0.0, 80.0)
    out["docks_available_clipped"] = docks.clip(0.0, 80.0)
    out["ebike_share_of_bikes"] = np.where(total > 0, ebikes / total.clip(lower=1.0), 0.0)
    out["dock_availability_fraction"] = (docks / capacity).clip(0.0, 1.0)
    out["current_bucket"] = np.select([ebikes <= 0, ebikes == 1, ebikes == 2], [0, 1, 2], default=3)
    if "has_ebike" not in out.columns and "y_has_ebike" in out.columns:
        out["has_ebike"] = out["y_has_ebike"]
    if "future_ebikes" not in out.columns and "e_future" in out.columns:
        out["future_ebikes"] = out["e_future"]
    if "future_total_bikes" not in out.columns and "q_future" in out.columns:
        out["future_total_bikes"] = out["q_future"]
    if add_priors or any(column not in out.columns for column in SHIFTED_PRIOR_COLUMNS):
        if "has_ebike" in out.columns:
            out = add_shifted_empirical_priors(out, alpha=alpha)
        else:
            for column in SHIFTED_PRIOR_COLUMNS:
                out[column] = 0.35
    for column in FLOW_LABEL_COLUMNS:
        if column not in out.columns:
            out[column] = 0.0 if column != "example_weight" else 1.0
    if {"e_now", "q_now", "e_future", "q_future"}.issubset(out.columns):
        missing_flow = out[["obs_e_depart", "obs_e_arrive", "obs_c_depart", "obs_c_arrive"]].abs().sum(axis=1) == 0
        if missing_flow.any():
            for idx in out.index[missing_flow]:
                labels = observed_flow_labels_from_states(
                    int(out.at[idx, "e_now"]),
                    int(out.at[idx, "q_now"]),
                    int(out.at[idx, "e_future"]),
                    int(out.at[idx, "q_future"]),
                )
                for key, value in labels.items():
                    out.at[idx, key] = value
    for column in CANONICAL_FEATURE_COLUMNS:
        if column not in out.columns:
            out[column] = 0.0
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
    return out


def base_flow_means(rows: pd.DataFrame) -> np.ndarray:
    rows = ensure_cdg_features(rows)
    horizon_scale = (_finite_series(rows, "horizon_minutes", 10.0).clip(1.0, 30.0) / 10.0)
    current_ebikes = _finite_series(rows, "num_ebikes_available", 0.0).clip(0.0, 80.0)
    total_bikes = _finite_series(rows, "num_bikes_available", 0.0).clip(lower=current_ebikes, upper=80.0)
    capacity = _finite_series(rows, "capacity", 15.0).clip(1.0, 80.0)
    ebike_share = np.where(total_bikes > 0, current_ebikes / total_bikes.clip(lower=1.0), 0.0)
    trip_ebike_share = _finite_series(rows, "trip_ebike_arrival_share_same_hour", 0.25).clip(0.05, 0.95)
    ebike_share = np.maximum(ebike_share, 0.5 * trip_ebike_share)
    trip_depart = (
        _finite_series(rows, "trip_departures_same_hour_10m")
        + _finite_series(rows, "trip_recent_departures_30m") / 3.0
        + _finite_series(rows, "station_hour_dow_depart_rate_shifted", 0.0)
    ).clip(lower=0.0)
    trip_arrive = (
        _finite_series(rows, "trip_arrivals_same_hour_10m")
        + _finite_series(rows, "trip_recent_arrivals_30m") / 3.0
        + _finite_series(rows, "route_inbound_due_horizon")
        + _finite_series(rows, "station_hour_dow_arrive_rate_shifted", 0.0)
    ).clip(lower=0.0)
    horizon = _finite_series(rows, "horizon_minutes", 10.0).round().astype(int)
    inflight_ebike = pd.Series(0.0, index=rows.index)
    inflight_classic = pd.Series(0.0, index=rows.index)
    for minutes in (5, 10, 15, 20):
        mask = horizon == minutes
        if mask.any():
            inflight_ebike.loc[mask] = _finite_series(rows.loc[mask], f"live_inflight_ebike_due_{minutes}m")
            inflight_classic.loc[mask] = _finite_series(rows.loc[mask], f"live_inflight_classic_due_{minutes}m")
    trend = (
        0.50 * _finite_series(rows, "trend_5m")
        + 0.30 * _finite_series(rows, "trend_10m")
        + 0.20 * _finite_series(rows, "trend_15m")
    )
    neighbor_support = _finite_series(rows, "station_neighbor_recent_ebikes") * 0.05
    station_prior = _finite_series(rows, "station_same_hour_rate", 0.35).clip(0.01, 0.99)
    neighbor_prior = _finite_series(rows, "station_neighbor_same_hour_rate", 0.35).clip(0.01, 0.99)
    appearance_prior = np.where(
        current_ebikes <= 0,
        -np.log1p(-(0.12 * station_prior + 0.08 * neighbor_prior).clip(0.0, 0.60)),
        0.0,
    )
    weather_factor = (
        1.0
        - 0.20 * _finite_series(rows, "weather_bad_conditions").clip(0.0, 1.0)
        - 0.025 * _finite_series(rows, "weather_precipitation").clip(0.0, 4.0)
    ).clip(0.55, 1.05)
    commute_factor = 1.0 + 0.08 * _finite_series(rows, "is_commute_hour").clip(0.0, 1.0)
    stale_factor = (1.0 - 0.01 * _finite_series(rows, "data_age_minutes", 0.0).clip(0.0, 30.0)).clip(0.70, 1.0)
    closed_depart = _finite_series(rows, "station_closed_penalty_flag", 0.0).clip(0.0, 1.0)
    closed_return = 1.0 - (1.0 - _finite_series(rows, "is_returning", 1.0).clip(0.0, 1.0))
    e_depart = (
        (trip_depart * np.maximum(0.08, ebike_share) + np.maximum(-trend, 0.0))
        * horizon_scale * weather_factor * commute_factor * (1.0 - closed_depart)
    ).clip(lower=0.0)
    c_depart = (
        trip_depart * np.maximum(0.05, 1.0 - ebike_share)
        * horizon_scale * weather_factor * commute_factor * (1.0 - closed_depart)
    ).clip(lower=0.0)
    e_arrive = (
        (trip_arrive * trip_ebike_share + inflight_ebike + np.maximum(trend, 0.0) + neighbor_support + appearance_prior)
        * horizon_scale * weather_factor * stale_factor * closed_return
    ).clip(lower=0.0)
    c_arrive = (
        (trip_arrive * (1.0 - trip_ebike_share) + inflight_classic)
        * horizon_scale * weather_factor * stale_factor * closed_return
    ).clip(lower=0.0)
    arr = np.column_stack([e_depart, e_arrive, c_depart, c_arrive]).astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=50.0, neginf=0.0)
    max_mu = np.minimum(50.0, 2.0 * capacity.to_numpy(dtype=np.float32)[:, None])
    return np.clip(arr, 1e-4, max_mu).astype(np.float32)


def tabular_matrix(rows: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    frame = ensure_cdg_features(rows)
    values = []
    for column in feature_columns:
        values.append(_finite_series(frame, column, 0.0).to_numpy(dtype=np.float32))
    return np.column_stack(values).astype(np.float32) if values else np.zeros((len(frame), 0), dtype=np.float32)


def station_indices(rows: pd.DataFrame, station_id_to_idx: dict[str, int], unknown_station_idx: int) -> np.ndarray:
    if "station_id" not in rows.columns:
        return np.full(len(rows), int(unknown_station_idx), dtype=np.int64)
    return rows["station_id"].astype(str).map(station_id_to_idx).fillna(int(unknown_station_idx)).astype(np.int64).to_numpy()


if nn is not None:

    class CDGNMIPNet(nn.Module):
        def __init__(
            self,
            *,
            config: CDGNMIPConfig,
            n_tabular: int,
            n_stations: int,
            seq_channels: int,
        ) -> None:
            super().__init__()
            self.config = config
            h = int(config.hidden_dim)
            self.tabular_encoder = nn.Sequential(
                nn.Linear(n_tabular, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(h, h),
                nn.LayerNorm(h),
                nn.GELU(),
            )
            self.temporal_encoder = nn.GRU(
                input_size=seq_channels,
                hidden_size=int(config.sequence_hidden_dim),
                batch_first=True,
            )
            self.station_embedding = nn.Embedding(int(n_stations), int(config.station_embedding_dim))
            self.horizon_encoder = nn.Sequential(
                nn.Linear(3, int(config.horizon_embedding_dim)),
                nn.GELU(),
            )
            fusion_in = h + int(config.sequence_hidden_dim) + int(config.station_embedding_dim) + int(config.horizon_embedding_dim)
            self.pre_graph_fusion = nn.Sequential(
                nn.Linear(fusion_in, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(config.dropout),
            )
            self.relation_types = tuple(config.graph_relation_types)
            self.graph_linears = nn.ModuleDict({relation: nn.Linear(h, h, bias=False) for relation in self.relation_types})
            self.graph_gate = nn.Linear(h, len(self.relation_types))
            self.final_fusion = nn.Sequential(
                nn.Linear(2 * h, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(h, h),
                nn.GELU(),
            )
            self.mu_head = nn.Linear(h, 4)
            self.theta_head = nn.Linear(h, 4)
            self.zeta_head = nn.Linear(h, 4)

        def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
            x_tab = batch["x_tab"]
            x_seq = batch["x_seq"]
            station_idx = batch["station_idx"]
            horizon = batch["horizon_minutes"].float()
            base_mu = batch["base_mu"].float().clamp_min(1e-5)
            capacity = batch["capacity"].float().clamp_min(1.0).unsqueeze(-1)
            tab = self.tabular_encoder(x_tab)
            _seq_out, seq_h = self.temporal_encoder(x_seq)
            seq = seq_h[-1]
            emb = self.station_embedding(station_idx)
            horizon_scaled = torch.stack(
                [
                    horizon / 20.0,
                    torch.sin(2.0 * math.pi * horizon / 60.0),
                    torch.cos(2.0 * math.pi * horizon / 60.0),
                ],
                dim=1,
            )
            h_emb = self.horizon_encoder(horizon_scaled)
            h0 = self.pre_graph_fusion(torch.cat([tab, seq, emb, h_emb], dim=1))
            graph_msg = torch.zeros_like(h0)
            relation_messages = []
            for relation in self.relation_types:
                src_dst = batch.get("edge_index_by_type", {}).get(relation)
                weights = batch.get("edge_weight_by_type", {}).get(relation)
                msg = torch.zeros_like(h0)
                if src_dst is not None and weights is not None and src_dst.numel() > 0:
                    src = src_dst[0].long()
                    dst = src_dst[1].long()
                    transformed = self.graph_linears[relation](h0[src]) * weights.float().unsqueeze(-1)
                    msg.index_add_(0, dst, transformed)
                relation_messages.append(msg)
            if relation_messages:
                stacked = torch.stack(relation_messages, dim=1)
                gates = torch.softmax(self.graph_gate(h0), dim=1).unsqueeze(-1)
                graph_msg = (stacked * gates).sum(dim=1)
            z = self.final_fusion(torch.cat([h0, graph_msg], dim=1))
            residual = torch.tanh(self.mu_head(z)) * 2.0
            mu = torch.exp(torch.log(base_mu + 1e-5) + residual)
            mu = torch.minimum(mu, torch.minimum(torch.full_like(mu, 50.0), 2.0 * capacity)).clamp_min(1e-6)
            theta = F.softplus(self.theta_head(z)).clamp(1e-3, 1000.0)
            zeta = (0.95 * torch.sigmoid(self.zeta_head(z))).clamp(0.0, 0.95)
            return {
                "mu": mu,
                "theta": theta,
                "zeta": zeta,
                "encoder_norm": torch.linalg.vector_norm(z, dim=1),
                "graph_message_norm": torch.linalg.vector_norm(graph_msg, dim=1),
                "temporal_message_norm": torch.linalg.vector_norm(seq, dim=1),
            }

else:
    CDGNMIPNet = None  # type: ignore[assignment]


def _local_graph_for_batch(
    station_idx: np.ndarray,
    horizon_minutes: np.ndarray,
    graph_cache: dict[str, Any] | None,
    *,
    relation_types: tuple[str, ...],
    device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    edge_index_by_type: dict[str, Any] = {}
    edge_weight_by_type: dict[str, Any] = {}
    if torch is None or not graph_cache:
        return edge_index_by_type, edge_weight_by_type
    full_edges = graph_cache.get("edge_index_by_type") or {}
    full_weights = graph_cache.get("edge_weight_by_type") or {}
    if not full_edges:
        return edge_index_by_type, edge_weight_by_type
    station_idx = np.asarray(station_idx, dtype=np.int64)
    horizon_minutes = np.asarray(horizon_minutes, dtype=np.int64)
    by_station: dict[int, list[int]] = {}
    by_station_horizon: dict[tuple[int, int], int] = {}
    for pos, (sid, horizon) in enumerate(zip(station_idx, horizon_minutes)):
        by_station.setdefault(int(sid), []).append(pos)
        by_station_horizon.setdefault((int(sid), int(horizon)), pos)
    for relation in relation_types:
        edges = np.asarray(full_edges.get(relation, np.empty((2, 0), dtype=np.int64)), dtype=np.int64)
        weights = np.asarray(full_weights.get(relation, np.empty((0,), dtype=np.float32)), dtype=np.float32)
        if edges.ndim != 2 or edges.shape[0] != 2 or edges.shape[1] == 0:
            continue
        src_local: list[int] = []
        dst_local: list[int] = []
        local_weights: list[float] = []
        for edge_pos in range(edges.shape[1]):
            src_station = int(edges[0, edge_pos])
            dst_station = int(edges[1, edge_pos])
            dst_positions = by_station.get(dst_station)
            src_positions = by_station.get(src_station)
            if not dst_positions or not src_positions:
                continue
            for dst_pos in dst_positions:
                horizon = int(horizon_minutes[dst_pos])
                src_pos = by_station_horizon.get((src_station, horizon), src_positions[0])
                src_local.append(int(src_pos))
                dst_local.append(int(dst_pos))
                local_weights.append(float(weights[edge_pos] if edge_pos < len(weights) else 1.0))
        if src_local:
            edge_index_by_type[relation] = torch.tensor([src_local, dst_local], dtype=torch.long, device=device)
            edge_weight_by_type[relation] = torch.tensor(local_weights, dtype=torch.float32, device=device)
    return edge_index_by_type, edge_weight_by_type


def make_batch(
    rows: pd.DataFrame,
    *,
    feature_columns: list[str],
    scaler: TabularScaler,
    station_id_to_idx: dict[str, int],
    unknown_station_idx: int,
    config: CDGNMIPConfig,
    graph_cache: dict[str, Any] | None,
    device,
) -> dict[str, Any]:
    frame = ensure_cdg_features(rows, alpha=config.empirical_bayes_alpha)
    x_tab = scaler.transform(tabular_matrix(frame, feature_columns))
    seq, _fallback = sequence_array_from_rows(frame, spec=config.sequence_spec())
    st_idx = station_indices(frame, station_id_to_idx, unknown_station_idx)
    horizon = _finite_series(frame, "horizon_minutes", 10.0).to_numpy(dtype=np.float32)
    base_mu = base_flow_means(frame)
    capacity = _finite_series(frame, "capacity", 15.0).clip(1.0, float(config.max_capacity)).to_numpy(dtype=np.float32)
    edge_index, edge_weight = _local_graph_for_batch(
        st_idx,
        horizon.astype(np.int64),
        graph_cache if config.use_graph else None,
        relation_types=tuple(config.graph_relation_types),
        device=device,
    )
    return {
        "x_tab": torch.tensor(x_tab, dtype=torch.float32, device=device),
        "x_seq": torch.tensor(seq, dtype=torch.float32, device=device),
        "station_idx": torch.tensor(st_idx, dtype=torch.long, device=device),
        "horizon_minutes": torch.tensor(horizon, dtype=torch.float32, device=device),
        "base_mu": torch.tensor(base_mu, dtype=torch.float32, device=device),
        "capacity": torch.tensor(capacity, dtype=torch.float32, device=device),
        "edge_index_by_type": edge_index,
        "edge_weight_by_type": edge_weight,
    }


def _zinb_nll_exact(observed, mu, theta, zeta):
    observed = observed.float().clamp_min(0.0)
    mu = mu.float().clamp_min(1e-6)
    theta = theta.float().clamp_min(1e-3)
    zeta = zeta.float().clamp(0.0, 0.95)
    log_p = torch.log(theta) - torch.log(theta + mu)
    log_1mp = torch.log(mu) - torch.log(theta + mu)
    log_nb = (
        torch.lgamma(observed + theta)
        - torch.lgamma(theta)
        - torch.lgamma(observed + 1.0)
        + theta * log_p
        + observed * log_1mp
    )
    is_zero = observed <= 0.0
    zero_prob = zeta + (1.0 - zeta) * torch.exp(log_nb)
    log_prob = torch.where(is_zero, torch.log(zero_prob.clamp_min(1e-12)), torch.log((1.0 - zeta).clamp_min(1e-12)) + log_nb)
    return -log_prob


def _smooth_min(a, b, beta: float = 4.0):
    """Differentiable approximation of ``min(a, b)``.

    Identity: ``min(a, b) = b - max(b - a, 0) = b - softplus(beta * (b - a)) / beta``.
    For ``beta → ∞`` this converges to the hard min. We use ``beta=4`` which is
    enough resolution for counts in [0, 30] while keeping gradients alive on
    the binding side of the constraint (the stockout regime, where the hard
    ``minimum`` zeroes them out).
    """
    return b - F.softplus(beta * (b - a)) / beta


def _binned_ece_loss(p, y, n_bins: int = 10):
    """Equal-width binned ECE, count-weighted. Differentiable w.r.t. ``p``.

    Bin assignment is detached (bucketize is not differentiable anyway), but the
    bin-wise mean of ``p`` carries gradient. The post-hoc isotonic/Platt
    calibrator already targets exactly this quantity offline — using it during
    training pulls the network toward producing probabilities the calibrator
    needs to correct less.
    """
    p = p.float()
    y = y.float()
    bins = torch.linspace(0.0, 1.0, n_bins + 1, device=p.device)
    idx = torch.bucketize(p.detach(), bins[1:-1]).clamp(0, n_bins - 1)
    ones = torch.ones_like(p)
    total = torch.zeros(n_bins, device=p.device).scatter_add_(0, idx, ones)
    sum_p = torch.zeros(n_bins, device=p.device).scatter_add_(0, idx, p)
    sum_y = torch.zeros(n_bins, device=p.device).scatter_add_(0, idx, y)
    safe = total.clamp_min(1.0)
    gap = ((sum_p - sum_y) / safe).abs()
    w = total / total.sum().clamp_min(1.0)
    return (gap * w * (total > 0).float()).sum()


def cdg_training_loss(batch: dict[str, Any], outputs: dict[str, Any], labels: dict[str, Any], config: CDGNMIPConfig) -> tuple[Any, dict[str, float]]:
    mu = outputs["mu"]
    theta = outputs["theta"]
    zeta = outputs["zeta"]
    e0 = labels["e0"].float()
    q0 = labels["q0"].float()
    cap = labels["capacity"].float().clamp_min(1.0)
    c0 = (q0 - e0).clamp_min(0.0)
    y_has = labels["y_has"].float()
    e_future = labels["e_future"].float().clamp_min(0.0)
    q_future = labels["q_future"].float().clamp_min(0.0)
    weights = labels["weights"].float().clamp(0.05, 5.0)

    exp_e_dep = _smooth_min(mu[:, 0], e0)
    exp_c_dep = _smooth_min(mu[:, 2], c0)
    open_after_dep = (cap - q0 + exp_e_dep + exp_c_dep).clamp_min(0.0)
    exp_e_arr = _smooth_min(mu[:, 1], open_after_dep)
    exp_c_arr = _smooth_min(mu[:, 3], (open_after_dep - exp_e_arr).clamp_min(0.0))
    exp_e = _smooth_min((e0 - exp_e_dep + exp_e_arr).clamp_min(0.0), cap)
    exp_q = _smooth_min((q0 - exp_e_dep - exp_c_dep + exp_e_arr + exp_c_arr).clamp_min(0.0), cap)
    p_has = (1.0 - torch.exp(-exp_e.clamp_min(0.0))).clamp(1e-4, 1.0 - 1e-4)

    bce = F.binary_cross_entropy(p_has, y_has, reduction="none")
    count = F.smooth_l1_loss(exp_e, e_future, reduction="none") + 0.35 * F.smooth_l1_loss(exp_q, q_future, reduction="none")
    flow = (
        _zinb_nll_exact(labels["obs_e_depart"], mu[:, 0], theta[:, 0], zeta[:, 0])
        + _zinb_nll_exact(labels["obs_e_arrive"], mu[:, 1], theta[:, 1], zeta[:, 1])
        + 0.5 * _zinb_nll_exact(labels["obs_c_depart"], mu[:, 2], theta[:, 2], zeta[:, 2])
        + 0.5 * _zinb_nll_exact(labels["obs_c_arrive"], mu[:, 3], theta[:, 3], zeta[:, 3])
    )
    cal = _binned_ece_loss(p_has, y_has, n_bins=10)
    loss = (
        config.loss_count_weight * (count * weights).mean()
        + config.loss_bce_weight * (bce * weights).mean()
        + config.loss_flow_weight * (flow * weights).mean()
        + config.loss_cal_weight * cal
    )
    return loss, {
        "loss": float(loss.detach().cpu().item()),
        "loss_count": float(count.mean().detach().cpu().item()),
        "loss_bce": float(bce.mean().detach().cpu().item()),
        "loss_flow": float(flow.mean().detach().cpu().item()),
        "loss_cal": float(cal.detach().cpu().item()),
    }


def build_training_arrays(
    rows: pd.DataFrame,
    *,
    feature_columns: list[str],
    scaler: TabularScaler | None,
    station_id_to_idx: dict[str, int],
    unknown_station_idx: int,
    config: CDGNMIPConfig,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], TabularScaler]:
    frame = ensure_cdg_features(rows, alpha=config.empirical_bayes_alpha, add_priors=any(c not in rows.columns for c in SHIFTED_PRIOR_COLUMNS))
    x_tab_raw = tabular_matrix(frame, feature_columns)
    fitted_scaler = scaler or TabularScaler.fit(x_tab_raw)
    x_tab = fitted_scaler.transform(x_tab_raw)
    seq, _ = sequence_array_from_rows(frame, spec=config.sequence_spec())
    st_idx = station_indices(frame, station_id_to_idx, unknown_station_idx)
    base_mu = base_flow_means(frame)
    e0 = _finite_series(frame, "num_ebikes_available", 0.0).to_numpy(dtype=np.float32)
    q0 = _finite_series(frame, "num_bikes_available", 0.0).to_numpy(dtype=np.float32)
    e_future = _finite_series(frame, "future_ebikes", np.nan).fillna(_finite_series(frame, "e_future", 0.0)).to_numpy(dtype=np.float32)
    q_future = _finite_series(frame, "future_total_bikes", np.nan).fillna(_finite_series(frame, "q_future", 0.0)).to_numpy(dtype=np.float32)
    y_has = _finite_series(frame, "has_ebike", np.nan).fillna(pd.Series(e_future >= 1, index=frame.index, dtype=float)).to_numpy(dtype=np.float32)
    weights = _finite_series(frame, "example_weight", 1.0).clip(0.05, 5.0).to_numpy(dtype=np.float32)
    arrays = {
        "x_tab": x_tab,
        "x_seq": seq.astype(np.float32),
        "station_idx": st_idx.astype(np.int64),
        "horizon": _finite_series(frame, "horizon_minutes", 10.0).to_numpy(dtype=np.float32),
        "base_mu": base_mu.astype(np.float32),
        "capacity": _finite_series(frame, "capacity", 15.0).clip(1.0, float(config.max_capacity)).to_numpy(dtype=np.float32),
        "e0": e0,
        "q0": q0,
        "e_future": e_future,
        "q_future": q_future,
        "y_has": y_has,
        "obs_e_depart": _finite_series(frame, "obs_e_depart", 0.0).to_numpy(dtype=np.float32),
        "obs_e_arrive": _finite_series(frame, "obs_e_arrive", 0.0).to_numpy(dtype=np.float32),
        "obs_c_depart": _finite_series(frame, "obs_c_depart", 0.0).to_numpy(dtype=np.float32),
        "obs_c_arrive": _finite_series(frame, "obs_c_arrive", 0.0).to_numpy(dtype=np.float32),
        "weights": weights,
    }
    return frame, arrays, fitted_scaler


def train_cdg_net(
    train_rows: pd.DataFrame,
    valid_rows: pd.DataFrame | None,
    *,
    config: CDGNMIPConfig,
    feature_columns: list[str],
    station_id_to_idx: dict[str, int],
    unknown_station_idx: int,
    graph_cache: dict[str, Any] | None,
) -> tuple[Any, TabularScaler, dict[str, Any]]:
    if torch is None or CDGNMIPNet is None:
        raise RuntimeError("PyTorch is required for DG-NISSM CDG-NMIP training")
    started = time.monotonic()
    rng = np.random.default_rng(int(config.seed))
    train = train_rows.copy()
    if len(train) > int(config.max_examples):
        strat_cols = [c for c in ["horizon_minutes", "has_ebike"] if c in train.columns]
        if strat_cols:
            train = (
                train.groupby(strat_cols, group_keys=False)
                .sample(frac=min(1.0, config.max_examples / len(train)), random_state=int(config.seed))
                .head(int(config.max_examples))
            )
        else:
            train = train.sample(n=int(config.max_examples), random_state=int(config.seed))
    train = train.sample(frac=1.0, random_state=int(config.seed)).reset_index(drop=True)
    train_frame, arrays, scaler = build_training_arrays(
        train,
        feature_columns=feature_columns,
        scaler=None,
        station_id_to_idx=station_id_to_idx,
        unknown_station_idx=unknown_station_idx,
        config=config,
    )
    device = choose_device(config.device)
    torch.manual_seed(int(config.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(config.seed))
    net = CDGNMIPNet(
        config=config,
        n_tabular=len(feature_columns),
        n_stations=max(station_id_to_idx.values(), default=-1) + 2,
        seq_channels=len(config.sequence_spec().channels),
    ).to(device)
    dataset = TensorDataset(
        torch.tensor(arrays["x_tab"], dtype=torch.float32),
        torch.tensor(arrays["x_seq"], dtype=torch.float32),
        torch.tensor(arrays["station_idx"], dtype=torch.long),
        torch.tensor(arrays["horizon"], dtype=torch.float32),
        torch.tensor(arrays["base_mu"], dtype=torch.float32),
        torch.tensor(arrays["capacity"], dtype=torch.float32),
        torch.tensor(arrays["e0"], dtype=torch.float32),
        torch.tensor(arrays["q0"], dtype=torch.float32),
        torch.tensor(arrays["e_future"], dtype=torch.float32),
        torch.tensor(arrays["q_future"], dtype=torch.float32),
        torch.tensor(arrays["y_has"], dtype=torch.float32),
        torch.tensor(arrays["obs_e_depart"], dtype=torch.float32),
        torch.tensor(arrays["obs_e_arrive"], dtype=torch.float32),
        torch.tensor(arrays["obs_c_depart"], dtype=torch.float32),
        torch.tensor(arrays["obs_c_arrive"], dtype=torch.float32),
        torch.tensor(arrays["weights"], dtype=torch.float32),
    )
    loader = DataLoader(
        dataset,
        batch_size=max(32, int(config.batch_size)),
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    optimizer = torch.optim.AdamW(net.parameters(), lr=float(config.lr), weight_decay=float(config.weight_decay))
    best_state: dict[str, Any] | None = None
    best_loss = float("inf")
    stale_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(max(1, int(config.epochs))):
        net.train()
        epoch_losses: list[float] = []
        for batch_tensors in loader:
            (
                x_tab,
                x_seq,
                st_idx,
                horizon,
                base_mu,
                capacity,
                e0,
                q0,
                e_future,
                q_future,
                y_has,
                obs_e_depart,
                obs_e_arrive,
                obs_c_depart,
                obs_c_arrive,
                weights,
            ) = [tensor.to(device, non_blocking=False) for tensor in batch_tensors]
            edge_index, edge_weight = _local_graph_for_batch(
                st_idx.detach().cpu().numpy(),
                horizon.detach().cpu().numpy().astype(np.int64),
                graph_cache if config.use_graph else None,
                relation_types=tuple(config.graph_relation_types),
                device=device,
            )
            batch = {
                "x_tab": x_tab,
                "x_seq": x_seq,
                "station_idx": st_idx,
                "horizon_minutes": horizon,
                "base_mu": base_mu,
                "capacity": capacity,
                "edge_index_by_type": edge_index,
                "edge_weight_by_type": edge_weight,
            }
            labels = {
                "e0": e0,
                "q0": q0,
                "capacity": capacity,
                "e_future": e_future,
                "q_future": q_future,
                "y_has": y_has,
                "obs_e_depart": obs_e_depart,
                "obs_e_arrive": obs_e_arrive,
                "obs_c_depart": obs_c_depart,
                "obs_c_arrive": obs_c_arrive,
                "weights": weights,
            }
            optimizer.zero_grad(set_to_none=True)
            outputs = net(batch)
            loss, parts = cdg_training_loss(batch, outputs, labels, config)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 2.0)
            optimizer.step()
            epoch_losses.append(parts["loss"])
        epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else float("inf")
        history.append({"epoch": float(epoch), "loss": epoch_loss})
        if epoch_loss < best_loss - 1e-4:
            best_loss = epoch_loss
            best_state = {key: value.detach().cpu().clone() for key, value in net.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= int(config.early_stopping_patience):
            break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    metrics = {
        "training_seconds": float(time.monotonic() - started),
        "training_device": str(device),
        "training_loss": float(best_loss),
        "epochs_completed": len(history),
        "loss_history": history,
        "n_train_after_subsample": int(len(train_frame)),
    }
    return net.cpu().eval(), scaler, metrics


def predict_intensity_parameters(
    net,
    rows: pd.DataFrame,
    *,
    feature_columns: list[str],
    scaler: TabularScaler,
    station_id_to_idx: dict[str, int],
    unknown_station_idx: int,
    config: CDGNMIPConfig,
    graph_cache: dict[str, Any] | None,
    batch_size: int | None = None,
    device_name: str | None = None,
) -> dict[str, np.ndarray]:
    if torch is None:
        raise RuntimeError("PyTorch is required for DG-NISSM prediction")
    if rows.empty:
        empty = np.empty((0, 4), dtype=np.float32)
        return {
            "mu": empty,
            "theta": empty,
            "zeta": empty,
            "encoder_norm": np.empty((0,), dtype=np.float32),
            "graph_message_norm": np.empty((0,), dtype=np.float32),
            "temporal_message_norm": np.empty((0,), dtype=np.float32),
        }
    device = choose_device(device_name or config.runtime_device)
    net = net.to(device).eval()
    frame = ensure_cdg_features(rows, alpha=config.empirical_bayes_alpha)
    bs = max(1, int(batch_size or min(len(frame), 8192)))
    outputs: dict[str, list[np.ndarray]] = {
        "mu": [],
        "theta": [],
        "zeta": [],
        "encoder_norm": [],
        "graph_message_norm": [],
        "temporal_message_norm": [],
    }
    with torch.no_grad():
        for start in range(0, len(frame), bs):
            batch_rows = frame.iloc[start : start + bs]
            batch = make_batch(
                batch_rows,
                feature_columns=feature_columns,
                scaler=scaler,
                station_id_to_idx=station_id_to_idx,
                unknown_station_idx=unknown_station_idx,
                config=config,
                graph_cache=graph_cache,
                device=device,
            )
            pred = net(batch)
            for key in outputs:
                outputs[key].append(pred[key].detach().cpu().numpy().astype(np.float32))
    net.cpu()
    return {key: np.concatenate(value, axis=0) if value else np.empty((0,), dtype=np.float32) for key, value in outputs.items()}


def logit_clip(p: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(p, dtype=float)
    arr = np.clip(arr, 1e-6, 1.0 - 1e-6)
    return np.log(arr / (1.0 - arr))


def sigmoid_np(x: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(arr, -35.0, 35.0)))


def fit_zero_calibrator(raw: pd.DataFrame, valid_rows: pd.DataFrame, *, min_segment_n: int = 200) -> dict[str, Any]:
    if raw.empty or valid_rows.empty or "p_zero" not in raw.columns:
        return {"segments": {}, "global": {"a": 1.0, "b": 0.0}, "fitted": False}
    y_zero = (_finite_series(valid_rows, "future_ebikes", np.nan).fillna(_finite_series(valid_rows, "e_future", 0.0)) <= 0).astype(int)
    p0 = pd.to_numeric(raw["p_zero"], errors="coerce").fillna(0.5).clip(1e-5, 1 - 1e-5)
    data = pd.DataFrame({
        "logit": logit_clip(p0.to_numpy(dtype=float)),
        "y": y_zero.to_numpy(dtype=int),
        "horizon": _finite_series(valid_rows, "horizon_minutes", 10.0).round().astype(int).to_numpy(),
        "current_zero": (_finite_series(valid_rows, "num_ebikes_available", 0.0) <= 0).astype(int).to_numpy(),
    })

    def fit_affine(segment: pd.DataFrame) -> dict[str, float] | None:
        if len(segment) < 20 or segment["y"].nunique() < 2:
            return None
        try:
            from sklearn.linear_model import LogisticRegression

            clf = LogisticRegression(C=10.0, solver="lbfgs")
            clf.fit(segment[["logit"]], segment["y"])
            return {"a": float(clf.coef_[0, 0]), "b": float(clf.intercept_[0]), "n": int(len(segment))}
        except Exception:
            pred = float(sigmoid_np(segment["logit"]).mean())
            obs = float(segment["y"].mean())
            return {"a": 1.0, "b": float(logit_clip(obs) - logit_clip(pred)), "n": int(len(segment))}

    global_fit = fit_affine(data) or {"a": 1.0, "b": 0.0, "n": int(len(data))}
    segments: dict[str, dict[str, float]] = {}
    for (horizon, current_zero), group in data.groupby(["horizon", "current_zero"]):
        if len(group) < int(min_segment_n):
            continue
        fit = fit_affine(group)
        if fit is not None:
            segments[f"h{int(horizon)}_z{int(current_zero)}"] = fit
    return {"segments": segments, "global": global_fit, "fitted": True}


def calibrate_joint_zero(
    joint: np.ndarray,
    *,
    calibrator: dict[str, Any] | None,
    horizon_minutes: int,
    current_ebikes: int,
) -> tuple[np.ndarray, float]:
    state = np.asarray(joint, dtype=float).copy()
    state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
    total = float(state.sum())
    if total <= 0.0:
        return state, 0.0
    state /= total
    if not calibrator or not calibrator.get("fitted"):
        return state, 0.0
    p0 = float(state[0, :].sum()) if state.size else 1.0
    key = f"h{int(round(horizon_minutes))}_z{int(current_ebikes <= 0)}"
    params = (calibrator.get("segments") or {}).get(key) or calibrator.get("global") or {"a": 1.0, "b": 0.0}
    p0_cal = float(sigmoid_np(float(params.get("a", 1.0)) * float(logit_clip(p0)) + float(params.get("b", 0.0))))
    p0_cal = float(np.clip(p0_cal, 1e-6, 1.0 - 1e-6))
    if p0 <= 1e-12 or p0 >= 1.0 - 1e-12:
        return state, 0.0
    out = state.copy()
    out[0, :] *= p0_cal / p0
    out[1:, :] *= (1.0 - p0_cal) / (1.0 - p0)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    out = out / max(1e-12, float(out.sum()))
    return out, float(p0_cal - p0)


def rollout_from_parameters(
    *,
    capacity: int,
    current_ebikes: int,
    current_total_bikes: int,
    horizon_minutes: int,
    mu: np.ndarray,
    theta: np.ndarray,
    zeta: np.ndarray,
    config: CDGNMIPConfig,
    is_renting: bool = True,
    is_returning: bool = True,
) -> inventory_dp.InventoryRolloutResult:
    if not bool(config.exact_inventory_dp):
        return fast_inventory_rollout_from_parameters(
            capacity=capacity,
            current_ebikes=current_ebikes,
            current_total_bikes=current_total_bikes,
            mu=mu,
            theta=theta,
            zeta=zeta,
            config=config,
            is_renting=is_renting,
            is_returning=is_returning,
        )
    steps = int(config.rollout_steps(horizon_minutes))
    intensities = []
    mu = np.asarray(mu, dtype=float)
    theta = np.asarray(theta, dtype=float)
    zeta = np.asarray(zeta, dtype=float)
    for _ in range(steps):
        intensities.append({
            "mu_e_depart": float(mu[0] / steps),
            "mu_e_arrive": float(mu[1] / steps),
            "mu_c_depart": float(mu[2] / steps),
            "mu_c_arrive": float(mu[3] / steps),
            "ebike_depart_theta": float(theta[0]),
            "ebike_arrive_theta": float(theta[1]),
            "classic_depart_theta": float(theta[2]),
            "classic_arrive_theta": float(theta[3]),
            "ebike_depart_zero_inflation": float(zeta[0]),
            "ebike_arrive_zero_inflation": float(zeta[1]),
            "classic_depart_zero_inflation": float(zeta[2]),
            "classic_arrive_zero_inflation": float(zeta[3]),
            "is_renting": float(bool(is_renting)),
            "is_returning": float(bool(is_returning)),
        })
    return inventory_dp.rollout_inventory_distribution_multistep(
        capacity=capacity,
        current_ebikes=current_ebikes,
        current_total_bikes=current_total_bikes,
        intensity_sequence=intensities,
        max_capacity=int(config.max_capacity),
        return_joint=True,
    )


def _discrete_moment_pmf(mean: float, variance: float, max_k: int) -> np.ndarray:
    max_k = max(0, int(max_k))
    counts = np.arange(max_k + 1, dtype=float)
    mean = float(np.clip(mean, 0.0, float(max_k)))
    variance = max(float(variance), 0.20)
    weights = np.exp(-0.5 * ((counts - mean) ** 2) / variance)
    if weights.sum() <= 0.0 or not np.isfinite(weights).all():
        weights = np.zeros(max_k + 1, dtype=float)
        weights[int(round(mean))] = 1.0
    return weights / weights.sum()


def fast_inventory_rollout_from_parameters(
    *,
    capacity: int,
    current_ebikes: int,
    current_total_bikes: int,
    mu: np.ndarray,
    theta: np.ndarray,
    zeta: np.ndarray,
    config: CDGNMIPConfig,
    is_renting: bool = True,
    is_returning: bool = True,
) -> inventory_dp.InventoryRolloutResult:
    """Fast finite-state approximation for request-time scoring.

    It computes constrained successful-flow moments, then projects them into a
    normalized joint PMF over the valid state space ``0 <= E <= Q <= K``. This
    keeps API outputs inventory-consistent without the heavier event-count DP.
    """
    cap = max(1, min(int(config.max_capacity), int(capacity)))
    e0 = max(0, min(cap, int(current_ebikes)))
    q0 = max(e0, min(cap, int(current_total_bikes)))
    c0 = max(0, q0 - e0)
    mu = np.nan_to_num(np.asarray(mu, dtype=float), nan=0.0, posinf=50.0, neginf=0.0).clip(0.0, min(50.0, 2.0 * cap))
    theta = np.nan_to_num(np.asarray(theta, dtype=float), nan=1.0, posinf=1000.0, neginf=1e-3).clip(1e-3, 1000.0)
    zeta = np.nan_to_num(np.asarray(zeta, dtype=float), nan=0.0, posinf=0.95, neginf=0.0).clip(0.0, 0.95)
    if not is_renting:
        mu[0] = 0.0
        mu[2] = 0.0
    if not is_returning:
        mu[1] = 0.0
        mu[3] = 0.0
    e_dep = min(float(mu[0]), float(e0))
    c_dep = min(float(mu[2]), float(c0))
    open_after_depart = max(0.0, float(cap - q0) + e_dep + c_dep)
    e_arr = min(float(mu[1]), open_after_depart)
    c_arr = min(float(mu[3]), max(0.0, open_after_depart - e_arr))
    expected_e = float(np.clip(e0 - e_dep + e_arr, 0.0, cap))
    expected_q = float(np.clip(q0 - e_dep - c_dep + e_arr + c_arr, expected_e, cap))
    # Closed-form per-channel ZINB variance: Var = (1-ζ)·μ·(1 + μ/θ + ζ·μ).
    # Sum across the channels that feed each state component, assuming inter-channel
    # independence (an existing model assumption — see Weakness #1).
    zinb_var = (1.0 - zeta) * mu * (1.0 + mu / theta + zeta * mu)
    zinb_var = np.maximum(zinb_var, 0.0)
    e_var = max(0.20, float(zinb_var[0] + zinb_var[1]))
    q_var = max(0.20, float(zinb_var[0] + zinb_var[1] + zinb_var[2] + zinb_var[3]))
    e_pmf = _discrete_moment_pmf(expected_e, e_var, cap)
    q_pmf = _discrete_moment_pmf(expected_q, q_var, cap)
    joint = np.outer(e_pmf, q_pmf)
    valid_mask = np.zeros_like(joint, dtype=bool)
    for e in range(cap + 1):
        valid_mask[e, e : cap + 1] = True
    joint = np.where(valid_mask, joint, 0.0)
    if joint.sum() <= 0.0:
        joint[:] = 0.0
        joint[e0, q0] = 1.0
    joint = joint / joint.sum()
    e_pmf = joint.sum(axis=1)
    q_pmf = joint.sum(axis=0)
    counts = np.arange(cap + 1, dtype=float)
    p_zero = float(e_pmf[0])
    attempted_arrivals = float(mu[1] + mu[3])
    p_dock_constrained = 0.0
    if attempted_arrivals > 1e-9:
        p_dock_constrained = float(np.clip((attempted_arrivals - open_after_depart) / (attempted_arrivals + 1.0), 0.0, 1.0))
    # TODO: emit p_has_open_dock as a first-class prediction target — the
    # parking-side dual of p_has_ebike. Riders returning a bike need a PMF
    # over open-dock count at horizon t, just as pickup planners use p_has_ebike.
    # p_dock_constrained above is a flow-constraint heuristic, not the same quantity.
    return inventory_dp.InventoryRolloutResult(
        p_has_ebike=float(1.0 - p_zero),
        p_zero=p_zero,
        expected_ebikes=float(np.dot(counts, e_pmf)),
        expected_total_bikes=float(np.dot(counts, q_pmf)),
        p_count_ebikes=inventory_dp.collapse_count_distribution(e_pmf),
        p_count_total=inventory_dp.collapse_count_distribution(q_pmf),
        p_capacity_violation=0.0,
        p_dock_constrained_arrival=p_dock_constrained,
        expected_ebike_departures=float(e_dep),
        expected_classic_departures=float(c_dep),
        expected_ebike_arrivals=float(e_arr),
        expected_classic_arrivals=float(c_arr),
        p_joint_e_q=joint,
    )


def distribution_from_joint(joint: np.ndarray) -> dict[str, Any]:
    state = np.asarray(joint, dtype=float)
    state = state / max(1e-12, float(state.sum()))
    e_pmf = state.sum(axis=1)
    q_pmf = state.sum(axis=0)
    counts = np.arange(state.shape[0], dtype=float)
    p_zero = float(e_pmf[0]) if e_pmf.size else 1.0
    return {
        "p_has_ebike": float(1.0 - p_zero),
        "p_zero": p_zero,
        "expected_ebikes": float(np.dot(counts[: e_pmf.size], e_pmf)),
        "expected_total_bikes": float(np.dot(counts[: q_pmf.size], q_pmf)),
        "p_count_ebikes": inventory_dp.collapse_count_distribution(e_pmf),
        "p_count_total": inventory_dp.collapse_count_distribution(q_pmf),
    }


def config_to_dict(config: CDGNMIPConfig) -> dict[str, Any]:
    data = asdict(config)
    data["graph_relation_types"] = list(config.graph_relation_types)
    return data
