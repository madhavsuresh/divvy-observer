"""MacFlow-NISSM-lite: Mac-trainable approximation of a citywide bike-flow world model.

Architecture (≤3M params):
  - numeric tabular features (current inventory, calendar, recent churn, weather)
  - station / community / role / horizon embeddings
  - compact 2-layer GELU MLP trunk (hidden_dim=64 by default)
  - four ZINB intensity heads (mu, theta, zeta) for {e_depart, e_arrive, c_depart, c_arrive}
  - inventory rollout reuses ``cdg_nmip.fast_inventory_rollout_from_parameters`` to
    guarantee capacity-consistent PMFs (0 ≤ E ≤ Q ≤ K).

Training supervises ZINB NLL on observed flow channels and BCE on a fast
differentiable ``p_has_ebike`` surrogate; the actual prediction-time PMF
comes from the exact rollout, ensuring inventory-consistent outputs.

Not implemented here on purpose: dense OD attention, large sequence
encoders, multi-GPU training, dense N×N tensors. See the plan doc for the
list of intentional non-goals.
"""

from __future__ import annotations

import json
import logging
import math
import os
import pickle
import time
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - torch is a hard dep but defensive
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

from . import inventory_dp
from .cdg_nmip import (
    CDG_DEBUG_COLUMNS,
    CDG_DIAGNOSTIC_COLUMNS,
    CDG_REQUIRED_OUTPUT_COLUMNS,
    CDGNMIPConfig,
    TabularScaler,
    calibrate_joint_zero,
    distribution_from_joint,
    fast_inventory_rollout_from_parameters,
    fit_zero_calibrator,
)
from .macflow_features import (
    MACFLOW_FEATURE_COLUMNS,
    NEUTRAL_DEFAULTS,
    CommunityRuntimeDefaults,
    apply_runtime_defaults,
    attach_macflow_features,
    build_community_runtime_defaults,
    build_station_aggregates,
)
from .mobility_partitions import (
    Partition,
    ROLE_TO_INT,
    ROLE_UNKNOWN,
    make_random_partition,
)


log = logging.getLogger(__name__)


NUMERIC_FEATURE_COLUMNS: list[str] = [
    "current_ebikes_clipped",
    "current_total_bikes_clipped",
    "docks_available_clipped",
    "ebike_share_of_bikes",
    "dock_availability_fraction",
    "hour_sin",
    "hour_cos",
    "dow",
    "is_weekend",
    "month_sin",
    "month_cos",
    "day_of_year_sin",
    "day_of_year_cos",
    "is_commute_hour",
    "trend_5m",
    "trend_10m",
    "trend_15m",
    "churn_rate",
    "station_same_hour_rate",
    "nearby_same_hour_rate",
    "station_neighbor_same_hour_rate",
    "horizon_minutes",
    "status_age_minutes",
]

MACFLOW_NUMERIC_COLUMNS: list[str] = NUMERIC_FEATURE_COLUMNS + [
    "boundary_score",
    "gateway_score",
    "inbound_internal_share",
    "outbound_internal_share",
    "community_recent_ebikes_mean",
    "community_recent_zero_share",
    "community_recent_churn",
    "community_recent_full_share",
    "community_recent_docks_mean",
    "neighbor_community_ebikes_mean",
    "neighbor_community_zero_share",
    "neighbor_community_churn",
    "od_departure_pressure_same_community",
    "od_arrival_pressure_same_community",
    "od_departure_pressure_external",
    "od_arrival_pressure_external",
    "community_exchange_in_pressure",
    "community_exchange_out_pressure",
]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _finite_float(value, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _safe_int(value, default: int = 0) -> int:
    return int(round(_finite_float(value, float(default))))


def select_device(preferred: str = "auto") -> str:
    """Return the device name the model should use.

    We trust ``torch.backends.mps.is_available()`` only; the real fallback
    happens inside fit() via a try/except around ``model.to(device)``.
    """

    if not _TORCH_AVAILABLE:
        return "cpu"
    pref = str(preferred or "auto").lower()
    if pref == "cpu":
        return "cpu"
    if pref in ("auto", "mps"):
        try:
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                return "mps"
        except Exception:
            pass
        if pref == "mps":
            return "cpu"
    if pref == "cuda":
        try:
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"
    return "cpu"


@dataclass
class MacFlowNISSMLiteConfig:
    """Compact config for a Mac-trainable PoC. Defaults aim for ≤3M params."""

    hidden_dim: int = 64
    station_embedding_dim: int = 8
    community_embedding_dim: int = 4
    horizon_embedding_dim: int = 4
    role_embedding_dim: int = 2
    n_layers: int = 2
    dropout: float = 0.05
    partition_mode: str = "full"
    device: str = "auto"
    runtime_device: str = "cpu"
    max_examples: int = 200_000
    epochs: int = 5
    batch_size_mps: int = 512
    batch_size_cpu: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    calibrate: bool = True
    early_stopping_patience: int = 2
    max_capacity: int = 80
    min_train_examples: int = 100
    min_positive_examples: int = 100
    min_zero_future_examples: int = 100
    min_valid_examples: int = 50
    n_communities_cap: int = 128
    n_roles: int = 4
    n_horizon_buckets: int = 8
    intensity_l2_weight: float = 0.01
    count_loss_weight: float = 0.5
    bce_loss_weight: float = 1.0
    partition_lookback_days: int = 30
    aggregate_lookback_minutes: int = 120
    model_version: str = "macflow-nissm-lite-v1"


def _config_from_dict(value: dict | MacFlowNISSMLiteConfig | None) -> MacFlowNISSMLiteConfig:
    if isinstance(value, MacFlowNISSMLiteConfig):
        return value
    data = dict(value or {})
    allowed = {f.name for f in fields(MacFlowNISSMLiteConfig)}
    return MacFlowNISSMLiteConfig(**{key: data[key] for key in data if key in allowed})


def _config_to_dict(config: MacFlowNISSMLiteConfig) -> dict[str, Any]:
    return asdict(config)


# ----------------------------------------------------------------------------
# Neural net
# ----------------------------------------------------------------------------


if _TORCH_AVAILABLE:

    class _MacFlowNet(nn.Module):  # pragma: no cover (covered indirectly via fit())
        """Compact intensity model: numeric + embeddings → 4-channel ZINB heads."""

        def __init__(
            self,
            *,
            n_numeric: int,
            n_stations: int,
            n_communities: int,
            n_roles: int,
            n_horizons: int,
            hidden_dim: int,
            station_dim: int,
            community_dim: int,
            role_dim: int,
            horizon_dim: int,
            dropout: float,
            n_layers: int,
        ) -> None:
            super().__init__()
            self.station_emb = nn.Embedding(max(1, n_stations), station_dim)
            self.community_emb = nn.Embedding(max(1, n_communities), community_dim)
            self.role_emb = nn.Embedding(max(1, n_roles), role_dim)
            self.horizon_emb = nn.Embedding(max(1, n_horizons), horizon_dim)
            in_dim = n_numeric + station_dim + community_dim + role_dim + horizon_dim
            layers: list[nn.Module] = []
            current = in_dim
            for _ in range(max(1, int(n_layers))):
                layers.append(nn.Linear(current, hidden_dim))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(float(dropout)))
                current = hidden_dim
            self.trunk = nn.Sequential(*layers)
            self.mu_head = nn.Linear(current, 4)
            self.theta_head = nn.Linear(current, 4)
            self.zeta_head = nn.Linear(current, 4)

        def forward(
            self,
            *,
            numeric: torch.Tensor,
            station_idx: torch.Tensor,
            community_idx: torch.Tensor,
            role_idx: torch.Tensor,
            horizon_idx: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            station_e = self.station_emb(station_idx)
            community_e = self.community_emb(community_idx)
            role_e = self.role_emb(role_idx)
            horizon_e = self.horizon_emb(horizon_idx)
            x = torch.cat([numeric, station_e, community_e, role_e, horizon_e], dim=-1)
            h = self.trunk(x)
            # Softplus bound intensities; clamp to a safe range to avoid overflow on MPS.
            mu = torch.nn.functional.softplus(self.mu_head(h)).clamp(min=1e-4, max=50.0)
            theta = torch.nn.functional.softplus(self.theta_head(h)).clamp(min=1e-2, max=200.0) + 1.0
            zeta = torch.sigmoid(self.zeta_head(h)).clamp(min=1e-4, max=0.95)
            # ``encoder_norm`` mirrors the cdg_nmip debug semantics so downstream
            # consumers that expect that column keep working.
            return {
                "mu": mu,
                "theta": theta,
                "zeta": zeta,
                "encoder_norm": h.norm(dim=-1),
            }

else:  # pragma: no cover - torch always installed in CI

    class _MacFlowNet:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("PyTorch is required for MacFlow-NISSM-lite")


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _bucket_horizon(horizon_minutes: int | float, *, max_buckets: int) -> int:
    h = _safe_int(horizon_minutes, 10)
    return int(min(max(0, h // 5), max_buckets - 1))


def _build_feature_matrix(rows: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    if rows.empty:
        return np.zeros((0, len(feature_columns)), dtype=np.float32)
    out = np.zeros((len(rows), len(feature_columns)), dtype=np.float32)
    for j, col in enumerate(feature_columns):
        if col in rows.columns:
            out[:, j] = pd.to_numeric(rows[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    return out


def _build_station_idx(rows: pd.DataFrame, mapping: dict[str, int], unknown_idx: int) -> np.ndarray:
    if rows.empty:
        return np.zeros(0, dtype=np.int64)
    return rows["station_id"].astype(str).map(lambda s: mapping.get(s, unknown_idx)).to_numpy(dtype=np.int64)


def _build_community_idx(rows: pd.DataFrame, cap: int) -> np.ndarray:
    if rows.empty:
        return np.zeros(0, dtype=np.int64)
    return rows["community_id"].fillna(0).astype(int).clip(0, max(0, cap - 1)).to_numpy(dtype=np.int64)


def _build_role_idx(rows: pd.DataFrame, n_roles: int) -> np.ndarray:
    if rows.empty:
        return np.zeros(0, dtype=np.int64)
    return rows["role_id"].fillna(ROLE_TO_INT[ROLE_UNKNOWN]).astype(int).clip(0, n_roles - 1).to_numpy(dtype=np.int64)


def _build_horizon_idx(rows: pd.DataFrame, n_buckets: int) -> np.ndarray:
    if rows.empty:
        return np.zeros(0, dtype=np.int64)
    return rows["horizon_minutes"].fillna(10).astype(float).map(
        lambda h: _bucket_horizon(h, max_buckets=n_buckets)
    ).to_numpy(dtype=np.int64)


def _zinb_log_pmf_at_zero(mu: "torch.Tensor", theta: "torch.Tensor", zeta: "torch.Tensor") -> "torch.Tensor":
    """log P_ZINB(0; mu, theta, zeta)."""

    log_zinb_zero = theta * (torch.log(theta) - torch.log(theta + mu))
    p_zero = zeta + (1.0 - zeta) * torch.exp(log_zinb_zero)
    return torch.log(p_zero.clamp(min=1e-12))


def _zinb_nll(observed: "torch.Tensor", mu: "torch.Tensor", theta: "torch.Tensor", zeta: "torch.Tensor") -> "torch.Tensor":
    """Negative log-likelihood of a ZINB distribution for non-negative integer ``observed``."""

    eps = 1e-12
    mu = mu.clamp(min=eps)
    theta = theta.clamp(min=1e-2)
    zeta = zeta.clamp(min=1e-6, max=1.0 - 1e-6)
    # P(k=0) = zeta + (1-zeta) * (theta/(theta+mu))**theta
    log_zinb_zero = theta * (torch.log(theta) - torch.log(theta + mu))
    log_p_zero = torch.log(zeta + (1.0 - zeta) * torch.exp(log_zinb_zero) + eps)
    # P(k>=1) NB log-prob (Anscombe): lgamma(k+theta) - lgamma(theta) - lgamma(k+1)
    # + theta log(theta/(theta+mu)) + k log(mu/(theta+mu))
    log_nb = (
        torch.lgamma(observed + theta)
        - torch.lgamma(theta)
        - torch.lgamma(observed + 1.0)
        + theta * (torch.log(theta) - torch.log(theta + mu))
        + observed * (torch.log(mu) - torch.log(theta + mu))
    )
    log_p_pos = torch.log(1.0 - zeta + eps) + log_nb
    nll = torch.where(observed <= 0, -log_p_zero, -log_p_pos)
    return nll


def _surrogate_p_has_ebike(
    mu: "torch.Tensor",
    theta: "torch.Tensor",
    zeta: "torch.Tensor",
    current_ebikes: "torch.Tensor",
    capacity: "torch.Tensor",
) -> "torch.Tensor":
    """Differentiable surrogate for P(E_final >= 1).

    Decomposition (assuming flow channels are independent at the intensity level):
      P(E_final = 0) ≈ P(arrivals_e = 0) * P(departures_e >= current_ebikes)
    For ``current_ebikes == 0`` the second term equals 1.
    We approximate ``P(departures_e >= e0)`` with a soft step centred at ``e0``
    based on the expected residual capacity. This is intentionally an
    approximation — exact PMF is used at prediction time via the rollout.
    """

    mu_e_arr = mu[..., 1]
    theta_e_arr = theta[..., 1]
    zeta_e_arr = zeta[..., 1]
    mu_e_dep = mu[..., 0]
    p_arrive_zero = zeta_e_arr + (1.0 - zeta_e_arr) * torch.exp(
        theta_e_arr * (torch.log(theta_e_arr.clamp(min=1e-6)) - torch.log((theta_e_arr + mu_e_arr).clamp(min=1e-6)))
    )
    p_arrive_zero = p_arrive_zero.clamp(min=1e-6, max=1.0 - 1e-6)
    # P(depart >= e0): use the Poisson-survival approximation via a smooth sigmoid
    # centered on (mu_e_dep - e0).
    depart_gap = mu_e_dep - current_ebikes
    p_depart_clears = torch.sigmoid(2.0 * depart_gap)
    # When e0 == 0, the depart term is irrelevant.
    is_zero_now = (current_ebikes <= 0).float()
    p_zero_final = p_arrive_zero * (is_zero_now + (1.0 - is_zero_now) * p_depart_clears)
    return (1.0 - p_zero_final).clamp(min=1e-6, max=1.0 - 1e-6)


# ----------------------------------------------------------------------------
# Model class
# ----------------------------------------------------------------------------


class MacFlowNISSMLite:
    """Mac-trainable PoC of the citywide bike-flow world model."""

    model_key = "macflow_nissm_lite"
    model_family = "macflow_nissm_lite"
    model_version = "macflow-nissm-lite-v1"
    method = "macflow_nissm_lite_untrained"

    def __init__(self, config: MacFlowNISSMLiteConfig | dict | None = None) -> None:
        self.config = _config_from_dict(config)
        self.feature_columns: list[str] = list(MACFLOW_NUMERIC_COLUMNS)
        self.tabular_scaler: TabularScaler | None = None
        self.station_id_to_idx: dict[str, int] = {}
        self.unknown_station_idx: int = 0
        self.partition: Partition | None = None
        self.community_runtime_defaults: CommunityRuntimeDefaults | None = None
        self.calibrator: dict[str, Any] | None = None
        self.metrics: dict[str, Any] = {}
        self.trained_at: datetime = _utc_now()
        self.training_examples: int = 0
        self.training_positive: int = 0
        self.net: Any | None = None
        self.trained: bool = False
        self.model_warning: str | None = "No trained MacFlow-NISSM-lite artifact loaded."
        self.method = "macflow_nissm_lite_untrained"
        self.partition_mode: str = self.config.partition_mode
        self._rollout_config: CDGNMIPConfig = CDGNMIPConfig(
            max_capacity=int(self.config.max_capacity),
            exact_inventory_dp=False,
        )

    # ------------------------------ public API ------------------------------

    def fit(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame | None = None,
        graph_cache: Any | None = None,
    ) -> "MacFlowNISSMLite":
        """Train the compact net.

        ``graph_cache`` may be:
          * a :class:`Partition` — use directly;
          * a dict with key ``"partition"`` — use the contained Partition;
          * any other / None — build a single-community fallback partition from train station IDs.
        """

        partition = self._resolve_partition(graph_cache, train_df)
        self.partition = partition

        partition_mode = self.partition_mode = self.config.partition_mode
        if partition_mode == "random" and partition is not None and partition.station_to_community:
            partition = make_random_partition(partition, seed=int(self.config.seed))
            self.partition = partition

        station_aggregates = _maybe_station_aggregates(train_df)
        train = attach_macflow_features(
            train_df,
            partition,
            station_aggregates=station_aggregates,
            partition_mode=partition_mode,
            lookback_minutes=int(self.config.aggregate_lookback_minutes),
        )
        valid: pd.DataFrame
        if valid_df is not None and len(valid_df) > 0:
            valid_aggregates = _maybe_station_aggregates(valid_df)
            valid = attach_macflow_features(
                valid_df,
                partition,
                station_aggregates=valid_aggregates,
                partition_mode=partition_mode,
                lookback_minutes=int(self.config.aggregate_lookback_minutes),
            )
        else:
            valid = pd.DataFrame()

        reason = self._quality_gate(train, valid)
        self.training_examples = int(len(train))
        y_train = pd.to_numeric(train.get("has_ebike", train.get("y_has_ebike", pd.Series(dtype=float))), errors="coerce").fillna(0).astype(int)
        self.training_positive = int(y_train.sum()) if len(train) else 0

        if reason is not None:
            self.trained = False
            self.method = "macflow_nissm_lite_skipped_insufficient_data"
            self.model_warning = reason
            self.metrics = {
                "status": "skipped",
                "reason": reason,
                "n_train": int(len(train)),
                "n_valid": int(len(valid)),
                "method": self.method,
            }
            return self

        # Subsample to max_examples if needed.
        if len(train) > int(self.config.max_examples):
            train = train.sample(n=int(self.config.max_examples), random_state=int(self.config.seed)).reset_index(drop=True)
            self.training_examples = int(len(train))

        station_ids = sorted({str(s) for s in train["station_id"].astype(str).dropna().unique().tolist()})
        self.station_id_to_idx = {sid: i for i, sid in enumerate(station_ids)}
        self.unknown_station_idx = len(self.station_id_to_idx)

        X_train = _build_feature_matrix(train, self.feature_columns)
        self.tabular_scaler = TabularScaler.fit(X_train)
        train_metrics = self._train_loop(train, valid)
        self.trained = True
        self.trained_at = _utc_now()
        self.method = "macflow_nissm_lite_trained_v1"
        self.model_warning = None

        # Snapshot per-community defaults so request-time scoring can attach community features.
        self.community_runtime_defaults = build_community_runtime_defaults(train, partition)

        # Optional calibrator on validation.
        if self.config.calibrate and not valid.empty and len(valid) >= 20:
            cal_frame = valid
            if len(cal_frame) > 1_000:
                cal_frame = cal_frame.sample(n=1_000, random_state=int(self.config.seed))
            raw = self.predict_distribution(cal_frame, debug=False, _apply_calibration=False)
            try:
                self.calibrator = fit_zero_calibrator(raw, cal_frame)
            except Exception as exc:  # calibration is optional; never fail training
                log.warning("calibration skipped: %s", exc)
                self.calibrator = {"segments": {}, "global": {"a": 1.0, "b": 0.0}, "fitted": False}
        else:
            self.calibrator = {"segments": {}, "global": {"a": 1.0, "b": 0.0}, "fitted": False}

        self.metrics = {
            **train_metrics,
            "n_train": int(len(train)),
            "n_valid": int(len(valid)),
            "training_examples": int(len(train)),
            "training_positive": self.training_positive,
            "model_version": self.config.model_version,
            "method": self.method,
            "partition_mode": partition_mode,
            "n_communities": int(self.partition.n_communities if self.partition else 0),
            "partition_algorithm": str(self.partition.algorithm if self.partition else "none"),
        }
        if not valid.empty:
            self.metrics.update(self._validation_metrics(valid))
        return self

    def predict_distribution(
        self,
        rows: pd.DataFrame,
        debug: bool = False,
        _apply_calibration: bool = True,
    ) -> pd.DataFrame:
        if rows.empty:
            cols = [
                *CDG_REQUIRED_OUTPUT_COLUMNS,
                *CDG_DIAGNOSTIC_COLUMNS,
                "used_sequence_fallback",
                *(CDG_DEBUG_COLUMNS if debug else []),
            ]
            return pd.DataFrame(columns=cols, index=rows.index)

        if not self.trained or self.net is None or self.tabular_scaler is None:
            raise RuntimeError("MacFlow-NISSM-lite has no trained artifact loaded")

        frame = self._prepare_runtime_rows(rows)
        mu_all, theta_all, zeta_all, encoder_norm = self._forward(frame)
        results: list[dict[str, Any]] = []
        for pos, (idx, row) in enumerate(frame.iterrows()):
            cap = max(1, min(int(self.config.max_capacity), _safe_int(row.get("capacity"), 15)))
            e0 = max(0, min(cap, _safe_int(row.get("num_ebikes_available"), 0)))
            q0 = max(e0, min(cap, _safe_int(row.get("num_bikes_available"), e0)))
            rollout = fast_inventory_rollout_from_parameters(
                capacity=cap,
                current_ebikes=e0,
                current_total_bikes=q0,
                mu=mu_all[pos],
                theta=theta_all[pos],
                zeta=zeta_all[pos],
                config=self._rollout_config,
                is_renting=bool(row.get("is_renting", True)),
                is_returning=bool(row.get("is_returning", True)),
            )
            joint = rollout.p_joint_e_q
            calibration_delta = 0.0
            if joint is not None and _apply_calibration and self.calibrator and self.calibrator.get("fitted"):
                joint, calibration_delta = calibrate_joint_zero(
                    joint,
                    calibrator=self.calibrator,
                    horizon_minutes=int(row.get("horizon_minutes", 10)),
                    current_ebikes=e0,
                )
                derived = distribution_from_joint(joint)
            else:
                derived = {
                    "p_has_ebike": rollout.p_has_ebike,
                    "p_zero": rollout.p_zero,
                    "expected_ebikes": rollout.expected_ebikes,
                    "expected_total_bikes": rollout.expected_total_bikes,
                    "p_count_ebikes": rollout.p_count_ebikes,
                    "p_count_total": rollout.p_count_total,
                }
            p_count_ebikes = derived["p_count_ebikes"]
            p_count_total = derived["p_count_total"]
            row_out: dict[str, Any] = {
                "p_has_ebike": float(np.clip(derived["p_has_ebike"], 0.0, 1.0)),
                "p_zero": float(np.clip(derived["p_zero"], 0.0, 1.0)),
                "p_appears": float(derived["p_has_ebike"]) if e0 <= 0 else np.nan,
                "p_survives": float(derived["p_has_ebike"]) if e0 > 0 else np.nan,
                "expected_ebikes": float(derived["expected_ebikes"]),
                "expected_total_bikes": float(derived["expected_total_bikes"]),
                "p_count_ebikes": p_count_ebikes,
                "p_count_total": p_count_total,
                "p_count_ebikes_json": json.dumps(p_count_ebikes, sort_keys=True),
                "p_count_total_json": json.dumps(p_count_total, sort_keys=True),
                "p_capacity_violation": float(rollout.p_capacity_violation),
                "p_dock_constrained_arrival": float(rollout.p_dock_constrained_arrival),
                "expected_ebike_departures": float(rollout.expected_ebike_departures),
                "expected_classic_departures": float(rollout.expected_classic_departures),
                "expected_ebike_arrivals": float(rollout.expected_ebike_arrivals),
                "expected_classic_arrivals": float(rollout.expected_classic_arrivals),
            }
            row_out["p_zero"] = float(1.0 - row_out["p_has_ebike"])
            row_out["used_sequence_fallback"] = True  # MacFlow does not use a sequence encoder
            if debug:
                row_out.update({
                    "mu_e_depart": float(mu_all[pos, 0]),
                    "mu_e_arrive": float(mu_all[pos, 1]),
                    "mu_c_depart": float(mu_all[pos, 2]),
                    "mu_c_arrive": float(mu_all[pos, 3]),
                    "theta_e_depart": float(theta_all[pos, 0]),
                    "theta_e_arrive": float(theta_all[pos, 1]),
                    "theta_c_depart": float(theta_all[pos, 2]),
                    "theta_c_arrive": float(theta_all[pos, 3]),
                    "zero_inflation_e_depart": float(zeta_all[pos, 0]),
                    "zero_inflation_e_arrive": float(zeta_all[pos, 1]),
                    "zero_inflation_c_depart": float(zeta_all[pos, 2]),
                    "zero_inflation_c_arrive": float(zeta_all[pos, 3]),
                    "dock_constraint_probability": float(rollout.p_dock_constrained_arrival),
                    "stockout_probability": float(row_out["p_zero"]),
                    "encoder_norm": float(encoder_norm[pos]) if pos < len(encoder_norm) else 0.0,
                    "graph_message_norm": 0.0,
                    "temporal_message_norm": 0.0,
                    "calibration_delta_zero": float(calibration_delta),
                })
            results.append(row_out)
        return pd.DataFrame(results, index=rows.index)

    def predict_proba(self, rows: pd.DataFrame) -> np.ndarray:
        dist = self.predict_distribution(rows, debug=False)
        p = dist["p_has_ebike"].to_numpy(dtype=float)
        return np.column_stack([1.0 - p, p])

    # ------------------------------ artifact ------------------------------

    def artifact_payload(self) -> dict[str, Any]:
        state_dict_cpu = None
        if _TORCH_AVAILABLE and self.net is not None:
            state_dict_cpu = {k: v.detach().cpu() for k, v in self.net.state_dict().items()}
        return {
            "config": _config_to_dict(self.config),
            "model_state_dict": state_dict_cpu,
            "feature_columns": list(self.feature_columns),
            "tabular_scaler": self.tabular_scaler,
            "station_id_to_idx": dict(self.station_id_to_idx),
            "unknown_station_idx": int(self.unknown_station_idx),
            "partition": self.partition,
            "partition_mode": self.partition_mode,
            "community_runtime_defaults": self.community_runtime_defaults,
            "calibrator": self.calibrator,
            "metrics": dict(self.metrics),
            "trained_at": self.trained_at,
            "training_examples": int(self.training_examples),
            "training_positive": int(self.training_positive),
            "model_version": self.config.model_version,
            "method": self.method,
            "model_warning": self.model_warning,
            "trained": bool(self.trained),
        }

    def __getstate__(self) -> dict[str, Any]:
        return self.artifact_payload()

    def __setstate__(self, state: dict[str, Any]) -> None:
        obj = self._from_payload(state)
        self.__dict__.update(obj.__dict__)

    def save(self, path: str | Path) -> None:
        if _TORCH_AVAILABLE and self.net is not None:
            self.net.cpu()
        with Path(path).open("wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def _from_payload(cls, payload: dict[str, Any]) -> "MacFlowNISSMLite":
        model = cls(payload.get("config"))
        model.feature_columns = list(payload.get("feature_columns") or MACFLOW_NUMERIC_COLUMNS)
        model.tabular_scaler = payload.get("tabular_scaler")
        model.station_id_to_idx = {str(k): int(v) for k, v in (payload.get("station_id_to_idx") or {}).items()}
        model.unknown_station_idx = int(payload.get("unknown_station_idx", len(model.station_id_to_idx)))
        model.partition = payload.get("partition")
        model.partition_mode = str(payload.get("partition_mode") or model.config.partition_mode)
        model.community_runtime_defaults = payload.get("community_runtime_defaults")
        model.calibrator = payload.get("calibrator")
        model.metrics = dict(payload.get("metrics") or {})
        model.trained_at = payload.get("trained_at") or _utc_now()
        model.training_examples = int(payload.get("training_examples") or 0)
        model.training_positive = int(payload.get("training_positive") or 0)
        model.method = str(payload.get("method") or "macflow_nissm_lite_trained_v1")
        model.model_warning = payload.get("model_warning")
        model.trained = bool(payload.get("trained", payload.get("model_state_dict") is not None))
        state_dict = payload.get("model_state_dict")
        if _TORCH_AVAILABLE and state_dict is not None and model.tabular_scaler is not None:
            model.net = model._build_net(
                n_numeric=len(model.feature_columns),
                n_stations=int(model.unknown_station_idx) + 2,
            )
            model.net.load_state_dict(state_dict)
            model.net.cpu().eval()
        return model

    @classmethod
    def load(cls, path: str | Path) -> "MacFlowNISSMLite":
        with Path(path).open("rb") as handle:
            obj = pickle.load(handle)
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, dict):
            return cls._from_payload(obj)
        raise TypeError(f"Artifact is not {cls.__name__}")

    # ------------------------------ internals ------------------------------

    def _quality_gate(self, train: pd.DataFrame, valid: pd.DataFrame) -> str | None:
        if len(train) < int(self.config.min_train_examples):
            return f"insufficient_train_examples:{len(train)}<{self.config.min_train_examples}"
        if "horizon_minutes" in train.columns and int(train["horizon_minutes"].nunique()) < 2:
            return "fewer_than_two_horizons"
        y = train.get("has_ebike", train.get("y_has_ebike"))
        if y is None:
            return "missing_has_ebike_label"
        y_int = pd.to_numeric(y, errors="coerce").fillna(0).astype(int)
        if int(y_int.sum()) < int(self.config.min_positive_examples):
            return "fewer_than_min_positive_examples"
        if int((1 - y_int).sum()) < int(self.config.min_zero_future_examples):
            return "fewer_than_min_zero_future_examples"
        if len(valid) > 0 and len(valid) < int(self.config.min_valid_examples):
            return "validation_too_small_for_registration"
        return None

    def _resolve_partition(self, graph_cache: Any, train_df: pd.DataFrame) -> Partition:
        if isinstance(graph_cache, Partition):
            return graph_cache
        if isinstance(graph_cache, dict):
            cand = graph_cache.get("partition")
            if isinstance(cand, Partition):
                return cand
        # Fallback: single-community partition derived from train station IDs.
        stations = sorted({str(s) for s in train_df.get("station_id", pd.Series(dtype=str)).astype(str).dropna().unique().tolist()})
        end_ts = pd.to_datetime(train_df.get("anchor_ts"), errors="coerce").max() if "anchor_ts" in train_df.columns else _utc_now()
        end = end_ts.to_pydatetime() if isinstance(end_ts, pd.Timestamp) else (end_ts or _utc_now())
        if isinstance(end, pd.Timestamp):
            end = end.to_pydatetime().replace(tzinfo=None)
        return Partition(
            partition_id="fallback_single_community",
            computed_at=_utc_now(),
            source_data_start=None,
            source_data_end=end,
            algorithm="single_community",
            n_communities=1,
            station_to_community={sid: 0 for sid in stations},
            station_to_role={sid: ROLE_UNKNOWN for sid in stations},
            boundary_score={sid: 0.0 for sid in stations},
            gateway_score={sid: 0.0 for sid in stations},
            inbound_internal_share={sid: 0.0 for sid in stations},
            outbound_internal_share={sid: 0.0 for sid in stations},
            community_to_neighbors={0: []},
        )

    def _build_net(self, *, n_numeric: int, n_stations: int) -> Any:
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch unavailable")
        return _MacFlowNet(
            n_numeric=n_numeric,
            n_stations=max(2, n_stations),
            n_communities=max(2, int(self.config.n_communities_cap)),
            n_roles=int(self.config.n_roles),
            n_horizons=int(self.config.n_horizon_buckets),
            hidden_dim=int(self.config.hidden_dim),
            station_dim=int(self.config.station_embedding_dim),
            community_dim=int(self.config.community_embedding_dim),
            role_dim=int(self.config.role_embedding_dim),
            horizon_dim=int(self.config.horizon_embedding_dim),
            dropout=float(self.config.dropout),
            n_layers=int(self.config.n_layers),
        )

    def _train_loop(self, train: pd.DataFrame, valid: pd.DataFrame) -> dict[str, Any]:
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required to train MacFlow-NISSM-lite")
        device_name = select_device(self.config.device)
        torch.manual_seed(int(self.config.seed))

        n_stations = self.unknown_station_idx + 2
        self.net = self._build_net(n_numeric=len(self.feature_columns), n_stations=n_stations)
        try:
            self.net = self.net.to(device_name)
        except Exception as exc:  # MPS unsupported-op or OOM
            log.warning("device %s failed (%s); falling back to CPU", device_name, exc)
            device_name = "cpu"
            self.net = self.net.cpu()

        opt = torch.optim.Adam(
            self.net.parameters(),
            lr=float(self.config.lr),
            weight_decay=float(self.config.weight_decay),
        )
        batch_size = int(self.config.batch_size_mps if device_name == "mps" else self.config.batch_size_cpu)

        tensors = self._make_tensors(train, device_name)
        valid_tensors = self._make_tensors(valid, device_name) if len(valid) else None

        best_loss = float("inf")
        patience = int(self.config.early_stopping_patience)
        no_improve = 0
        epoch_metrics: list[dict[str, float]] = []
        epoch_count = max(1, int(self.config.epochs))

        n = tensors["numeric"].shape[0]
        for epoch in range(epoch_count):
            self.net.train()
            perm = torch.randperm(n, device=device_name)
            running = {"loss": 0.0, "bce": 0.0, "count": 0.0, "reg": 0.0, "n": 0}
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch_idx = perm[start:end]
                out = self.net(
                    numeric=tensors["numeric"][batch_idx],
                    station_idx=tensors["station"][batch_idx],
                    community_idx=tensors["community"][batch_idx],
                    role_idx=tensors["role"][batch_idx],
                    horizon_idx=tensors["horizon"][batch_idx],
                )
                mu = out["mu"]
                theta = out["theta"]
                zeta = out["zeta"]
                e0 = tensors["e0"][batch_idx]
                cap = tensors["capacity"][batch_idx]
                y = tensors["y"][batch_idx]
                p_has = _surrogate_p_has_ebike(mu, theta, zeta, e0, cap)
                bce = -(y * torch.log(p_has) + (1.0 - y) * torch.log(1.0 - p_has)).mean()
                # Count NLL on observed flows if labels present.
                count_loss = torch.tensor(0.0, device=device_name)
                count_terms = 0
                obs = tensors["obs"][batch_idx]
                weights = tensors["weights"][batch_idx]
                # Channels: [e_depart, e_arrive, c_depart, c_arrive].
                for ch in range(4):
                    valid_mask = ~torch.isnan(obs[:, ch])
                    if valid_mask.any():
                        observed = obs[valid_mask, ch].clamp(min=0.0)
                        nll = _zinb_nll(
                            observed,
                            mu[valid_mask, ch],
                            theta[valid_mask, ch],
                            zeta[valid_mask, ch],
                        )
                        weighted = (nll * weights[valid_mask]).sum() / weights[valid_mask].sum().clamp(min=1e-6)
                        count_loss = count_loss + weighted
                        count_terms += 1
                if count_terms:
                    count_loss = count_loss / count_terms
                reg = (mu.pow(2).mean()) * float(self.config.intensity_l2_weight)
                loss = (
                    float(self.config.bce_loss_weight) * bce
                    + float(self.config.count_loss_weight) * count_loss
                    + reg
                )
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=5.0)
                opt.step()
                size = end - start
                running["loss"] += float(loss.detach()) * size
                running["bce"] += float(bce.detach()) * size
                running["count"] += float(count_loss.detach()) * size
                running["reg"] += float(reg.detach()) * size
                running["n"] += size
            avg = {k: (v / running["n"] if running["n"] and isinstance(v, float) else 0.0) for k, v in running.items()}
            epoch_loss = avg["loss"]
            valid_loss = None
            if valid_tensors is not None and valid_tensors["numeric"].shape[0] > 0:
                self.net.eval()
                with torch.no_grad():
                    vout = self.net(
                        numeric=valid_tensors["numeric"],
                        station_idx=valid_tensors["station"],
                        community_idx=valid_tensors["community"],
                        role_idx=valid_tensors["role"],
                        horizon_idx=valid_tensors["horizon"],
                    )
                    vp = _surrogate_p_has_ebike(
                        vout["mu"], vout["theta"], vout["zeta"], valid_tensors["e0"], valid_tensors["capacity"]
                    )
                    vy = valid_tensors["y"]
                    valid_loss = float(-(vy * torch.log(vp) + (1.0 - vy) * torch.log(1.0 - vp)).mean())
            epoch_metrics.append({
                "epoch": int(epoch),
                "train_loss": float(epoch_loss),
                "train_bce": float(avg["bce"]),
                "train_count": float(avg["count"]),
                "valid_bce": float(valid_loss) if valid_loss is not None else None,
            })
            current = valid_loss if valid_loss is not None else epoch_loss
            if current + 1e-6 < best_loss:
                best_loss = current
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        self.net.cpu().eval()
        return {
            "epoch_history": epoch_metrics,
            "device": device_name,
            "best_loss": float(best_loss),
            "batch_size": int(batch_size),
        }

    def _make_tensors(self, frame: pd.DataFrame, device: str) -> dict[str, "torch.Tensor"]:
        if frame.empty:
            zeros_long = torch.zeros(0, dtype=torch.long, device=device)
            zeros_f = torch.zeros(0, dtype=torch.float32, device=device)
            return {
                "numeric": torch.zeros((0, len(self.feature_columns)), dtype=torch.float32, device=device),
                "station": zeros_long,
                "community": zeros_long,
                "role": zeros_long,
                "horizon": zeros_long,
                "e0": zeros_f,
                "capacity": zeros_f,
                "y": zeros_f,
                "obs": torch.zeros((0, 4), dtype=torch.float32, device=device),
                "weights": zeros_f,
            }
        assert self.tabular_scaler is not None
        X = self.tabular_scaler.transform(_build_feature_matrix(frame, self.feature_columns))
        station_idx = _build_station_idx(frame, self.station_id_to_idx, self.unknown_station_idx)
        community_idx = _build_community_idx(frame, self.config.n_communities_cap)
        role_idx = _build_role_idx(frame, self.config.n_roles)
        horizon_idx = _build_horizon_idx(frame, self.config.n_horizon_buckets)
        e0 = pd.to_numeric(frame.get("num_ebikes_available", 0), errors="coerce").fillna(0).astype(float).clip(0, self.config.max_capacity).to_numpy(dtype=np.float32)
        cap = pd.to_numeric(frame.get("capacity", 15), errors="coerce").fillna(15.0).astype(float).clip(1, self.config.max_capacity).to_numpy(dtype=np.float32)
        y = pd.to_numeric(frame.get("has_ebike", frame.get("y_has_ebike", 0)), errors="coerce").fillna(0).astype(float).clip(0, 1).to_numpy(dtype=np.float32)
        # Flow labels — may be NaN for rows where they weren't computed.
        obs = np.stack(
            [
                pd.to_numeric(frame.get("obs_e_depart", np.nan), errors="coerce").to_numpy(dtype=np.float32),
                pd.to_numeric(frame.get("obs_e_arrive", np.nan), errors="coerce").to_numpy(dtype=np.float32),
                pd.to_numeric(frame.get("obs_c_depart", np.nan), errors="coerce").to_numpy(dtype=np.float32),
                pd.to_numeric(frame.get("obs_c_arrive", np.nan), errors="coerce").to_numpy(dtype=np.float32),
            ],
            axis=1,
        )
        weights = pd.to_numeric(frame.get("example_weight", 1.0), errors="coerce").fillna(1.0).astype(float).to_numpy(dtype=np.float32)

        return {
            "numeric": torch.tensor(X, dtype=torch.float32, device=device),
            "station": torch.tensor(station_idx, dtype=torch.long, device=device),
            "community": torch.tensor(community_idx, dtype=torch.long, device=device),
            "role": torch.tensor(role_idx, dtype=torch.long, device=device),
            "horizon": torch.tensor(horizon_idx, dtype=torch.long, device=device),
            "e0": torch.tensor(e0, dtype=torch.float32, device=device),
            "capacity": torch.tensor(cap, dtype=torch.float32, device=device),
            "y": torch.tensor(y, dtype=torch.float32, device=device),
            "obs": torch.tensor(obs, dtype=torch.float32, device=device),
            "weights": torch.tensor(weights, dtype=torch.float32, device=device),
        }

    def _prepare_runtime_rows(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Ensure the runtime frame has every column needed for forward()."""

        frame = rows.copy()
        # If features missing, fall back to runtime defaults (training-period snapshot).
        if not all(col in frame.columns for col in MACFLOW_FEATURE_COLUMNS):
            if self.community_runtime_defaults is not None:
                frame = apply_runtime_defaults(frame, self.community_runtime_defaults, partition_mode=self.partition_mode)
            else:
                for col, default in NEUTRAL_DEFAULTS.items():
                    if col not in frame.columns:
                        frame[col] = default
        # Fill missing numeric columns with 0 so the scaler doesn't choke.
        for col in self.feature_columns:
            if col not in frame.columns:
                frame[col] = 0.0
        for col in ("station_id", "horizon_minutes"):
            if col not in frame.columns:
                frame[col] = "" if col == "station_id" else 10
        return frame

    def _forward(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not _TORCH_AVAILABLE or self.net is None or self.tabular_scaler is None:
            raise RuntimeError("MacFlow-NISSM-lite net not loaded")
        device = "cpu"  # inference is fastest on CPU for the small batches we score
        X = self.tabular_scaler.transform(_build_feature_matrix(frame, self.feature_columns))
        station = _build_station_idx(frame, self.station_id_to_idx, self.unknown_station_idx)
        community = _build_community_idx(frame, self.config.n_communities_cap)
        role = _build_role_idx(frame, self.config.n_roles)
        horizon = _build_horizon_idx(frame, self.config.n_horizon_buckets)
        with torch.no_grad():
            out = self.net(
                numeric=torch.tensor(X, dtype=torch.float32, device=device),
                station_idx=torch.tensor(station, dtype=torch.long, device=device),
                community_idx=torch.tensor(community, dtype=torch.long, device=device),
                role_idx=torch.tensor(role, dtype=torch.long, device=device),
                horizon_idx=torch.tensor(horizon, dtype=torch.long, device=device),
            )
        return (
            out["mu"].cpu().numpy(),
            out["theta"].cpu().numpy(),
            out["zeta"].cpu().numpy(),
            out["encoder_norm"].cpu().numpy(),
        )

    def _validation_metrics(self, valid: pd.DataFrame) -> dict[str, Any]:
        sample = valid
        if len(sample) > 512:
            sample = sample.sample(n=512, random_state=int(self.config.seed))
        dist = self.predict_distribution(sample, debug=False)
        y = pd.to_numeric(sample.get("has_ebike", sample.get("y_has_ebike", 0)), errors="coerce").fillna(0).to_numpy(dtype=float)
        p = pd.to_numeric(dist["p_has_ebike"], errors="coerce").clip(1e-5, 1 - 1e-5).to_numpy(dtype=float)
        brier = float(np.mean((p - y) ** 2)) if len(y) else None
        log_loss = float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))) if len(y) else None
        by_horizon: dict[str, dict[str, float]] = {}
        if "horizon_minutes" in sample.columns:
            tmp = sample[["horizon_minutes"]].copy()
            tmp["_y"] = y
            tmp["_p"] = p
            for horizon, group in tmp.groupby("horizon_minutes"):
                gy = group["_y"].to_numpy(dtype=float)
                gp = group["_p"].clip(1e-5, 1 - 1e-5).to_numpy(dtype=float)
                by_horizon[str(int(horizon))] = {
                    "n": int(len(group)),
                    "brier_score": float(np.mean((gp - gy) ** 2)),
                    "log_loss": float(-np.mean(gy * np.log(gp) + (1 - gy) * np.log(1 - gp))),
                    "mean_prediction": float(np.mean(gp)),
                    "observed_rate": float(np.mean(gy)),
                }
        by_role: dict[str, dict[str, float]] = {}
        if "role_id" in sample.columns:
            for role_int in sorted(int(r) for r in sample["role_id"].dropna().unique().tolist()):
                mask = sample["role_id"].astype(int) == role_int
                if not mask.any():
                    continue
                gy = y[mask.to_numpy()]
                gp = p[mask.to_numpy()]
                role_name = next((k for k, v in ROLE_TO_INT.items() if v == role_int), str(role_int))
                by_role[role_name] = {
                    "n": int(mask.sum()),
                    "brier_score": float(np.mean((gp - gy) ** 2)) if len(gy) else None,
                    "log_loss": float(-np.mean(gy * np.log(gp) + (1 - gy) * np.log(1 - gp))) if len(gy) else None,
                }
        return {
            "brier_score": brier,
            "log_loss": log_loss,
            "rank_loss": float(brier + 0.05 * log_loss) if brier is not None and log_loss is not None else None,
            "mean_prediction": float(np.mean(p)) if len(p) else None,
            "observed_rate": float(np.mean(y)) if len(y) else None,
            "by_horizon": by_horizon,
            "by_role": by_role,
        }


def _maybe_station_aggregates(frame: pd.DataFrame) -> pd.DataFrame | None:
    """Return a build_station_aggregates frame if the training frame carries status snapshots."""

    cols_needed = {"station_id"}
    ts_col = None
    for cand in ("observation_ts", "anchor_ts", "last_reported", "fetched_at"):
        if cand in frame.columns:
            ts_col = cand
            break
    if ts_col is None or not cols_needed.issubset(frame.columns):
        return None
    # Pick out the per-anchor "current" snapshot columns and reframe them as station_status-like rows.
    e_col = "num_ebikes_available" if "num_ebikes_available" in frame.columns else (
        "e_now" if "e_now" in frame.columns else None
    )
    q_col = "num_bikes_available" if "num_bikes_available" in frame.columns else (
        "q_now" if "q_now" in frame.columns else None
    )
    if e_col is None or q_col is None:
        return None
    proxy = pd.DataFrame(
        {
            "station_id": frame["station_id"].astype(str),
            "observation_ts": pd.to_datetime(frame[ts_col], errors="coerce"),
            "num_ebikes_available": pd.to_numeric(frame[e_col], errors="coerce"),
            "num_bikes_available": pd.to_numeric(frame[q_col], errors="coerce"),
            "num_docks_available": pd.to_numeric(
                frame.get("num_docks_available", frame.get("docks_now", 0)),
                errors="coerce",
            ),
        }
    )
    return build_station_aggregates(proxy)
