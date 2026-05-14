from __future__ import annotations

import json
import math
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import inventory_dp
from .cdg_nmip import (
    CANONICAL_FEATURE_COLUMNS,
    CDG_DEBUG_COLUMNS,
    CDG_DIAGNOSTIC_COLUMNS,
    CDG_REQUIRED_OUTPUT_COLUMNS,
    CDGNMIPConfig,
    CDGNMIPNet,
    TabularScaler,
    calibrate_joint_zero,
    config_from_dict,
    config_to_dict,
    distribution_from_joint,
    ensure_cdg_features,
    fit_zero_calibrator,
    predict_intensity_parameters,
    rollout_from_parameters,
    train_cdg_net,
)
from .dg_nissm_features import apply_train_valid_shifted_priors, sequence_array_from_rows
from .graph_cache import build_graph_cache_from_examples, empty_graph_cache, graph_cache_edge_count


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


class DGNISSMModel:
    model_key = "dg_nissm"
    model_family = "dg_nissm"
    model_version = "dg-nissm-cdg-nmip-v1"
    method = "dg_nissm_cdg_nmip_untrained"

    def __init__(self, config: CDGNMIPConfig | dict[str, Any] | None = None) -> None:
        self.config = config_from_dict(config)
        self.feature_columns: list[str] = list(CANONICAL_FEATURE_COLUMNS)
        self.tabular_scaler: TabularScaler | None = None
        self.station_id_to_idx: dict[str, int] = {}
        self.unknown_station_idx: int = 0
        self.graph_cache: dict[str, Any] = empty_graph_cache({})
        self.calibrator: dict[str, Any] | None = None
        self.metrics: dict[str, Any] = {}
        self.trained_at: datetime = _utc_now()
        self.training_examples: int = 0
        self.training_positive: int = 0
        self.net: Any | None = None
        self.trained: bool = False
        self.model_warning: str | None = "No trained DG-NISSM artifact loaded."
        self.method = "dg_nissm_unavailable_no_artifact"

    def _quality_gate_reason(self, train_df: pd.DataFrame, valid_df: pd.DataFrame | None) -> str | None:
        if len(train_df) < int(self.config.min_train_examples):
            return f"insufficient_train_examples:{len(train_df)}<{self.config.min_train_examples}"
        if "horizon_minutes" in train_df and train_df["horizon_minutes"].nunique() < 2:
            return "fewer_than_two_horizons"
        y = train_df.get("has_ebike", train_df.get("y_has_ebike"))
        if y is None:
            return "missing_has_ebike_label"
        y = pd.to_numeric(y, errors="coerce").fillna(0).astype(int)
        if int(y.sum()) < int(self.config.min_positive_examples) or int((1 - y).sum()) < int(self.config.min_zero_future_examples):
            return "fewer_than_100_positive_or_zero_future_examples"
        if valid_df is not None and len(valid_df) > 0 and len(valid_df) < min(100, int(self.config.min_valid_examples)):
            return "validation_too_small_for_registration"
        return None

    def fit(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame | None = None,
        graph_cache=None,
    ) -> "DGNISSMModel":
        train = ensure_cdg_features(train_df, alpha=self.config.empirical_bayes_alpha, add_priors=True)
        valid = ensure_cdg_features(valid_df, alpha=self.config.empirical_bayes_alpha, add_priors=False) if valid_df is not None else pd.DataFrame()
        if valid_df is not None and not valid.empty:
            train, valid = apply_train_valid_shifted_priors(
                train,
                valid,
                alpha=self.config.empirical_bayes_alpha,
            )
        reason = self._quality_gate_reason(train, valid if valid_df is not None else None)
        self.training_examples = int(len(train))
        y = train.get("has_ebike", train.get("y_has_ebike", pd.Series(dtype=float)))
        self.training_positive = int(pd.to_numeric(y, errors="coerce").fillna(0).sum()) if len(train) else 0
        if reason is not None:
            self.trained = False
            self.method = "dg_nissm_skipped_insufficient_data"
            self.model_warning = reason
            self.metrics = {
                "status": "skipped",
                "reason": reason,
                "n_train": int(len(train)),
                "n_valid": int(len(valid)),
                "method": self.method,
            }
            return self

        station_ids = sorted(train["station_id"].astype(str).dropna().unique().tolist()) if "station_id" in train else []
        self.station_id_to_idx = {station_id: idx for idx, station_id in enumerate(station_ids)}
        self.unknown_station_idx = len(self.station_id_to_idx)
        self.graph_cache = graph_cache or (
            build_graph_cache_from_examples(
                train,
                station_id_to_idx=self.station_id_to_idx,
                top_k=int(self.config.top_k),
            )
            if self.config.use_graph
            else empty_graph_cache(self.station_id_to_idx)
        )

        self.net, self.tabular_scaler, train_metrics = train_cdg_net(
            train,
            valid if not valid.empty else None,
            config=self.config,
            feature_columns=self.feature_columns,
            station_id_to_idx=self.station_id_to_idx,
            unknown_station_idx=self.unknown_station_idx,
            graph_cache=self.graph_cache,
        )
        self.trained = True
        self.trained_at = _utc_now()
        self.method = "dg_nissm_cdg_nmip_trained_v1"
        self.model_version = self.config.model_version
        self.model_warning = None
        self.metrics = {
            **train_metrics,
            "n_train": int(len(train)),
            "n_valid": int(len(valid)),
            "training_examples": int(len(train)),
            "training_positive": self.training_positive,
            "graph_edge_count": graph_cache_edge_count(self.graph_cache),
            "model_version": self.model_version,
            "method": self.method,
        }
        if self.config.calibrate and not valid.empty and len(valid) >= 20:
            cal_frame = valid
            if len(cal_frame) > 1_000:
                cal_frame = cal_frame.sample(n=1_000, random_state=int(self.config.seed))
            raw = self.predict_distribution(cal_frame, debug=False, _apply_calibration=False)
            self.calibrator = fit_zero_calibrator(raw, cal_frame)
            self.metrics["calibration"] = self.calibrator
        else:
            self.calibrator = {"segments": {}, "global": {"a": 1.0, "b": 0.0}, "fitted": False}
        if not valid.empty:
            self.metrics.update(self._validation_metrics(valid))
        return self

    def _ensure_ready(self) -> None:
        if not self.trained or self.net is None or self.tabular_scaler is None:
            raise RuntimeError("DG-NISSM CDG-NMIP has no trained artifact loaded")

    def _validation_metrics(self, valid: pd.DataFrame) -> dict[str, Any]:
        sample = valid
        if len(sample) > 512:
            sample = sample.sample(n=512, random_state=int(self.config.seed))
        dist = self.predict_distribution(sample, debug=False)
        y = pd.to_numeric(sample.get("has_ebike", sample.get("y_has_ebike")), errors="coerce").fillna(0).to_numpy(dtype=float)
        p = pd.to_numeric(dist["p_has_ebike"], errors="coerce").clip(1e-5, 1 - 1e-5).to_numpy(dtype=float)
        brier = float(np.mean((p - y) ** 2)) if len(y) else None
        log_loss = float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))) if len(y) else None
        count_losses: list[float] = []
        for pmf, obs in zip(dist["p_count_ebikes"], sample.get("future_ebikes", sample.get("e_future", pd.Series(index=sample.index, dtype=float)))):
            if not isinstance(pmf, dict) or pd.isna(obs):
                continue
            obs_int = int(max(0, round(float(obs))))
            key = str(obs_int) if obs_int < 5 else "5_plus"
            count_losses.append(float(-math.log(max(1e-12, float(pmf.get(key, 0.0))))))
        by_horizon = {}
        if "horizon_minutes" in sample:
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
        return {
            "brier_score": brier,
            "log_loss": log_loss,
            "count_nll": float(np.mean(count_losses)) if count_losses else None,
            "count_log_loss": float(np.mean(count_losses)) if count_losses else None,
            "rank_loss": float(brier + 0.05 * log_loss) if brier is not None and log_loss is not None else None,
            "mean_prediction": float(np.mean(p)) if len(p) else None,
            "observed_rate": float(np.mean(y)) if len(y) else None,
            "by_horizon": by_horizon,
        }

    def _parameter_frame(self, rows: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
        self._ensure_ready()
        frame = ensure_cdg_features(rows, alpha=self.config.empirical_bayes_alpha)
        params = predict_intensity_parameters(
            self.net,
            frame,
            feature_columns=self.feature_columns,
            scaler=self.tabular_scaler,
            station_id_to_idx=self.station_id_to_idx,
            unknown_station_idx=self.unknown_station_idx,
            config=self.config,
            graph_cache=self.graph_cache,
            batch_size=max(1, min(len(frame), int(self.config.batch_size))),
            device_name=self.config.runtime_device,
        )
        return frame, params

    def predict_distribution(
        self,
        rows: pd.DataFrame,
        debug: bool = False,
        _apply_calibration: bool = True,
    ) -> pd.DataFrame:
        if rows.empty:
            return pd.DataFrame(columns=[*CDG_REQUIRED_OUTPUT_COLUMNS, *CDG_DIAGNOSTIC_COLUMNS, "used_sequence_fallback", *(CDG_DEBUG_COLUMNS if debug else [])], index=rows.index)
        frame, params = self._parameter_frame(rows)
        _seq, fallback_mask = sequence_array_from_rows(frame, spec=self.config.sequence_spec())
        results: list[dict[str, Any]] = []
        mu_all = params["mu"]
        theta_all = params["theta"]
        zeta_all = params["zeta"]
        for pos, (idx, row) in enumerate(frame.iterrows()):
            cap = max(1, min(int(self.config.max_capacity), _safe_int(row.get("capacity", row.get("capacity_clipped")), 15)))
            e0 = max(0, min(cap, _safe_int(row.get("num_ebikes_available", row.get("current_ebikes_clipped")), 0)))
            q0 = max(e0, min(cap, _safe_int(row.get("num_bikes_available", row.get("current_total_bikes_clipped")), e0)))
            horizon = max(1, _safe_int(row.get("horizon_minutes"), 10))
            rollout = rollout_from_parameters(
                capacity=cap,
                current_ebikes=e0,
                current_total_bikes=q0,
                horizon_minutes=horizon,
                mu=mu_all[pos],
                theta=theta_all[pos],
                zeta=zeta_all[pos],
                config=self.config,
                is_renting=bool(row.get("is_renting", True)),
                is_returning=bool(row.get("is_returning", True)),
            )
            joint = rollout.p_joint_e_q
            calibration_delta = 0.0
            if joint is not None and _apply_calibration:
                joint, calibration_delta = calibrate_joint_zero(
                    joint,
                    calibrator=self.calibrator,
                    horizon_minutes=horizon,
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
            row_out["used_sequence_fallback"] = bool(fallback_mask[pos] >= 0.5) if pos < len(fallback_mask) else True
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
                    "encoder_norm": float(params["encoder_norm"][pos]),
                    "graph_message_norm": float(params["graph_message_norm"][pos]),
                    "temporal_message_norm": float(params["temporal_message_norm"][pos]),
                    "calibration_delta_zero": float(calibration_delta),
                })
            results.append(row_out)
        out = pd.DataFrame(results, index=rows.index)
        return out

    def predict_proba(self, rows: pd.DataFrame) -> np.ndarray:
        dist = self.predict_distribution(rows, debug=False)
        p = dist["p_has_ebike"].to_numpy(dtype=float)
        return np.column_stack([1.0 - p, p])

    def artifact_payload(self) -> dict[str, Any]:
        state_dict_cpu = None
        if self.net is not None:
            state_dict_cpu = {key: value.detach().cpu() for key, value in self.net.state_dict().items()}
        return {
            "config": config_to_dict(self.config),
            "model_state_dict": state_dict_cpu,
            "feature_columns": list(self.feature_columns),
            "tabular_scaler": self.tabular_scaler,
            "station_id_to_idx": dict(self.station_id_to_idx),
            "unknown_station_idx": int(self.unknown_station_idx),
            "graph_cache": self.graph_cache,
            "calibrator": self.calibrator,
            "metrics": dict(self.metrics),
            "trained_at": self.trained_at,
            "training_examples": int(self.training_examples),
            "model_version": self.model_version,
            "method": self.method,
            "model_warning": self.model_warning,
            "trained": bool(self.trained),
            "training_positive": int(self.training_positive),
        }

    def __getstate__(self) -> dict[str, Any]:
        return self.artifact_payload()

    def __setstate__(self, state: dict[str, Any]) -> None:
        obj = self._from_payload(state)
        self.__dict__.update(obj.__dict__)

    def save(self, path: str | Path) -> None:
        if self.net is not None:
            self.net.cpu()
        with Path(path).open("wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def _from_payload(cls, payload: dict[str, Any]) -> "DGNISSMModel":
        model = cls(payload.get("config"))
        model.feature_columns = list(payload.get("feature_columns") or CANONICAL_FEATURE_COLUMNS)
        model.tabular_scaler = payload.get("tabular_scaler")
        model.station_id_to_idx = {str(k): int(v) for k, v in (payload.get("station_id_to_idx") or {}).items()}
        model.unknown_station_idx = int(payload.get("unknown_station_idx", len(model.station_id_to_idx)))
        model.graph_cache = payload.get("graph_cache") or empty_graph_cache(model.station_id_to_idx)
        model.calibrator = payload.get("calibrator")
        model.metrics = dict(payload.get("metrics") or {})
        model.trained_at = payload.get("trained_at") or _utc_now()
        model.training_examples = int(payload.get("training_examples") or model.metrics.get("n_train") or 0)
        model.training_positive = int(payload.get("training_positive") or model.metrics.get("training_positive") or 0)
        model.model_version = str(payload.get("model_version") or model.config.model_version)
        model.method = str(payload.get("method") or "dg_nissm_cdg_nmip_trained_v1")
        model.model_warning = payload.get("model_warning")
        model.trained = bool(payload.get("trained", payload.get("model_state_dict") is not None))
        state_dict = payload.get("model_state_dict")
        if state_dict is not None and model.tabular_scaler is not None and CDGNMIPNet is not None:
            model.net = CDGNMIPNet(
                config=model.config,
                n_tabular=len(model.feature_columns),
                n_stations=max(model.station_id_to_idx.values(), default=-1) + 2,
                seq_channels=len(model.config.sequence_spec().channels),
            )
            model.net.load_state_dict(state_dict)
            model.net.cpu().eval()
        return model

    @classmethod
    def load(cls, path: str | Path) -> "DGNISSMModel":
        with Path(path).open("rb") as handle:
            obj = pickle.load(handle)
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, dict):
            return cls._from_payload(obj)
        raise TypeError(f"Artifact is not {cls.__name__}")


def benchmark_runtime(model: DGNISSMModel, rows: pd.DataFrame, *, repeats: int = 3) -> dict[str, Any]:
    if rows.empty:
        return {"rows": 0, "inference_ms_per_1000_rows_cpu": None}
    old_device = model.config.runtime_device
    model.config.runtime_device = "cpu"
    try:
        durations = []
        for _ in range(max(1, repeats)):
            start = time.perf_counter()
            model.predict_distribution(rows, debug=False)
            durations.append(time.perf_counter() - start)
        per_1000 = min(durations) * 1000.0 / max(1.0, len(rows) / 1000.0)
        return {"rows": int(len(rows)), "inference_ms_per_1000_rows_cpu": float(per_1000)}
    finally:
        model.config.runtime_device = old_device
