from __future__ import annotations

import argparse
import importlib.util
import json
import math
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

from . import config, db, decision_metrics, dynamic_graph, label_builder, model_registry, predictor


BASELINE_TRAIN_KEYS = ("empirical", "logistic", "random_forest", "gradient_boosting")
SOTA_TRAIN_KEYS = ("cc_nissm", "stg_ncde_inventory", "dg_nissm", "tft_inventory", "macflow_nissm_lite")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _parse_dt(value: str | None):
    if not value:
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.to_pydatetime()


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return str(value)


def _positive_rate(frame: pd.DataFrame) -> float | None:
    if frame.empty or "has_ebike" not in frame:
        return None
    return float(frame["has_ebike"].astype(float).mean())


def _dataset_summary(examples: pd.DataFrame, train: pd.DataFrame, valid: pd.DataFrame) -> dict:
    out: dict[str, Any] = {
        "n_examples": int(len(examples)),
        "n_train": int(len(train)),
        "n_valid": int(len(valid)),
        "positive_rate": _positive_rate(examples),
        "positive_rate_train": _positive_rate(train),
        "positive_rate_valid": _positive_rate(valid),
        "positive_rate_by_horizon": {},
    }
    if not examples.empty and "horizon_minutes" in examples:
        out["positive_rate_by_horizon"] = {
            str(int(horizon)): _positive_rate(group)
            for horizon, group in examples.groupby("horizon_minutes")
        }
    for prefix, column in [
        ("anchor", "anchor_ts"),
        ("fetched", "fetched_at"),
        ("source_last_reported", "source_last_reported"),
    ]:
        if column in examples and not examples.empty:
            values = pd.to_datetime(examples[column], errors="coerce").dropna()
            out[f"first_{prefix}_ts"] = values.min().isoformat() if not values.empty else None
            out[f"last_{prefix}_ts"] = values.max().isoformat() if not values.empty else None
    return out


def _configure_runtime(args: argparse.Namespace) -> dict:
    device = str(getattr(args, "device", "auto") or "auto").lower()
    if device in {"auto", "cpu", "mps", "cuda"}:
        predictor.STG_NCDE_DEVICE = device
    if getattr(args, "stg_max_examples", None) is not None:
        predictor.STG_NCDE_MAX_EXAMPLES = int(args.stg_max_examples)
    if getattr(args, "stg_epochs", None) is not None:
        predictor.STG_NCDE_EPOCHS = int(args.stg_epochs)
    if getattr(args, "stg_batch_size", None) is not None:
        predictor.STG_NCDE_BATCH_SIZE = int(args.stg_batch_size)
    return {
        "stg_device": predictor.STG_NCDE_DEVICE,
        "stg_max_examples": predictor.STG_NCDE_MAX_EXAMPLES,
        "stg_epochs": predictor.STG_NCDE_EPOCHS,
        "stg_batch_size": predictor.STG_NCDE_BATCH_SIZE,
        "dg_device": device,
    }


def _split_examples(examples: pd.DataFrame, valid_hours: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if examples.empty:
        return examples.copy(), examples.copy()
    ordered = examples.copy()
    ordered["anchor_ts"] = pd.to_datetime(ordered["anchor_ts"], errors="coerce")
    ordered = ordered.dropna(subset=["anchor_ts"]).sort_values("anchor_ts").reset_index(drop=True)
    if ordered.empty:
        return ordered.copy(), ordered.copy()
    max_anchor = ordered["anchor_ts"].max()
    threshold = max_anchor - pd.Timedelta(hours=max(1, int(valid_hours)))
    train = ordered[ordered["anchor_ts"] < threshold].copy()
    valid = ordered[ordered["anchor_ts"] >= threshold].copy()
    if train.empty or valid.empty:
        split = max(1, int(len(ordered) * 0.8))
        train = ordered.iloc[:split].copy()
        valid = ordered.iloc[split:].copy()
    if valid.empty:
        valid = train.tail(min(len(train), max(1, len(train) // 5))).copy()
    return train.reset_index(drop=True), valid.reset_index(drop=True)


def _build_examples(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    horizons = tuple(int(h) for h in args.horizons)
    read_only = bool(getattr(args, "coexist_live", False))
    with db.session(read_only=read_only, retries=60, retry_sleep=1.0) as conn:
        examples = label_builder.build_leak_free_examples(
            conn,
            history_hours=int(args.history_hours),
            anchor_every_min=int(args.anchor_every_min),
            horizons=horizons,
            clock_col="fetched_at",
            max_current_status_age_min=30,
            max_label_status_age_min=30,
            max_source_rows=int(args.max_source_rows),
            include_sequences=False,
            seq_len=int(getattr(args, "seq_len", 24) or 24),
            seq_step_minutes=int(getattr(args, "seq_step_minutes", 2) or 2),
        )
    train, valid = _split_examples(examples, int(args.valid_hours))
    try:
        from .dg_nissm_features import apply_train_valid_shifted_priors

        train, valid = apply_train_valid_shifted_priors(train, valid)
    except Exception:
        pass
    return examples, train, valid, _dataset_summary(examples, train, valid)


def _clip_prob(values) -> np.ndarray:
    return np.asarray(values, dtype=float).clip(0.001, 0.999)


def _predict_prob(model_key: str, model_obj: object, rows: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame | None]:
    if rows.empty:
        return np.array([], dtype=float), None
    if hasattr(model_obj, "predict_distribution"):
        try:
            dist = model_obj.predict_distribution(rows, debug=False)
        except TypeError:
            dist = model_obj.predict_distribution(rows)
        return _clip_prob(dist["p_has_ebike"]), dist
    if hasattr(model_obj, "predict_proba"):
        feature_columns = predictor._feature_columns_for_model(model_key)
        try:
            probs = model_obj.predict_proba(rows[feature_columns])[:, 1]
        except Exception:
            probs = model_obj.predict_proba(rows)[:, 1]
        return _clip_prob(probs), None
    raise TypeError(f"{model_key} model has neither predict_distribution nor predict_proba")


def _count_metric(distribution: pd.DataFrame | None, valid: pd.DataFrame) -> tuple[float | None, float | None]:
    if distribution is None or distribution.empty or "p_count_ebikes" not in distribution or "future_ebikes" not in valid:
        return None, None
    log_losses: list[float] = []
    crps_values: list[float] = []
    for dist, observed in zip(distribution["p_count_ebikes"], valid["future_ebikes"]):
        if not isinstance(dist, dict) or pd.isna(observed):
            continue
        observed_int = int(max(0, round(float(observed))))
        key = str(observed_int) if observed_int < 5 else "5_plus"
        prob = float(dist.get(key, 0.0) or 0.0)
        log_losses.append(float(-math.log(max(1e-12, prob))))
        cdf = 0.0
        crps = 0.0
        for idx, bucket in enumerate(["0", "1", "2", "3", "4", "5_plus"]):
            cdf += float(dist.get(bucket, 0.0) or 0.0)
            observed_cdf = 1.0 if idx >= min(observed_int, 5) else 0.0
            crps += (cdf - observed_cdf) ** 2
        crps_values.append(float(crps / 6.0))
    return (
        float(np.mean(log_losses)) if log_losses else None,
        float(np.mean(crps_values)) if crps_values else None,
    )


def _evaluate_model(model_key: str, model_obj: object, train: pd.DataFrame, valid: pd.DataFrame, method: str) -> dict:
    metrics: dict[str, Any] = {
        "method": method,
        "n_train": int(len(train)),
        "n_valid": int(len(valid)),
        "training_examples": int(len(train)),
        "training_positive": int(train["has_ebike"].sum()) if not train.empty and "has_ebike" in train else 0,
        "training_negative": int(len(train) - int(train["has_ebike"].sum())) if not train.empty and "has_ebike" in train else 0,
        "positive_rate_train": _positive_rate(train),
        "positive_rate_valid": _positive_rate(valid),
        "warning": None,
        "by_horizon": {},
    }
    if valid.empty or "has_ebike" not in valid:
        metrics["warning"] = "validation set empty; metrics unavailable"
        return metrics
    y = valid["has_ebike"].astype(float).to_numpy()
    p, distribution = _predict_prob(model_key, model_obj, valid)
    metrics["brier_score"] = float(np.mean((p - y) ** 2))
    metrics["log_loss"] = float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
    metrics["rank_loss"] = float(metrics["brier_score"] + 0.05 * metrics["log_loss"])
    metrics["ece"] = decision_metrics.ece_score(y, p)
    metrics["mean_prediction"] = float(np.mean(p))
    metrics["observed_rate"] = float(np.mean(y))
    count_nll, crps = _count_metric(distribution, valid)
    metrics["count_nll"] = count_nll
    metrics["count_log_loss"] = count_nll
    metrics["crps"] = crps
    metrics["decision_rank_loss"] = decision_metrics.decision_rank_loss(metrics)
    if len(valid) < 50 or valid["has_ebike"].nunique() < 2:
        metrics["warning"] = "validation set too small or one-class; artifact is provisional until more outcomes accrue"
    if "horizon_minutes" in valid:
        for horizon, group in valid.assign(_p=p, _y=y).groupby("horizon_minutes"):
            gy = group["_y"].astype(float).to_numpy()
            gp = _clip_prob(group["_p"])
            row = {
                "n": int(len(group)),
                "brier_score": float(np.mean((gp - gy) ** 2)),
                "log_loss": float(-np.mean(gy * np.log(gp) + (1.0 - gy) * np.log(1.0 - gp))),
                "observed_rate": float(np.mean(gy)),
                "mean_prediction": float(np.mean(gp)),
                "ece": decision_metrics.ece_score(gy, gp),
            }
            row["rank_loss"] = float(row["brier_score"] + 0.05 * row["log_loss"])
            row["decision_rank_loss"] = decision_metrics.decision_rank_loss(row)
            metrics["by_horizon"][str(int(horizon))] = row
    return metrics


def _mark_model_metadata(model_obj: object, *, method: str, model_version: str) -> None:
    try:
        setattr(model_obj, "method", method)
    except Exception:
        pass
    try:
        setattr(model_obj, "model_version", model_version)
    except Exception:
        pass
    try:
        setattr(model_obj, "trained_at", _utc_now())
    except Exception:
        pass


def _trained_result(
    model_key: str,
    model_obj: object,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    *,
    model_family: str | None = None,
    model_version: str | None = None,
    feature_columns: list[str] | None = None,
    is_primary_eligible: bool = True,
) -> dict:
    method = str(getattr(model_obj, "method", f"{model_key}_trained_v1"))
    version = str(model_version or getattr(model_obj, "model_version", None) or predictor.MODEL_SPECS[model_key]["version"])
    _mark_model_metadata(model_obj, method=method, model_version=version)
    metrics = _evaluate_model(model_key, model_obj, train, valid, method)
    return {
        "model_key": model_key,
        "status": "trained",
        "model_obj": model_obj,
        "model_family": model_family or model_key,
        "model_version": version,
        "feature_columns": feature_columns or predictor._feature_columns_for_model(model_key),
        "is_primary_eligible": bool(is_primary_eligible),
        "metrics": metrics,
    }


def _train_baseline(model_key: str, train: pd.DataFrame, valid: pd.DataFrame) -> dict:
    fitted = predictor._fit_sklearn_model(train, model_key)
    if fitted.model is None:
        return {"model_key": model_key, "status": "failed", "reason": "insufficient_data_or_fit_failed"}
    _mark_model_metadata(fitted.model, method=fitted.method, model_version=fitted.model_version)
    return _trained_result(
        model_key,
        fitted.model,
        train,
        valid,
        model_version=fitted.model_version,
        feature_columns=predictor._feature_columns_for_model(model_key),
        is_primary_eligible=False,
    )


def _train_cc_nissm(train: pd.DataFrame, valid: pd.DataFrame) -> dict:
    from .cc_nissm import CCNISSMModel

    model = CCNISSMModel()
    model.fit(train, valid)
    return _trained_result(
        "cc_nissm",
        model,
        train,
        valid,
        model_version="cc-nissm-trained-v1",
        feature_columns=predictor.FEATURE_COLUMNS,
        is_primary_eligible=True,
    )


def _train_stg_ncde(train: pd.DataFrame, valid: pd.DataFrame) -> dict:
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("torchcde") is None:
        return {"model_key": "stg_ncde_inventory", "status": "skipped", "reason": "torch/torchcde unavailable"}
    fitted = predictor._fit_stg_ncde_model(train)
    if fitted.model is None:
        return {"model_key": "stg_ncde_inventory", "status": "failed", "reason": "fit returned no model"}
    _mark_model_metadata(
        fitted.model,
        method=str(fitted.method or "stg_ncde_inventory_trained_v1"),
        model_version="stg-ncde-inventory-trained-v1",
    )
    return _trained_result(
        "stg_ncde_inventory",
        fitted.model,
        train,
        valid,
        model_version="stg-ncde-inventory-trained-v1",
        feature_columns=predictor.FEATURE_COLUMNS,
        is_primary_eligible=True,
    )


def _dynamic_graph_edge_count(conn) -> int:
    try:
        return int(conn.execute("SELECT COUNT(*) FROM dynamic_graph_edges").fetchone()[0] or 0)
    except Exception:
        return 0


def _train_dg_nissm(train: pd.DataFrame, valid: pd.DataFrame, horizons: tuple[int, ...], args: argparse.Namespace | None = None) -> dict:
    from .dg_nissm import DGNISSMModel
    from .cdg_nmip import CDGNMIPConfig

    args = args or argparse.Namespace()
    config_obj = CDGNMIPConfig(
        device=str(getattr(args, "device", "auto") or "auto"),
        epochs=int(getattr(args, "epochs", 8) or 8),
        batch_size=int(getattr(args, "batch_size", 4096) or 4096),
        max_examples=int(getattr(args, "max_examples", 600_000) or 600_000),
        hidden_dim=int(getattr(args, "hidden_dim", 128) or 128),
        station_embedding_dim=int(getattr(args, "station_embedding_dim", 32) or 32),
        seq_len=int(getattr(args, "seq_len", 24) or 24),
        seq_step_minutes=int(getattr(args, "seq_step_minutes", 2) or 2),
        top_k=int(getattr(args, "top_k", 16) or 16),
        lr=float(getattr(args, "lr", 1e-3) or 1e-3),
        weight_decay=float(getattr(args, "weight_decay", 1e-4) or 1e-4),
        seed=int(getattr(args, "seed", 42) or 42),
        use_sequence=not bool(getattr(args, "no_sequence", False)),
        use_graph=not bool(getattr(args, "no_graph", False)),
        calibrate=bool(getattr(args, "calibrate", True)),
    )

    if config_obj.use_graph:
        try:
            with db.session(read_only=False, retries=60, retry_sleep=1.0) as conn:
                db.init_schema(conn)
                if _dynamic_graph_edge_count(conn) <= 0:
                    dynamic_graph.refresh_dynamic_graph_cache(conn, lookback_days=30, top_k=int(config_obj.top_k), horizons=horizons)
        except Exception:
            pass
    model = DGNISSMModel(config_obj)
    model.fit(train, valid)
    if not getattr(model, "trained", False):
        return {
            "model_key": "dg_nissm",
            "status": "skipped",
            "reason": getattr(model, "model_warning", "fit skipped"),
            "metrics": getattr(model, "metrics", {}),
            "is_primary_eligible": False,
        }
    _mark_model_metadata(model, method=model.method, model_version=model.model_version)
    result = {
        "model_key": "dg_nissm",
        "status": "trained",
        "model_obj": model,
        "model_family": "dg_nissm",
        "model_version": model.model_version,
        "feature_columns": list(model.feature_columns),
        "is_primary_eligible": True,
        "metrics": dict(getattr(model, "metrics", {}) or {}),
    }
    if result["metrics"].get("decision_rank_loss") is None:
        result["metrics"]["decision_rank_loss"] = decision_metrics.decision_rank_loss(result["metrics"])
    quality = _dg_nissm_quality_gate(model, valid)
    if not quality["ok"]:
        result["is_primary_eligible"] = False
        result.setdefault("metrics", {})["quality_gate"] = quality
    if bool(getattr(args, "benchmark_runtime", False)):
        try:
            from .dg_nissm import benchmark_runtime

            bench_rows = valid.head(min(len(valid), 2_000)) if not valid.empty else train.head(min(len(train), 2_000))
            result.setdefault("metrics", {})["runtime_benchmark"] = benchmark_runtime(model, bench_rows)
        except Exception as exc:
            result.setdefault("metrics", {})["runtime_benchmark_error"] = str(exc)
    return result


def _dg_nissm_quality_gate(model: object, valid: pd.DataFrame) -> dict:
    if "bootstrap" in str(getattr(model, "method", "")).lower() or "fallback" in str(getattr(model, "method", "")).lower():
        return {"ok": False, "reason": "method_contains_bootstrap_or_fallback"}
    if valid.empty or len(valid) < 20:
        return {"ok": False, "reason": "validation_too_small_for_output_gate"}
    try:
        sample = valid.head(min(len(valid), 256)).copy()
        dist = model.predict_distribution(sample, debug=True)
    except Exception as exc:
        return {"ok": False, "reason": f"predict_distribution_failed:{exc}"}
    required = {
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
        "mu_e_depart",
        "mu_e_arrive",
        "theta_e_depart",
        "theta_e_arrive",
    }
    missing = sorted(required.difference(dist.columns))
    if missing:
        return {"ok": False, "reason": "missing_output_columns", "missing": missing}
    for col in ["p_count_ebikes", "p_count_total"]:
        for value in dist[col].head(64):
            if not isinstance(value, dict) or not math.isfinite(float(sum(value.values()))) or abs(float(sum(value.values())) - 1.0) > 1e-6:
                return {"ok": False, "reason": f"{col}_not_normalized"}
    p = pd.to_numeric(dist["p_has_ebike"], errors="coerce")
    if not p.between(0.0, 1.0).all():
        return {"ok": False, "reason": "probabilities_out_of_range"}
    return {"ok": True}


def _train_macflow_nissm_lite(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    horizons: tuple[int, ...],
    args: argparse.Namespace | None = None,
) -> dict:
    from .macflow_nissm import MacFlowNISSMLite, MacFlowNISSMLiteConfig
    from .mobility_partitions import build_partition, upsert_station_communities

    args = args or argparse.Namespace()
    partition_mode = str(getattr(args, "partition_mode", "full") or "full")
    config_obj = MacFlowNISSMLiteConfig(
        hidden_dim=int(getattr(args, "hidden_dim", 64) or 64),
        station_embedding_dim=int(getattr(args, "station_embedding_dim", 8) or 8),
        device=str(getattr(args, "device", "auto") or "auto"),
        epochs=int(getattr(args, "epochs", 5) or 5),
        max_examples=int(getattr(args, "max_examples", 200_000) or 200_000),
        lr=float(getattr(args, "lr", 1e-3) or 1e-3),
        weight_decay=float(getattr(args, "weight_decay", 1e-4) or 1e-4),
        seed=int(getattr(args, "seed", 42) or 42),
        partition_mode=partition_mode,
        calibrate=bool(getattr(args, "calibrate", True)),
    )

    partition = None
    if not train.empty and "anchor_ts" in train.columns:
        training_window_end = pd.to_datetime(train["anchor_ts"], errors="coerce").max()
        if pd.notna(training_window_end):
            try:
                with db.session(read_only=False, retries=60, retry_sleep=1.0) as conn:
                    db.init_schema(conn)
                    partition = build_partition(
                        conn,
                        training_window_end=training_window_end.to_pydatetime().replace(tzinfo=None),
                        lookback_days=int(config_obj.partition_lookback_days),
                        seed=int(config_obj.seed),
                    )
                    try:
                        upsert_station_communities(conn, partition)
                    except Exception:
                        pass
            except Exception:
                partition = None

    model = MacFlowNISSMLite(config_obj)
    model.fit(train, valid, graph_cache=partition)

    if not getattr(model, "trained", False):
        return {
            "model_key": "macflow_nissm_lite",
            "status": "skipped",
            "reason": getattr(model, "model_warning", "fit skipped"),
            "metrics": getattr(model, "metrics", {}),
            "is_primary_eligible": False,
        }
    _mark_model_metadata(model, method=model.method, model_version=model.model_version)
    result = {
        "model_key": "macflow_nissm_lite",
        "status": "trained",
        "model_obj": model,
        "model_family": "macflow_nissm_lite",
        "model_version": model.model_version,
        "feature_columns": list(model.feature_columns),
        "is_primary_eligible": True,
        "metrics": dict(getattr(model, "metrics", {}) or {}),
    }
    if result["metrics"].get("decision_rank_loss") is None:
        result["metrics"]["decision_rank_loss"] = decision_metrics.decision_rank_loss(result["metrics"])
    quality = _dg_nissm_quality_gate(model, valid)
    if not quality["ok"]:
        result["is_primary_eligible"] = False
        result.setdefault("metrics", {})["quality_gate"] = quality
    return result


def _train_tft(train: pd.DataFrame, valid: pd.DataFrame) -> dict:
    if importlib.util.find_spec("pytorch_forecasting") is None or importlib.util.find_spec("lightning") is None:
        return {
            "model_key": "tft_inventory",
            "status": "skipped",
            "reason": "pytorch_forecasting/lightning unavailable",
        }
    from .tft_inventory import TFTInventoryModel

    model = TFTInventoryModel()
    model.fit(train, valid)
    return _trained_result(
        "tft_inventory",
        model,
        train,
        valid,
        model_version="tft-inventory-trained-v1",
        feature_columns=predictor.FEATURE_COLUMNS,
        is_primary_eligible=True,
    )


def _register_results(
    results: list[dict],
    train: pd.DataFrame,
    valid: pd.DataFrame,
    horizons: tuple[int, ...],
) -> list[dict]:
    registered: list[dict] = []
    train_start = train["anchor_ts"].min().to_pydatetime() if not train.empty else None
    train_end = train["anchor_ts"].max().to_pydatetime() if not train.empty else None
    valid_start = valid["anchor_ts"].min().to_pydatetime() if not valid.empty else None
    valid_end = valid["anchor_ts"].max().to_pydatetime() if not valid.empty else None
    with db.session(read_only=False, retries=60, retry_sleep=1.0) as conn:
        db.init_schema(conn)
        for result in results:
            public = {key: value for key, value in result.items() if key != "model_obj"}
            if result.get("status") != "trained":
                registered.append(public)
                continue
            artifact = model_registry.save_artifact(
                conn,
                str(result["model_key"]),
                result["model_obj"],
                model_family=str(result["model_family"]),
                model_version=str(result["model_version"]),
                feature_columns=list(result["feature_columns"]),
                train_start=train_start,
                train_end=train_end,
                valid_start=valid_start,
                valid_end=valid_end,
                horizons=list(horizons),
                metrics=dict(result["metrics"]),
                calibration={},
                is_primary_eligible=bool(result["is_primary_eligible"]),
                is_active=False,
            )
            public.update(artifact)
            registered.append(public)
        try:
            db.refresh_read_replica()
        except Exception:
            pass
    return registered


def _activate_requested(results: list[dict], activate: str | None) -> dict | None:
    if not activate:
        return None
    target = "cc_nissm" if activate == "__self__" else str(activate).replace("-", "_")
    candidates = [
        result for result in results
        if result.get("status") == "trained"
        and result.get("model_key") == target
        and result.get("artifact_id")
    ]
    if not candidates:
        return {"status": "not_activated", "target": target, "reason": "target did not produce a trained artifact"}
    artifact_id = str(candidates[-1]["artifact_id"])
    with db.session(read_only=False, retries=60, retry_sleep=1.0) as conn:
        db.init_schema(conn)
        model_registry.set_active_artifact(conn, artifact_id)
        try:
            db.refresh_read_replica()
        except Exception:
            pass
    predictor._MODEL_CACHE.clear()
    return {"status": "activated", "target": target, "artifact_id": artifact_id}


def _best_trained_sota_key(results: list[dict]) -> str | None:
    candidates = []
    for result in results:
        if (
            result.get("status") != "trained"
            or result.get("model_key") not in predictor.SOTA_PRIMARY_MODEL_KEYS
            or not result.get("artifact_id")
            or not bool(result.get("is_primary_eligible", True))
        ):
            continue
        metrics = result.get("metrics") or {}
        value = metrics.get("decision_rank_loss")
        if value is None:
            value = metrics.get("rank_loss")
        if value is None:
            continue
        try:
            metric_value = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(metric_value):
            continue
        candidates.append((metric_value, str(result["model_key"])))
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])[1]


def _probability_summary(conn) -> dict:
    try:
        from . import model_eval

        candidates = model_eval._all_station_candidates(conn)
        if candidates.empty:
            return {"status": "no_stations"}
        scored, suite = predictor.score_candidates(conn, candidates, model_keys=(predictor.ACTIVE_MODEL_KEY,))
        p_cols = [c for c in scored.columns if c.startswith("p_has_ebike_") and c.endswith("m")]
        summary = {}
        for col in p_cols:
            values = pd.to_numeric(scored[col], errors="coerce").dropna()
            if values.empty:
                continue
            summary[col] = {
                "count": int(values.count()),
                "mean": float(values.mean()),
                "min": float(values.min()),
                "p50": float(values.quantile(0.50)),
                "p95": float(values.quantile(0.95)),
                "max": float(values.max()),
            }
        return {"status": "ok", "active_model_key": suite.active_key, "probabilities": summary}
    except Exception as exc:
        return {"status": "failed", "error": str(exc)}


def _post_train_diagnostics() -> dict:
    predictor._MODEL_CACHE.clear()
    with db.session(read_only=False, retries=60, retry_sleep=1.0) as conn:
        db.init_schema(conn)
        artifacts = model_registry.list_artifacts(conn)
        active = model_registry.load_active_artifact(conn)
        health = predictor.model_health(conn)
        probability_summary = _probability_summary(conn)
    bootstrap_models = [
        model for model in health.get("models", [])
        if model.get("bootstrap_or_fallback")
    ]
    return {
        "model_artifacts": artifacts,
        "active_artifact": {
            key: value for key, value in (active or {}).items()
            if key != "model"
        } if active else None,
        "model_health": health,
        "probability_summary": probability_summary,
        "bootstrap_models_still_used": len(bootstrap_models),
        "bootstrap_model_keys": [row.get("model_key") for row in bootstrap_models],
    }


def train_all(args: argparse.Namespace) -> dict:
    started = time.monotonic()
    runtime = _configure_runtime(args)
    horizons = tuple(int(h) for h in args.horizons)
    build_started = time.monotonic()
    examples, train, valid, dataset = _build_examples(args)
    stage_timings: dict[str, float] = {
        "build_examples_seconds": round(time.monotonic() - build_started, 3)
    }
    results: list[dict] = []
    if train.empty:
        return {
            "status": "failed",
            "reason": "no training examples",
            "dataset": dataset,
            "models": [],
        }

    def time_exhausted() -> bool:
        budget = float(getattr(args, "time_budget_hours", 0.0) or 0.0)
        return budget > 0.0 and (time.monotonic() - started) >= budget * 3600.0

    train_order = [
        *BASELINE_TRAIN_KEYS,
        "cc_nissm",
        "stg_ncde_inventory",
        "dg_nissm",
        "tft_inventory",
    ]
    for model_key in train_order:
        model_started = time.monotonic()
        if time_exhausted():
            results.append({"model_key": model_key, "status": "skipped", "reason": "time budget exhausted"})
            continue
        try:
            if model_key in BASELINE_TRAIN_KEYS:
                result = _train_baseline(model_key, train, valid)
            elif model_key == "cc_nissm":
                result = _train_cc_nissm(train, valid)
            elif model_key == "stg_ncde_inventory":
                result = _train_stg_ncde(train, valid)
            elif model_key == "dg_nissm":
                result = _train_dg_nissm(train, valid, horizons, args)
            elif model_key == "tft_inventory":
                result = _train_tft(train, valid)
            else:
                result = {"model_key": model_key, "status": "skipped", "reason": "unknown model"}
        except Exception as exc:
            result = {"model_key": model_key, "status": "failed", "error": str(exc)}
            if getattr(args, "strict", False):
                raise
        result["fit_seconds"] = round(time.monotonic() - model_started, 3)
        stage_timings[f"fit_{model_key}_seconds"] = result["fit_seconds"]
        results.append(result)

    register_started = time.monotonic()
    public_results = _register_results(results, train, valid, horizons) if args.register else [
        {key: value for key, value in result.items() if key != "model_obj"}
        for result in results
    ]
    stage_timings["register_seconds"] = round(time.monotonic() - register_started, 3)
    activate_started = time.monotonic()
    activate_target = args.activate
    if args.activate_best_sota and not activate_target:
        activate_target = _best_trained_sota_key(public_results)
    activation = _activate_requested(public_results, activate_target) if args.register else None
    stage_timings["activate_seconds"] = round(time.monotonic() - activate_started, 3)
    diagnostics_started = time.monotonic()
    diagnostics = _post_train_diagnostics() if args.register else {}
    stage_timings["diagnostics_seconds"] = round(time.monotonic() - diagnostics_started, 3)
    status = "ok"
    if args.strict and any(result.get("status") != "trained" for result in public_results):
        status = "failed"
    return {
        "status": status,
        "runtime": runtime,
        "dataset": dataset,
        "models": public_results,
        "activation": activation,
        "diagnostics": diagnostics,
        "stage_timings": stage_timings,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def _single_command_to_key(command: str) -> str:
    return {
        "cc-nissm": "cc_nissm",
        "dg-nissm": "dg_nissm",
        "stg-ncde": "stg_ncde_inventory",
        "stg-ncde-inventory": "stg_ncde_inventory",
        "tft": "tft_inventory",
        "tft-inventory": "tft_inventory",
        "macflow-nissm-lite": "macflow_nissm_lite",
    }[command]


def train_single(args: argparse.Namespace) -> dict:
    runtime = _configure_runtime(args)
    horizons = tuple(int(h) for h in args.horizons)
    _examples, train, valid, dataset = _build_examples(args)
    model_key = _single_command_to_key(args.command)
    if model_key == "cc_nissm":
        result = _train_cc_nissm(train, valid)
    elif model_key == "stg_ncde_inventory":
        result = _train_stg_ncde(train, valid)
    elif model_key == "dg_nissm":
        result = _train_dg_nissm(train, valid, horizons, args)
    elif model_key == "tft_inventory":
        result = _train_tft(train, valid)
    elif model_key == "macflow_nissm_lite":
        result = _train_macflow_nissm_lite(train, valid, horizons, args)
    else:
        result = {"model_key": model_key, "status": "skipped", "reason": "unknown model"}
    public_results = _register_results([result], train, valid, horizons) if args.register else [
        {key: value for key, value in result.items() if key != "model_obj"}
    ]
    activate_target = model_key if args.activate == "__self__" else args.activate
    activation = _activate_requested(public_results, activate_target) if args.register else None
    diagnostics = _post_train_diagnostics() if args.register else {}
    return {"status": "ok", "runtime": runtime, "dataset": dataset, "models": public_results, "activation": activation, "diagnostics": diagnostics}


def train_baselines_command(args: argparse.Namespace) -> dict:
    runtime = _configure_runtime(args)
    horizons = tuple(int(h) for h in args.horizons)
    _examples, train, valid, dataset = _build_examples(args)
    results = [_train_baseline(model_key, train, valid) for model_key in BASELINE_TRAIN_KEYS]
    public_results = _register_results(results, train, valid, horizons) if args.register else [
        {key: value for key, value in result.items() if key != "model_obj"}
        for result in results
    ]
    return {"status": "ok", "runtime": runtime, "dataset": dataset, "models": public_results}


def run_nightly(
    train_days: int = 60,
    valid_days: int = 7,
    *,
    activate_if_none: bool = True,
) -> dict:
    args = argparse.Namespace(
        command="all",
        history_hours=max(24, min(int(train_days) * 24, 24 * 7)),
        valid_hours=max(4, min(int(valid_days) * 24, 24)),
        anchor_every_min=config.TRAIN_ANCHOR_EVERY_MIN,
        horizons=list(predictor.HORIZONS),
        max_source_rows=2_000_000,
        device="auto",
        register=True,
        activate=None,
        activate_best_sota=bool(activate_if_none),
        coexist_live=True,
        time_budget_hours=8.0,
        strict=False,
        epochs=8,
        batch_size=4096,
        max_examples=600_000,
        hidden_dim=128,
        station_embedding_dim=32,
        seq_len=24,
        seq_step_minutes=2,
        top_k=16,
        lr=1e-3,
        weight_decay=1e-4,
        seed=42,
        no_sequence=False,
        no_graph=False,
        calibrate=True,
        benchmark_runtime=True,
        stg_max_examples=None,
        stg_epochs=None,
        stg_batch_size=None,
    )
    return train_all(args)


def run_weekly(train_days: int = 60, valid_days: int = 7, top_k: int = 16) -> dict:
    del top_k
    return run_nightly(train_days=train_days, valid_days=valid_days, activate_if_none=True)


def _add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--history-hours", type=int, default=24)
    parser.add_argument("--valid-hours", type=int, default=4)
    parser.add_argument("--anchor-every-min", type=int, default=config.TRAIN_ANCHOR_EVERY_MIN)
    parser.add_argument("--horizons", type=int, nargs="+", default=list(predictor.HORIZONS))
    parser.add_argument("--max-source-rows", type=int, default=2_000_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--activate", nargs="?", const="__self__")
    parser.add_argument("--activate-best-sota", action="store_true")
    parser.add_argument("--coexist-live", action="store_true")
    parser.add_argument("--time-budget-hours", type=float, default=8.0)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--stg-max-examples", type=int)
    parser.add_argument("--stg-epochs", type=int)
    parser.add_argument("--stg-batch-size", type=int)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-examples", type=int, default=600_000)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--station-embedding-dim", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--seq-step-minutes", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-sequence", action="store_true")
    parser.add_argument("--no-graph", action="store_true")
    parser.add_argument("--calibrate", dest="calibrate", action="store_true", default=True)
    parser.add_argument("--no-calibrate", dest="calibrate", action="store_false")
    parser.add_argument("--benchmark-runtime", action="store_true")
    parser.add_argument(
        "--partition-mode",
        default="full",
        choices=["off", "id_only", "id_plus_role", "full", "random", "spatial"],
        help="MacFlow-NISSM-lite community-feature ablation mode (default: full).",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train Divvy offline inventory models")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in [
        "all",
        "cc-nissm",
        "dg-nissm",
        "stg-ncde",
        "stg-ncde-inventory",
        "tft",
        "tft-inventory",
        "baselines",
        "macflow-nissm-lite",
    ]:
        _add_shared_args(sub.add_parser(name))
    p = sub.add_parser("all-nightly")
    p.add_argument("--train-days", type=int, default=60)
    p.add_argument("--valid-days", type=int, default=7)
    p = sub.add_parser("all-weekly")
    p.add_argument("--train-days", type=int, default=60)
    p.add_argument("--valid-days", type=int, default=7)
    p.add_argument("--top-k", type=int, default=16)
    args = parser.parse_args(argv)

    if args.command == "all":
        result = train_all(args)
    elif args.command == "baselines":
        result = train_baselines_command(args)
    elif args.command == "all-nightly":
        result = run_nightly(train_days=args.train_days, valid_days=args.valid_days)
    elif args.command == "all-weekly":
        result = run_weekly(train_days=args.train_days, valid_days=args.valid_days, top_k=args.top_k)
    else:
        result = train_single(args)
    print(json.dumps(result, indent=2, default=_json_default))
    return 0 if result.get("status") != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
