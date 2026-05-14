from __future__ import annotations

import json
from datetime import datetime, timezone

import duckdb

from . import config, db, model_eval, model_registry, predictor


SOTA_PRIMARY_MODEL_KEYS = predictor.SOTA_PRIMARY_MODEL_KEYS
BASELINE_MODEL_KEYS = predictor.BASELINE_MODEL_KEYS
SELECTION_METRIC = "decision_rank_loss"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _json_text(value) -> str:
    return json.dumps(value or {}, separators=(",", ":"), sort_keys=True, default=str)


def _artifact_for_key(conn: duckdb.DuckDBPyConnection, model_key: str | None) -> dict | None:
    if not model_key:
        return None
    try:
        return model_registry.load_latest_artifact(conn, str(model_key))
    except Exception:
        return None


def _active_artifact(conn: duckdb.DuckDBPyConnection) -> dict | None:
    try:
        return model_registry.load_active_artifact(conn)
    except Exception:
        return None


def _metric_value(row: dict | None, metric: str = SELECTION_METRIC) -> float | None:
    if not row:
        return None
    value = row.get(metric)
    if value is None and metric == SELECTION_METRIC:
        value = row.get("rank_loss")
    return float(value) if value is not None else None


def _find_model(summary: dict, model_key: str | None) -> dict | None:
    if not model_key:
        return None
    for row in summary.get("model_leaderboard") or summary.get("by_model") or []:
        if row.get("model_key") == model_key:
            return row
    return None


def _latest_selection(conn: duckdb.DuckDBPyConnection) -> dict | None:
    try:
        row = conn.execute(
            """
            SELECT *
            FROM model_selection_state
            ORDER BY computed_at DESC
            LIMIT 1
            """
        ).df()
    except Exception:
        return None
    if row.empty:
        return None
    return row.iloc[0].to_dict()


def select_primary_driver(
    conn: duckdb.DuckDBPyConnection,
    window_hours: int | None = None,
    min_resolved: int | None = None,
    *,
    promote: bool = True,
) -> dict:
    """Select and optionally promote the active SOTA driver conservatively."""
    db.init_schema(conn)
    window = int(window_hours or config.ACTIVE_SWITCH_WINDOW_HOURS)
    min_n = int(min_resolved or config.ACTIVE_SWITCH_MIN_RESOLVED)
    forced_key = config.ACTIVE_MODEL_KEY or None
    active_artifact = _active_artifact(conn)
    active_key = forced_key or (active_artifact or {}).get("model_key")
    active_artifact_id = (active_artifact or {}).get("artifact_id")
    active_source = "forced_env" if forced_key else "registry_active"

    if not active_key:
        for key in ("cc_nissm", "dg_nissm", "stg_ncde_inventory", "tft_inventory"):
            artifact = _artifact_for_key(conn, key)
            if artifact:
                active_key = key
                active_artifact_id = artifact.get("artifact_id")
                active_source = "default_cc_nissm" if key == "cc_nissm" else "artifact_available"
                break
    if not active_key:
        active_key = "cc_nissm"
        active_source = "cold_start_sota_bootstrap"

    summary = model_eval.performance_summary(
        conn,
        window_hours=window,
        resolve=False,
        initialize_schema=False,
    )
    best_all = model_eval.best_performing_model(
        conn,
        window_hours=window,
        min_n=min_n,
        eligible_model_keys=predictor.MODEL_KEYS,
        metric=SELECTION_METRIC,
    )
    best_sota = model_eval.best_performing_model(
        conn,
        window_hours=window,
        min_n=min_n,
        eligible_model_keys=SOTA_PRIMARY_MODEL_KEYS,
        metric=SELECTION_METRIC,
    )
    best_baseline = model_eval.best_performing_model(
        conn,
        window_hours=window,
        min_n=min_n,
        eligible_model_keys=BASELINE_MODEL_KEYS,
        metric=SELECTION_METRIC,
    )

    reason = "kept_active"
    candidate_key = best_sota.get("best_model_key")
    candidate_artifact = _artifact_for_key(conn, candidate_key)
    active_row = _find_model(summary, active_key)
    candidate_row = _find_model(summary, candidate_key)
    active_metric = _metric_value(active_row)
    candidate_metric = _metric_value(candidate_row) or best_sota.get("metric_value")
    can_switch = False
    if forced_key:
        reason = "forced_by_env"
    elif not candidate_key:
        reason = "insufficient_resolved_outcomes"
    elif not candidate_artifact:
        reason = "best_sota_has_no_loadable_artifact"
    elif int(best_sota.get("n") or 0) < min_n:
        reason = "insufficient_resolved_outcomes"
    elif (candidate_row or {}).get("capacity_violation_rate") not in (None, 0.0) and float((candidate_row or {}).get("capacity_violation_rate") or 0) > 1e-6:
        reason = "capacity_violation_gate_failed"
    elif candidate_metric is None:
        reason = "candidate_metric_missing"
    elif active_key == candidate_key:
        reason = "active_is_best_sota"
    elif active_metric is not None and candidate_metric > active_metric - float(config.ACTIVE_SWITCH_MARGIN):
        reason = "improvement_below_margin"
    else:
        previous = _latest_selection(conn)
        if previous and previous.get("best_sota_model_key") == candidate_key:
            can_switch = True
            reason = "promoted_best_sota_recent_performance"
        else:
            reason = "waiting_for_persistent_improvement"

    if can_switch and promote and candidate_artifact:
        try:
            model_registry.set_active_artifact(conn, str(candidate_artifact["artifact_id"]))
            active_key = str(candidate_key)
            active_artifact_id = str(candidate_artifact["artifact_id"])
            active_source = "best_sota_recent_performance"
        except Exception as exc:
            reason = f"promotion_failed: {exc}"

    result = {
        "computed_at": _utc_now().isoformat(),
        "active_model_key": active_key,
        "active_artifact_id": active_artifact_id,
        "active_model_source": active_source,
        "best_evaluated_model_key": best_all.get("best_model_key"),
        "best_sota_model_key": best_sota.get("best_model_key"),
        "best_baseline_model_key": best_baseline.get("best_model_key"),
        "active_equals_best": bool(active_key == best_all.get("best_model_key")) if best_all.get("best_model_key") else None,
        "selection_metric": SELECTION_METRIC if best_sota.get("metric") == "decision_rank_loss" else best_sota.get("metric", SELECTION_METRIC),
        "selection_window_hours": window,
        "min_resolved": min_n,
        "reason": reason,
        "metrics": {
            "active": active_row,
            "best_evaluated": best_all,
            "best_sota": best_sota,
            "best_baseline": best_baseline,
            "summary_overall": summary.get("overall"),
        },
    }
    persist_selection_state(conn, result)
    return result


def persist_selection_state(conn: duckdb.DuckDBPyConnection, state: dict) -> None:
    db.init_schema(conn)
    computed = _utc_now()
    conn.execute(
        """
        INSERT INTO model_selection_state (
          computed_at, active_model_key, active_artifact_id, active_model_source,
          best_evaluated_model_key, best_sota_model_key, best_baseline_model_key,
          active_equals_best, selection_metric, selection_window_hours,
          min_resolved, reason, metrics_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            computed,
            state.get("active_model_key"),
            state.get("active_artifact_id"),
            state.get("active_model_source"),
            state.get("best_evaluated_model_key"),
            state.get("best_sota_model_key"),
            state.get("best_baseline_model_key"),
            state.get("active_equals_best"),
            state.get("selection_metric"),
            int(state.get("selection_window_hours") or 0),
            int(state.get("min_resolved") or 0),
            state.get("reason"),
            _json_text(state.get("metrics")),
        ],
    )


def latest_selection_state(conn: duckdb.DuckDBPyConnection) -> dict:
    latest = _latest_selection(conn)
    if latest:
        metrics = latest.get("metrics_json")
        if isinstance(metrics, str):
            try:
                latest["metrics_json"] = json.loads(metrics)
            except json.JSONDecodeError:
                latest["metrics_json"] = {}
        return latest
    return select_primary_driver(conn, promote=False)
