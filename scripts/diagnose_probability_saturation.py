#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from divvy import config, db, inventory_dp, label_builder, model_eval, predictor  # noqa: E402


PROBABILITY_COLUMN_RE = re.compile(
    r"^(p_has_ebike_\d+m(?:_.+)?|p_zero_\d+m(?:_.+)?|p_appears_\d+m(?:_.+)?|"
    r"p_survives_\d+m(?:_.+)?|p_arrival(?:_.+)?|rank_probability(?:_.+)?|"
    r"reliable_probability_lcb(?:_.+)?|walk_adjusted_score(?:_.+)?|"
    r"p_capacity_violation_\d+m(?:_.+)?|p_dock_constrained_arrival_\d+m(?:_.+)?)$"
)

FEATURE_COLUMNS_OF_INTEREST = [
    "current_ebikes_clipped",
    "current_total_bikes_clipped",
    "docks_available_clipped",
    "capacity_clipped",
    "ebike_share_of_bikes",
    "dock_availability_fraction",
    "station_same_hour_rate",
    "nearby_same_hour_rate",
    "station_neighbor_same_hour_rate",
    "station_neighbor_recent_ebikes",
    "station_neighbor_recent_zero_rate",
    "trend_5m",
    "trend_10m",
    "trend_15m",
    "churn_rate",
    "trip_departures_same_hour_10m",
    "trip_arrivals_same_hour_10m",
    "trip_net_arrivals_same_hour_10m",
    "trip_recent_departures_30m",
    "trip_recent_arrivals_30m",
    "trip_recent_net_arrivals_30m",
    "route_inbound_trips_same_hour",
    "route_inbound_ebike_share_same_hour",
    "route_inbound_median_duration_minutes",
    "route_inbound_due_horizon",
    "weather_bad_conditions",
    "data_age_minutes",
]

ZERO_EXAMPLE_BASE_COLUMNS = [
    "station_id",
    "name",
    "capacity",
    "num_bikes_available",
    "num_ebikes_available",
    "num_docks_available",
    "last_reported",
    "data_age_minutes",
    "is_renting",
    "trip_arrivals_same_hour_10m",
    "trip_departures_same_hour_10m",
    "trip_recent_arrivals_30m",
    "trip_recent_departures_30m",
    "station_same_hour_rate",
    "station_neighbor_same_hour_rate",
    "station_neighbor_recent_ebikes",
    "trend_5m",
    "trend_10m",
    "trend_15m",
    "churn_rate",
    "route_inbound_due_horizon",
]


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def json_default(value: Any) -> Any:
    if isinstance(value, (datetime, pd.Timestamp)):
        if pd.isna(value):
            return None
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def clean_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (datetime, pd.Timestamp)):
        if pd.isna(value):
            return None
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def csv_safe_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in out.columns:
        needs_json = False
        for value in out[column].head(50):
            if isinstance(value, (dict, list, tuple, np.ndarray)):
                needs_json = True
                break
        if needs_json:
            out[column] = out[column].map(
                lambda v: json.dumps(v, default=json_default, sort_keys=True)
                if isinstance(v, (dict, list, tuple, np.ndarray))
                else clean_scalar(v)
            )
        elif pd.api.types.is_datetime64_any_dtype(out[column]):
            out[column] = out[column].map(clean_scalar)
    return out


def write_csv(df: pd.DataFrame, path: Path) -> None:
    csv_safe_frame(df).to_csv(path, index=False)


def write_json(payload: dict, path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n")


def table_exists(conn, table_name: str) -> bool:
    try:
        row = conn.execute(
            """
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_name = ?
            """,
            [table_name],
        ).fetchone()
        return bool(row and row[0])
    except Exception:
        return False


def query_df(conn, sql: str, params: list[Any] | None = None) -> pd.DataFrame:
    try:
        return conn.execute(sql, params or []).df()
    except Exception as exc:
        return pd.DataFrame({"query_error": [str(exc)]})


def query_one(conn, sql: str, params: list[Any] | None = None) -> tuple:
    try:
        row = conn.execute(sql, params or []).fetchone()
        return tuple(row or ())
    except Exception:
        return tuple()


def all_station_candidates(conn) -> pd.DataFrame:
    try:
        return model_eval._all_station_candidates(conn)
    except Exception:
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
            )
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
              0.0 AS distance_km
            FROM stations s
            JOIN latest l USING (station_id)
            WHERE s.lat IS NOT NULL AND s.lon IS NOT NULL
            """
        ).df()


def numeric_summary(series: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(series, errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    count = int(finite.count())
    missing = int(len(series) - count)
    out: dict[str, Any] = {
        "count": count,
        "missing": missing,
        "min": None,
        "p01": None,
        "p05": None,
        "p10": None,
        "p25": None,
        "median": None,
        "mean": None,
        "p75": None,
        "p90": None,
        "p95": None,
        "p99": None,
        "max": None,
    }
    if count:
        quantiles = finite.quantile([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
        out.update({
            "min": float(finite.min()),
            "p01": float(quantiles.loc[0.01]),
            "p05": float(quantiles.loc[0.05]),
            "p10": float(quantiles.loc[0.10]),
            "p25": float(quantiles.loc[0.25]),
            "median": float(quantiles.loc[0.50]),
            "mean": float(finite.mean()),
            "p75": float(quantiles.loc[0.75]),
            "p90": float(quantiles.loc[0.90]),
            "p95": float(quantiles.loc[0.95]),
            "p99": float(quantiles.loc[0.99]),
            "max": float(finite.max()),
        })
    return out


def probability_column_summary(scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in scored.columns:
        if not PROBABILITY_COLUMN_RE.match(str(column)):
            continue
        numeric = pd.to_numeric(scored[column], errors="coerce")
        finite = numeric[np.isfinite(numeric)]
        if finite.empty:
            continue
        summary = numeric_summary(scored[column])
        rounded_values = sorted(finite.round(3).dropna().unique().tolist())
        n = int(finite.count())
        n_ge_099 = int((finite >= 0.99).sum())
        n_ge_0995 = int((finite >= 0.995).sum())
        n_ge_0999 = int((finite >= 0.999).sum())
        n_eq_1 = int((finite == 1.0).sum())
        flags = []
        if summary["max"] == 1.0:
            flags.append("max_eq_1")
        if summary["mean"] is not None and summary["mean"] > 0.95:
            flags.append("mean_gt_0.95")
        if n and n_ge_0995 / n > 0.5:
            flags.append("majority_ge_0.995")
        rows.append({
            "column": column,
            **summary,
            "n_ge_0_99": n_ge_099,
            "n_ge_0_995": n_ge_0995,
            "n_ge_0_999": n_ge_0999,
            "n_eq_1_0": n_eq_1,
            "n_lt_0_01": int((finite < 0.01).sum()),
            "unique_rounded_3_count": int(len(rounded_values)),
            "unique_rounded_3_values": json.dumps(rounded_values[:80]),
            "suspicious_flags": ",".join(flags),
        })
    return pd.DataFrame(rows).sort_values(["n_ge_0_995", "mean"], ascending=[False, False])


def model_horizon_probability_summary(scored: pd.DataFrame, suite) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    current = pd.to_numeric(scored.get("num_ebikes_available"), errors="coerce")
    zero_mask = current.fillna(-1) == 0
    for model_key in predictor.MODEL_KEYS:
        for horizon in predictor.HORIZONS:
            col = f"p_has_ebike_{horizon}m_{model_key}"
            if col not in scored.columns:
                continue
            numeric = pd.to_numeric(scored[col], errors="coerce")
            finite = numeric[np.isfinite(numeric)]
            if finite.empty:
                continue
            summary = numeric_summary(numeric)
            rows.append({
                "model_key": model_key,
                "horizon_minutes": int(horizon),
                "source_column": col,
                "active": model_key == suite.active_key,
                "model_method": getattr(suite.models.get(model_key), "method", None),
                **summary,
                "n_ge_0_99": int((finite >= 0.99).sum()),
                "n_ge_0_995": int((finite >= 0.995).sum()),
                "n_ge_0_999": int((finite >= 0.999).sum()),
                "n_eq_1_0": int((finite == 1.0).sum()),
                "zero_current_n": int(zero_mask.sum()),
                "zero_current_n_ge_0_995": int((numeric[zero_mask] >= 0.995).sum()),
                "zero_current_mean": float(numeric[zero_mask].mean()) if zero_mask.any() else None,
                "zero_current_max": float(numeric[zero_mask].max()) if zero_mask.any() else None,
            })
    return pd.DataFrame(rows).sort_values(["model_key", "horizon_minutes"])


def current_ebike_bucket(value: Any) -> str:
    if value is None or pd.isna(value):
        return "missing"
    value = float(value)
    if value <= 0:
        return "0"
    if value == 1:
        return "1"
    if value == 2:
        return "2"
    return "3+"


def probability_by_current_bucket(scored: pd.DataFrame) -> pd.DataFrame:
    working = scored.copy()
    working["current_ebike_bucket"] = working["num_ebikes_available"].map(current_ebike_bucket)
    rows: list[dict[str, Any]] = []
    for model_key in predictor.MODEL_KEYS:
        for horizon in predictor.HORIZONS:
            col = f"p_has_ebike_{horizon}m_{model_key}"
            if col not in working.columns:
                continue
            values = pd.to_numeric(working[col], errors="coerce")
            for bucket, group in working.groupby("current_ebike_bucket", dropna=False):
                idx = group.index
                p = values.loc[idx].dropna()
                if p.empty:
                    continue
                rows.append({
                    "model_key": model_key,
                    "horizon_minutes": int(horizon),
                    "current_ebike_bucket": bucket,
                    "n": int(len(p)),
                    "mean_p_has": float(p.mean()),
                    "median_p_has": float(p.median()),
                    "min_p_has": float(p.min()),
                    "max_p_has": float(p.max()),
                    "p95_p_has": float(p.quantile(0.95)),
                    "n_ge_0_995": int((p >= 0.995).sum()),
                    "observed_current_ebikes_mean": float(pd.to_numeric(group["num_ebikes_available"], errors="coerce").mean()),
                    "mean_current_total_bikes": float(pd.to_numeric(group["num_bikes_available"], errors="coerce").mean()),
                    "mean_docks": float(pd.to_numeric(group["num_docks_available"], errors="coerce").mean()),
                    "mean_status_age_minutes": float(pd.to_numeric(group.get("data_age_minutes"), errors="coerce").mean()),
                })
    return pd.DataFrame(rows).sort_values(["model_key", "horizon_minutes", "current_ebike_bucket"])


def active_vs_suffixed(scored: pd.DataFrame, active_key: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    comparisons = []
    for horizon in predictor.HORIZONS:
        comparisons.extend([
            (f"p_has_ebike_{horizon}m", f"p_has_ebike_{horizon}m_{active_key}"),
            (f"p_zero_{horizon}m", f"p_zero_{horizon}m_{active_key}"),
            (f"p_appears_{horizon}m", f"p_appears_{horizon}m_{active_key}"),
            (f"p_survives_{horizon}m", f"p_survives_{horizon}m_{active_key}"),
        ])
    comparisons.extend([
        ("p_arrival", f"p_arrival_{active_key}"),
        ("rank_probability", f"rank_probability_{active_key}"),
        ("walk_adjusted_score", f"walk_adjusted_score_{active_key}"),
        ("reliable_probability_lcb", f"reliable_probability_lcb_{active_key}"),
    ])
    for unsuffixed, suffixed in comparisons:
        if unsuffixed not in scored.columns or suffixed not in scored.columns:
            rows.append({
                "unsuffixed_column": unsuffixed,
                "active_suffixed_column": suffixed,
                "status": "missing_column",
                "max_abs_diff": None,
                "mean_abs_diff": None,
                "n_mismatched_gt_1e_9": None,
            })
            continue
        a = pd.to_numeric(scored[unsuffixed], errors="coerce")
        b = pd.to_numeric(scored[suffixed], errors="coerce")
        diff = (a - b).abs()
        finite = diff[np.isfinite(diff)]
        rows.append({
            "unsuffixed_column": unsuffixed,
            "active_suffixed_column": suffixed,
            "status": "ok",
            "max_abs_diff": float(finite.max()) if not finite.empty else None,
            "mean_abs_diff": float(finite.mean()) if not finite.empty else None,
            "n_mismatched_gt_1e_9": int((finite > 1e-9).sum()),
            "sample_mismatches": json.dumps(
                scored.loc[finite[finite > 1e-9].head(5).index, ["station_id", "name"]]
                .assign(abs_diff=finite[finite > 1e-9].head(5))
                .to_dict(orient="records"),
                default=json_default,
            ),
        })
    return pd.DataFrame(rows)


def p_zero_identity_checks(scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    current = pd.to_numeric(scored.get("num_ebikes_available"), errors="coerce")
    zero_mask = current.fillna(-1) == 0
    positive_mask = current.fillna(0) > 0
    for model_key in predictor.MODEL_KEYS:
        for horizon in predictor.HORIZONS:
            p_has_col = f"p_has_ebike_{horizon}m_{model_key}"
            p_zero_col = f"p_zero_{horizon}m_{model_key}"
            if p_has_col not in scored.columns or p_zero_col not in scored.columns:
                continue
            p_has = pd.to_numeric(scored[p_has_col], errors="coerce")
            p_zero = pd.to_numeric(scored[p_zero_col], errors="coerce")
            identity = (p_has + p_zero - 1.0).abs()
            appears_col = f"p_appears_{horizon}m_{model_key}"
            survives_col = f"p_survives_{horizon}m_{model_key}"
            appears = pd.to_numeric(scored[appears_col], errors="coerce") if appears_col in scored.columns else pd.Series(np.nan, index=scored.index)
            survives = pd.to_numeric(scored[survives_col], errors="coerce") if survives_col in scored.columns else pd.Series(np.nan, index=scored.index)
            appears_diff = (appears[zero_mask] - p_has[zero_mask]).abs()
            survives_diff = (survives[positive_mask] - p_has[positive_mask]).abs()
            rows.append({
                "model_key": model_key,
                "horizon_minutes": int(horizon),
                "max_abs_p_has_plus_p_zero_minus_1": float(identity.max(skipna=True)) if identity.notna().any() else None,
                "n_errors_gt_1e_6": int((identity > 1e-6).sum()),
                "n_errors_gt_1e_3": int((identity > 1e-3).sum()),
                "zero_current_n": int(zero_mask.sum()),
                "zero_current_appears_max_abs_diff_from_p_has": float(appears_diff.max(skipna=True)) if appears_diff.notna().any() else None,
                "zero_current_appears_errors_gt_1e_6": int((appears_diff > 1e-6).sum()),
                "positive_current_n": int(positive_mask.sum()),
                "positive_current_survives_max_abs_diff_from_p_has": float(survives_diff.max(skipna=True)) if survives_diff.notna().any() else None,
                "positive_current_survives_errors_gt_1e_6": int((survives_diff > 1e-6).sum()),
                "appears_not_null_when_current_positive": int(appears[positive_mask].notna().sum()),
                "survives_not_null_when_current_zero": int(survives[zero_mask].notna().sum()),
            })
    return pd.DataFrame(rows)


def parse_distribution(value: Any) -> dict[str, float] | None:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if not isinstance(value, dict):
        return None
    out: dict[str, float] = {}
    for key, prob in value.items():
        try:
            p = float(prob)
        except (TypeError, ValueError):
            continue
        if math.isfinite(p):
            out[str(key)] = p
    return out or None


def distribution_integrity(scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    c = 10
    pi = np.zeros((c + 1, c + 1))
    pi[0, 5] = 1.0
    rows.append({
        "check_name": "synthetic_axis_pi_e_q_zero_ebikes_total_5",
        "status": "pass" if pi[0, :].sum() == 1.0 and 1.0 - pi[0, :].sum() == 0.0 else "fail",
        "observed": json.dumps({
            "correct_p_zero_pi_0_sum_q": float(pi[0, :].sum()),
            "wrong_axis_pi_sum_e_q0": float(pi[:, 0].sum()),
        }),
        "expected": "p_zero=1,p_has=0 using pi[0, :].sum()",
    })
    rollout_cases = [
        (
            "rollout_zero_current_no_flows",
            dict(capacity=10, current_ebikes=0, current_total_bikes=5, ebike_departure_mean=0, classic_departure_mean=0, ebike_arrival_mean=0, classic_arrival_mean=0),
            {"p_has_ebike": 0.0, "p_zero": 1.0, "expected_ebikes": 0.0},
        ),
        (
            "rollout_one_current_no_flows",
            dict(capacity=10, current_ebikes=1, current_total_bikes=5, ebike_departure_mean=0, classic_departure_mean=0, ebike_arrival_mean=0, classic_arrival_mean=0),
            {"p_has_ebike": 1.0, "p_zero": 0.0, "expected_ebikes": 1.0},
        ),
        (
            "rollout_zero_current_classic_only_arrivals",
            dict(capacity=10, current_ebikes=0, current_total_bikes=5, ebike_departure_mean=0, classic_departure_mean=0, ebike_arrival_mean=0, classic_arrival_mean=8),
            {"p_has_ebike": 0.0},
        ),
    ]
    for name, kwargs, expected in rollout_cases:
        result = inventory_dp.rollout_inventory_distribution(**kwargs)
        observed = {
            "p_has_ebike": result.p_has_ebike,
            "p_zero": result.p_zero,
            "expected_ebikes": result.expected_ebikes,
            "p_count_ebikes": result.p_count_ebikes,
        }
        ok = True
        for key, expected_value in expected.items():
            ok = ok and abs(float(observed[key]) - float(expected_value)) <= 1e-9
        rows.append({
            "check_name": name,
            "status": "pass" if ok else "fail",
            "observed": json.dumps(observed, default=json_default, sort_keys=True),
            "expected": json.dumps(expected, sort_keys=True),
        })
    for model_key in predictor.MODEL_KEYS:
        for horizon in predictor.HORIZONS:
            dist_col = f"p_count_ebikes_{horizon}m_{model_key}"
            p_has_col = f"p_has_ebike_{horizon}m_{model_key}"
            p_zero_col = f"p_zero_{horizon}m_{model_key}"
            if dist_col not in scored.columns or p_has_col not in scored.columns:
                continue
            n = 0
            sum_errors = 0
            bucket_zero_errors = 0
            max_sum_error = 0.0
            max_pzero_error = 0.0
            for idx, value in scored[dist_col].items():
                dist = parse_distribution(value)
                if dist is None:
                    continue
                n += 1
                total = float(sum(dist.values()))
                sum_error = abs(total - 1.0)
                max_sum_error = max(max_sum_error, sum_error)
                if sum_error > 1e-6:
                    sum_errors += 1
                p_zero_from_dist = float(dist.get("0", 0.0))
                if p_zero_col in scored.columns:
                    p_zero = pd.to_numeric(pd.Series([scored.at[idx, p_zero_col]]), errors="coerce").iloc[0]
                else:
                    p_zero = 1.0 - pd.to_numeric(pd.Series([scored.at[idx, p_has_col]]), errors="coerce").iloc[0]
                if pd.notna(p_zero):
                    pzero_error = abs(float(p_zero) - p_zero_from_dist)
                    max_pzero_error = max(max_pzero_error, pzero_error)
                    if pzero_error > 1e-6:
                        bucket_zero_errors += 1
            rows.append({
                "check_name": f"scored_distribution_{model_key}_{horizon}m",
                "status": "pass" if sum_errors == 0 and bucket_zero_errors == 0 else "warn",
                "observed": json.dumps({
                    "n_distributions": n,
                    "sum_errors_gt_1e_6": sum_errors,
                    "max_sum_error": max_sum_error,
                    "p_zero_bucket_errors_gt_1e_6": bucket_zero_errors,
                    "max_p_zero_bucket_error": max_pzero_error,
                }, sort_keys=True),
                "expected": "distribution sums to 1 and p_count_ebikes['0'] equals p_zero",
            })
    return pd.DataFrame(rows)


def saturation_examples(scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_key in predictor.MODEL_KEYS:
        for horizon in predictor.HORIZONS:
            p_col = f"p_has_ebike_{horizon}m_{model_key}"
            if p_col not in scored.columns:
                continue
            p = pd.to_numeric(scored[p_col], errors="coerce")
            sample = scored.loc[p >= 0.995].copy()
            if sample.empty:
                continue
            sample = sample.assign(
                horizon=int(horizon),
                model_key=model_key,
                p_has=p.loc[sample.index],
                p_zero=pd.to_numeric(scored.get(f"p_zero_{horizon}m_{model_key}"), errors="coerce").loc[sample.index]
                if f"p_zero_{horizon}m_{model_key}" in scored.columns
                else np.nan,
                p_appears=pd.to_numeric(scored.get(f"p_appears_{horizon}m_{model_key}"), errors="coerce").loc[sample.index]
                if f"p_appears_{horizon}m_{model_key}" in scored.columns
                else np.nan,
                p_survives=pd.to_numeric(scored.get(f"p_survives_{horizon}m_{model_key}"), errors="coerce").loc[sample.index]
                if f"p_survives_{horizon}m_{model_key}" in scored.columns
                else np.nan,
                expected_ebikes=pd.to_numeric(scored.get(f"expected_ebikes_{horizon}m_{model_key}"), errors="coerce").loc[sample.index]
                if f"expected_ebikes_{horizon}m_{model_key}" in scored.columns
                else np.nan,
            )
            columns = [c for c in ZERO_EXAMPLE_BASE_COLUMNS if c in sample.columns]
            rows.extend(sample[[*columns, "horizon", "model_key", "p_has", "p_zero", "p_appears", "p_survives", "expected_ebikes"]].to_dict(orient="records"))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["p_has", "model_key", "horizon"], ascending=[False, True, True])


def zero_ebike_saturation_examples(scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    current = pd.to_numeric(scored.get("num_ebikes_available"), errors="coerce").fillna(-1)
    for model_key in predictor.MODEL_KEYS:
        for horizon in predictor.HORIZONS:
            p_col = f"p_has_ebike_{horizon}m_{model_key}"
            if p_col not in scored.columns:
                continue
            p = pd.to_numeric(scored[p_col], errors="coerce")
            mask = (current == 0) & (p >= 0.995)
            if not mask.any():
                continue
            for idx, row in scored.loc[mask].iterrows():
                payload = {column: clean_scalar(row.get(column)) for column in ZERO_EXAMPLE_BASE_COLUMNS if column in scored.columns}
                payload.update({
                    "horizon": int(horizon),
                    "model_key": model_key,
                    "p_has": clean_scalar(row.get(p_col)),
                    "p_zero": clean_scalar(row.get(f"p_zero_{horizon}m_{model_key}")),
                    "p_appears": clean_scalar(row.get(f"p_appears_{horizon}m_{model_key}")),
                    "p_survives": clean_scalar(row.get(f"p_survives_{horizon}m_{model_key}")),
                    "p_empirical": clean_scalar(row.get(f"p_empirical_{horizon}m_{model_key}")),
                    "p_learned": clean_scalar(row.get(f"p_learned_{horizon}m_{model_key}")),
                    "expected_ebikes": clean_scalar(row.get(f"expected_ebikes_{horizon}m_{model_key}")),
                    "p_count_ebikes_json": json.dumps(row.get(f"p_count_ebikes_{horizon}m_{model_key}"), default=json_default, sort_keys=True)
                    if isinstance(row.get(f"p_count_ebikes_{horizon}m_{model_key}"), dict)
                    else clean_scalar(row.get(f"p_count_ebikes_{horizon}m_{model_key}")),
                    "expected_ebike_departures": clean_scalar(row.get(f"expected_ebike_departures_{horizon}m_{model_key}")),
                    "expected_ebike_arrivals": clean_scalar(row.get(f"expected_ebike_arrivals_{horizon}m_{model_key}")),
                    "expected_classic_departures": clean_scalar(row.get(f"expected_classic_departures_{horizon}m_{model_key}")),
                    "expected_classic_arrivals": clean_scalar(row.get(f"expected_classic_arrivals_{horizon}m_{model_key}")),
                    "mu_e_depart": clean_scalar(row.get(f"mu_e_depart_{horizon}m_{model_key}")),
                    "mu_e_arrive": clean_scalar(row.get(f"mu_e_arrive_{horizon}m_{model_key}")),
                    "mu_c_depart": clean_scalar(row.get(f"mu_c_depart_{horizon}m_{model_key}")),
                    "mu_c_arrive": clean_scalar(row.get(f"mu_c_arrive_{horizon}m_{model_key}")),
                })
                rows.append(payload)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["p_has", "model_key", "horizon"], ascending=[False, True, True])


def add_derived_feature_columns(scored: pd.DataFrame) -> pd.DataFrame:
    out = scored.copy()
    if "current_ebikes_clipped" not in out.columns:
        out["current_ebikes_clipped"] = pd.to_numeric(out.get("num_ebikes_available"), errors="coerce").fillna(0).clip(0, 6)
    if "current_total_bikes_clipped" not in out.columns:
        out["current_total_bikes_clipped"] = pd.to_numeric(out.get("num_bikes_available"), errors="coerce").fillna(0).clip(0, 80)
    if "docks_available_clipped" not in out.columns:
        out["docks_available_clipped"] = pd.to_numeric(out.get("num_docks_available"), errors="coerce").fillna(0).clip(0, 80)
    if "capacity_clipped" not in out.columns:
        out["capacity_clipped"] = pd.to_numeric(out.get("capacity"), errors="coerce").fillna(0).clip(0, 80)
    return out


def feature_extremes(scored: pd.DataFrame) -> pd.DataFrame:
    working = add_derived_feature_columns(scored)
    rows: list[dict[str, Any]] = []
    for column in FEATURE_COLUMNS_OF_INTEREST:
        if column not in working.columns:
            rows.append({"feature": column, "present": False, "missing": len(working), "flag": "missing_feature"})
            continue
        numeric = pd.to_numeric(working[column], errors="coerce")
        finite = numeric[np.isfinite(numeric)]
        summary = numeric_summary(numeric)
        flags = []
        if column in {"station_same_hour_rate", "nearby_same_hour_rate", "station_neighbor_same_hour_rate"} and summary["median"] == 1.0:
            flags.append("median_eq_1")
        if column == "station_neighbor_recent_ebikes" and summary["p99"] is not None and summary["p99"] > 20:
            flags.append("neighbor_recent_ebikes_high")
        if column == "route_inbound_due_horizon" and summary["p99"] is not None and summary["p99"] > 10:
            flags.append("route_inbound_due_high")
        if column == "data_age_minutes" and summary["p95"] is not None and summary["p95"] > 20:
            flags.append("stale_status_high")
        rows.append({
            "feature": column,
            "present": True,
            **summary,
            "n_infinite": int(np.isinf(numeric).sum()),
            "n_nan": int(numeric.isna().sum()),
            "std": float(finite.std()) if not finite.empty else None,
            "flag": ",".join(flags),
        })
    integrity_checks = [
        (
            "num_ebikes_available_gt_num_bikes_available",
            (pd.to_numeric(working.get("num_ebikes_available"), errors="coerce") > pd.to_numeric(working.get("num_bikes_available"), errors="coerce")).sum(),
        ),
        (
            "bikes_plus_docks_gt_capacity_plus_2",
            (
                pd.to_numeric(working.get("num_bikes_available"), errors="coerce").fillna(0)
                + pd.to_numeric(working.get("num_docks_available"), errors="coerce").fillna(0)
                > pd.to_numeric(working.get("capacity"), errors="coerce").fillna(0) + 2
            ).sum(),
        ),
    ]
    for name, count in integrity_checks:
        rows.append({
            "feature": name,
            "present": True,
            "count": int(count),
            "missing": 0,
            "flag": "data_integrity_count",
        })
    return pd.DataFrame(rows)


def intensity_summary(scored: pd.DataFrame) -> pd.DataFrame:
    columns = [
        c for c in scored.columns
        if re.search(r"(mu_[ec]_(?:depart|arrive)|theta|zero_inflation|expected_.*(?:departures|arrivals))", str(c))
    ]
    rows: list[dict[str, Any]] = []
    for column in columns:
        numeric = pd.to_numeric(scored[column], errors="coerce")
        finite = numeric[np.isfinite(numeric)]
        if finite.empty:
            continue
        flags = []
        if "mu_e_arrive" in column and int((finite > 5).sum()) > 0 and "_5m" in column:
            flags.append("mu_e_arrive_gt_5_at_5m")
        if "mu_e_arrive" in column and int((finite > 10).sum()) > 0 and "_10m" in column:
            flags.append("mu_e_arrive_gt_10_at_10m")
        if re.search(r"(depart|arrive)", column) and int((finite < 0).sum()) > 0:
            flags.append("negative_intensity")
        if "theta" in column and int((finite <= 0).sum()) > 0:
            flags.append("theta_nonpositive")
        if "zero_inflation" in column and int(((finite < 0) | (finite > 1)).sum()) > 0:
            flags.append("zero_inflation_outside_0_1")
        rows.append({"column": column, **numeric_summary(numeric), "flags": ",".join(flags)})
    return pd.DataFrame(rows).sort_values("column") if rows else pd.DataFrame()


def db_recent_forecast_outputs(conn) -> dict[str, pd.DataFrame]:
    if not table_exists(conn, "model_forecasts"):
        empty = pd.DataFrame({"query_error": ["model_forecasts table missing"]})
        return {
            "db_recent_forecast_summary": empty,
            "db_recent_forecast_by_inventory": empty,
            "db_zero_current_saturation": empty,
            "db_recent_forecast_sources": empty,
        }
    return {
        "db_recent_forecast_summary": query_df(
            conn,
            """
            SELECT
              model_key,
              horizon_minutes,
              COUNT(*) AS n,
              MIN(p_has_ebike) AS min_p,
              AVG(p_has_ebike) AS avg_p,
              MAX(p_has_ebike) AS max_p,
              SUM(CASE WHEN p_has_ebike >= 0.995 THEN 1 ELSE 0 END) AS n_ge_995,
              SUM(CASE WHEN p_has_ebike = 1.0 THEN 1 ELSE 0 END) AS n_eq_1
            FROM model_forecasts
            WHERE forecasted_at >= now() - INTERVAL '2 hours'
            GROUP BY model_key, horizon_minutes
            ORDER BY model_key, horizon_minutes
            """,
        ),
        "db_recent_forecast_by_inventory": query_df(
            conn,
            """
            SELECT
              model_key,
              horizon_minutes,
              current_ebikes,
              COUNT(*) AS n,
              MIN(p_has_ebike) AS min_p,
              AVG(p_has_ebike) AS avg_p,
              MAX(p_has_ebike) AS max_p,
              SUM(CASE WHEN p_has_ebike >= 0.995 THEN 1 ELSE 0 END) AS n_ge_995
            FROM model_forecasts
            WHERE forecasted_at >= now() - INTERVAL '2 hours'
            GROUP BY model_key, horizon_minutes, current_ebikes
            ORDER BY model_key, horizon_minutes, current_ebikes
            """,
        ),
        "db_zero_current_saturation": query_df(
            conn,
            """
            SELECT *
            FROM model_forecasts
            WHERE forecasted_at >= now() - INTERVAL '2 hours'
              AND current_ebikes = 0
              AND p_has_ebike >= 0.995
            LIMIT 100
            """,
        ),
        "db_recent_forecast_sources": query_df(
            conn,
            """
            SELECT
              source,
              model_key,
              COUNT(*) AS n,
              MIN(forecasted_at) AS first_forecast,
              MAX(forecasted_at) AS last_forecast
            FROM model_forecasts
            WHERE forecasted_at >= now() - INTERVAL '24 hours'
            GROUP BY source, model_key
            ORDER BY source, model_key
            """,
        ),
    }


def live_prediction_cache_summary(conn) -> pd.DataFrame:
    if not table_exists(conn, "live_station_predictions"):
        return pd.DataFrame({"query_error": ["live_station_predictions table missing"]})
    return query_df(
        conn,
        """
        SELECT
          model_key,
          active_model_key,
          horizon_minutes,
          COUNT(*) AS n,
          MIN(as_of) AS first_as_of,
          MAX(as_of) AS last_as_of,
          MIN(p_has_ebike) AS min_p,
          AVG(p_has_ebike) AS avg_p,
          MAX(p_has_ebike) AS max_p,
          SUM(CASE WHEN p_has_ebike >= 0.995 THEN 1 ELSE 0 END) AS n_ge_995,
          SUM(CASE WHEN p_has_ebike = 1.0 THEN 1 ELSE 0 END) AS n_eq_1
        FROM live_station_predictions
        WHERE as_of >= now() - INTERVAL '2 hours'
        GROUP BY model_key, active_model_key, horizon_minutes
        ORDER BY model_key, horizon_minutes
        """,
    )


def dashboard_formatting_locations() -> pd.DataFrame:
    paths = [SRC / "divvy" / name for name in ["dashboard.py", "recommendations.py", "api.py"]]
    patterns = [
        (re.compile(r"\.0%"), "zero_decimal_percent"),
        (re.compile(r"\.1%"), "one_decimal_percent"),
        (re.compile(r"format=[\"']percent[\"']"), "streamlit_percent_column"),
        (re.compile(r"round\s*\([^)]*\*\s*100"), "round_percent"),
        (re.compile(r"int\s*\([^)]*\*\s*100"), "int_percent"),
        (re.compile(r"st\.metric"), "metric"),
        (re.compile(r"p_has_ebike|p_arrival|rank_probability|walk_adjusted_score"), "probability_reference"),
    ]
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        current_func = None
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("def "):
                current_func = stripped.split("(", 1)[0].replace("def ", "")
            hits = [label for regex, label in patterns if regex.search(line)]
            if not hits:
                continue
            rows.append({
                "file": str(path),
                "line": int(lineno),
                "function": current_func,
                "patterns": ",".join(hits),
                "source": stripped,
                "rounds_0_995_to_100": ".0%" in line,
            })
    return pd.DataFrame(rows)


def cold_start_health(conn, suite, args) -> dict[str, Any]:
    status_row = query_one(
        conn,
        """
        SELECT COUNT(*), COUNT(DISTINCT station_id), MIN(last_reported), MAX(last_reported)
        FROM station_status
        """,
    )
    forecast_total = query_one(conn, "SELECT COUNT(*) FROM model_forecasts") if table_exists(conn, "model_forecasts") else (0,)
    outcomes_total = query_one(conn, "SELECT COUNT(*) FROM model_outcomes") if table_exists(conn, "model_outcomes") else (0,)
    outcomes_24h = query_one(
        conn,
        "SELECT COUNT(*) FROM model_outcomes WHERE resolved_at >= now() - INTERVAL '24 hours'",
    ) if table_exists(conn, "model_outcomes") else (0,)
    label_report: dict[str, Any] = {
        "history_hours": args.label_history_hours,
        "anchor_every_min": args.label_anchor_every_min,
        "n_examples": None,
        "positive_rate": None,
        "by_horizon": [],
        "error": None,
    }
    try:
        examples = label_builder.build_leak_free_examples(
            conn,
            horizons=tuple(int(h) for h in predictor.HORIZONS),
            anchor_every_min=int(args.label_anchor_every_min),
            history_hours=int(args.label_history_hours),
            max_source_rows=int(args.label_max_source_rows),
        )
        label_report["n_examples"] = int(len(examples))
        if not examples.empty and "has_ebike" in examples.columns:
            label_report["positive_rate"] = float(examples["has_ebike"].mean())
            label_report["positive"] = int(examples["has_ebike"].sum())
            label_report["negative"] = int(len(examples) - examples["has_ebike"].sum())
            by_horizon = (
                examples.groupby("horizon_minutes")
                .agg(n=("has_ebike", "size"), positives=("has_ebike", "sum"), positive_rate=("has_ebike", "mean"))
                .reset_index()
            )
            label_report["by_horizon"] = by_horizon.to_dict(orient="records")
    except Exception as exc:
        label_report["error"] = f"{type(exc).__name__}: {exc}"
    source_text = (SRC / "divvy" / "label_builder.py").read_text()
    label_report["label_builder_method"] = {
        "uses_backward_asof_or_searchsorted_previous": "searchsorted(reported, target64, side=\"right\") - 1" in source_text,
        "uses_forward_asof": "direction=\"forward\"" in source_text or "direction='forward'" in source_text,
        "function": "label_builder.build_leak_free_examples",
    }
    return {
        "station_status_rows": int(status_row[0] or 0) if status_row else 0,
        "station_status_distinct_stations": int(status_row[1] or 0) if len(status_row) > 1 else 0,
        "station_status_first_reported": status_row[2] if len(status_row) > 2 else None,
        "station_status_last_reported": status_row[3] if len(status_row) > 3 else None,
        "model_forecasts_total": int(forecast_total[0] or 0) if forecast_total else 0,
        "model_outcomes_total": int(outcomes_total[0] or 0) if outcomes_total else 0,
        "model_outcomes_resolved_24h": int(outcomes_24h[0] or 0) if outcomes_24h else 0,
        "suite_models": suite.summary(),
        "label_examples": label_report,
    }


def format_markdown_report(report: dict[str, Any], tables: dict[str, pd.DataFrame], output_paths: dict[str, str]) -> str:
    active = report.get("active_model_key")
    active_source = report.get("active_model_source")
    best = report.get("best_evaluated_model_key")
    best_usable = report.get("best_usable_model_key")
    best_trained_sota = report.get("best_trained_sota_model_key")
    prob_summary = tables.get("probability_summary_by_model_horizon", pd.DataFrame())
    bucket_summary = tables.get("probability_summary_by_current_ebike_bucket", pd.DataFrame())
    zero_examples = tables.get("zero_ebike_saturation_examples", pd.DataFrame())
    identity = tables.get("probability_identity_checks", pd.DataFrame())
    distribution = tables.get("distribution_integrity_report", pd.DataFrame())
    active_mismatch = tables.get("active_vs_suffixed_column_mismatch", pd.DataFrame())
    formatting = tables.get("dashboard_formatting_locations", pd.DataFrame())

    active_prob = prob_summary[prob_summary["model_key"] == active] if not prob_summary.empty and active else pd.DataFrame()
    active_lines = []
    if not active_prob.empty:
        for _, row in active_prob.sort_values("horizon_minutes").iterrows():
            active_lines.append(
                f"| {row['model_key']} | {int(row['horizon_minutes'])} | {row['min']:.6f} | "
                f"{row['mean']:.6f} | {row['max']:.6f} | {int(row['n_ge_0_995'])} | "
                f"{int(row['zero_current_n_ge_0_995'])} |"
            )
    else:
        active_lines.append("| n/a | n/a | n/a | n/a | n/a | n/a | n/a |")

    zero_current_count = int(len(zero_examples)) if not zero_examples.empty else 0
    active_zero_current_count = 0
    if not active_prob.empty and "zero_current_n_ge_0_995" in active_prob.columns:
        active_zero_current_count = int(
            pd.to_numeric(active_prob["zero_current_n_ge_0_995"], errors="coerce").fillna(0).sum()
        )
    identity_failures = 0
    if not identity.empty:
        identity_failures = int((pd.to_numeric(identity["n_errors_gt_1e_6"], errors="coerce").fillna(0) > 0).sum())
    distribution_failures = 0
    if not distribution.empty:
        distribution_failures = int((distribution["status"] == "fail").sum())
    mismatch_count = 0
    if not active_mismatch.empty and "n_mismatched_gt_1e_9" in active_mismatch.columns:
        mismatch_count = int(pd.to_numeric(active_mismatch["n_mismatched_gt_1e_9"], errors="coerce").fillna(0).sum())
    rounds_to_100 = False
    if not formatting.empty and "rounds_0_995_to_100" in formatting.columns:
        rounds_to_100 = bool(formatting["rounds_0_995_to_100"].fillna(False).any())

    causes = report.get("likely_causes", [])
    causes_lines = [f"{idx}. {cause}" for idx, cause in enumerate(causes, start=1)] or ["1. No likely causes ranked; inspect CSV outputs."]

    bucket_active = bucket_summary[bucket_summary["model_key"] == active] if not bucket_summary.empty and active else pd.DataFrame()
    bucket_lines = []
    if not bucket_active.empty:
        for _, row in bucket_active.sort_values(["horizon_minutes", "current_ebike_bucket"]).iterrows():
            bucket_lines.append(
                f"| {int(row['horizon_minutes'])} | {row['current_ebike_bucket']} | {int(row['n'])} | "
                f"{row['mean_p_has']:.6f} | {row['max_p_has']:.6f} | {int(row['n_ge_0_995'])} |"
            )
    else:
        bucket_lines.append("| n/a | n/a | n/a | n/a | n/a | n/a |")

    paths_lines = [f"- `{name}`: `{path}`" for name, path in sorted(output_paths.items())]

    return "\n".join([
        "# Probability Saturation Diagnostic Report",
        "",
        "## Executive summary",
        f"- active_model_key: `{active}`",
        f"- active_model_source: `{active_source}`",
        f"- best_evaluated_model_key: `{best}`",
        f"- best_usable_model_key: `{best_usable}`",
        f"- best_trained_sota_model_key: `{best_trained_sota}`",
        f"- HORIZONS: `{tuple(report.get('horizons') or [])}`",
        f"- n_stations_scored: `{report.get('n_stations_scored')}`",
        f"- n_recent_model_forecasts_2h: `{report.get('n_recent_model_forecasts_2h')}`",
        f"- n_recent_model_outcomes_24h: `{report.get('n_recent_model_outcomes_24h')}`",
        f"- active zero-current scored rows with p_has>=0.995: `{active_zero_current_count}`",
        f"- all-model zero-current scored rows with p_has>=0.995: `{zero_current_count}`",
        f"- p_has+p_zero identity failures: `{identity_failures}` model/horizon rows",
        f"- DP collapse axis failures: `{distribution_failures}`",
        f"- active-vs-suffixed mismatches: `{mismatch_count}` cells",
        f"- dashboard/recommendation zero-decimal percent rounding found: `{rounds_to_100}`",
        "",
        "## Top likely causes",
        *causes_lines,
        "",
        "## Active raw probability summary",
        "| model | horizon | min | mean | max | n>=0.995 | zero-current n>=0.995 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        *active_lines,
        "",
        "## Active probability by current eBike bucket",
        "| horizon | bucket | n | mean p_has | max p_has | n>=0.995 |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
        *bucket_lines,
        "",
        "## Notes",
        f"- Intensity debug columns exposed: `{report.get('has_mu_intensity_debug_columns')}`",
        f"- Label builder method: `{report.get('label_builder_method')}`",
        f"- Display note: zero-decimal percent rounding paths found: `{rounds_to_100}`; shared display labels cap model predictions at `≥99%`.",
        "",
        "## Immediate recommended fixes",
        "- Change dashboard/API display so capped values do not render as literal 100%; show raw debug values or display `>=99%` for capped values.",
        "- Add cold-start caps for SOTA fallback and inventory-world probabilities until enough same-station outcomes exist.",
        "- Expose CC-NISSM/DG-NISSM intensity debug columns (`mu_e_arrive`, `mu_e_depart`, etc.) in diagnostic mode.",
        "- Keep the DP axis convention tests in place; current diagnostics expect `pi[e, q]`, `p_zero = pi[0, :].sum()`.",
        "- Keep leak-free backward-as-of labels; do not reintroduce forward-as-of labels.",
        "",
        "## Generated files",
        *paths_lines,
        "",
    ])


def rank_likely_causes(report: dict[str, Any], tables: dict[str, pd.DataFrame]) -> list[str]:
    causes: list[str] = []
    formatting = tables.get("dashboard_formatting_locations", pd.DataFrame())
    if not formatting.empty and bool(formatting.get("rounds_0_995_to_100", pd.Series(False)).fillna(False).any()):
        causes.append("Dashboard/recommendation formatting uses zero-decimal percent labels, so raw 0.995 or 0.999 displays as 100%.")

    prob_summary = tables.get("probability_summary_by_model_horizon", pd.DataFrame())
    if not prob_summary.empty:
        saturated = prob_summary[pd.to_numeric(prob_summary["n_ge_0_995"], errors="coerce").fillna(0) > 0]
        if not saturated.empty:
            models = ", ".join(sorted(saturated["model_key"].astype(str).unique()))
            causes.append(f"Raw scored probabilities are clipped or saturated at >=0.995 for model(s): {models}.")
        zero_sat = prob_summary[pd.to_numeric(prob_summary["zero_current_n_ge_0_995"], errors="coerce").fillna(0) > 0]
        if not zero_sat.empty:
            models = ", ".join(sorted(zero_sat["model_key"].astype(str).unique()))
            causes.append(f"Zero-current stations also saturate for model(s): {models}; inspect arrival pressure and p_zero collapse.")

    intensity = tables.get("intensity_summary", pd.DataFrame())
    if intensity.empty or not any("mu_" in str(c) for c in intensity.get("column", [])):
        causes.append("SOTA models do not expose raw intensity debug columns, so arrival-intensity saturation cannot be directly ruled out.")
    else:
        flagged = intensity[intensity.get("flags", pd.Series("", index=intensity.index)).astype(str) != ""]
        if not flagged.empty:
            causes.append("Intensity-like outputs have flagged extremes; inspect `intensity_summary.csv`.")

    active_mismatch = tables.get("active_vs_suffixed_column_mismatch", pd.DataFrame())
    if not active_mismatch.empty and "n_mismatched_gt_1e_9" in active_mismatch:
        mismatch_count = int(pd.to_numeric(active_mismatch["n_mismatched_gt_1e_9"], errors="coerce").fillna(0).sum())
        if mismatch_count:
            causes.append("Unsuffixed dashboard columns differ from the active model's suffixed columns.")

    cold = report.get("cold_start_health") or {}
    first = cold.get("station_status_first_reported")
    last = cold.get("station_status_last_reported")
    if first and last:
        hours = (pd.Timestamp(last) - pd.Timestamp(first)).total_seconds() / 3600.0
        if hours < 24:
            causes.append(f"Database is in cold start: station_status spans only {hours:.2f} hours.")

    if not causes:
        causes.append("No direct saturation cause was isolated; inspect all probability summaries and recent forecast logs.")
    return causes[:5]


def run(args: argparse.Namespace) -> int:
    timestamp = utc_now().strftime("%Y%m%dT%H%M%SZ")
    output_dir = ROOT / "diagnostics" / f"probability_saturation_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths: dict[str, str] = {}
    tables: dict[str, pd.DataFrame] = {}
    report: dict[str, Any] = {
        "generated_at": utc_now(),
        "project_root": str(ROOT),
        "db_path": str(config.DB_PATH),
        "read_db_path": str(config.READ_DB_PATH),
        "read_only_connection_path": db._connection_path(True),
        "model_keys": list(predictor.MODEL_KEYS),
        "horizons": list(predictor.HORIZONS),
        "active_model_policy": predictor.ACTIVE_MODEL_POLICY,
        "forced_active_model_key": predictor.FORCED_ACTIVE_MODEL_KEY,
    }

    with db.session(read_only=True) as conn:
        status_bounds = query_one(conn, "SELECT MIN(last_reported), MAX(last_reported) FROM station_status")
        report["station_status_first_reported"] = status_bounds[0] if len(status_bounds) > 0 else None
        report["station_status_last_reported"] = status_bounds[1] if len(status_bounds) > 1 else None

        candidates = all_station_candidates(conn)
        report["n_station_candidates"] = int(len(candidates))
        scored, suite = predictor.score_candidates(
            conn,
            candidates,
            search_radius_km=None,
            model_keys=tuple(predictor.MODEL_KEYS),
            debug=True,
        )
        report.update({
            "active_model_key": suite.active_key,
            "active_model_source": suite.active_source,
            "best_evaluated_model_key": suite.best_evaluated_model_key,
            "best_usable_model_key": suite.best_usable_model_key,
            "best_sota_model_key": suite.best_sota_model_key,
            "best_trained_sota_model_key": suite.best_trained_sota_model_key,
            "best_baseline_model_key": suite.best_baseline_model_key,
            "suite_summary": suite.summary(),
            "n_stations_scored": int(len(scored)),
        })

        all_station_path = output_dir / "all_station_scored.csv"
        write_csv(scored, all_station_path)
        output_paths["all_station_scored.csv"] = str(all_station_path)

        tables["all_probability_column_summary"] = probability_column_summary(scored)
        tables["probability_summary_by_model_horizon"] = model_horizon_probability_summary(scored, suite)
        tables["probability_summary_by_current_ebike_bucket"] = probability_by_current_bucket(scored)
        tables["saturation_examples"] = saturation_examples(scored)
        tables["zero_ebike_saturation_examples"] = zero_ebike_saturation_examples(scored)
        tables["active_vs_suffixed_column_mismatch"] = active_vs_suffixed(scored, suite.active_key)
        tables["probability_identity_checks"] = p_zero_identity_checks(scored)
        tables["distribution_integrity_report"] = distribution_integrity(scored)
        tables["feature_extremes"] = feature_extremes(scored)
        tables["intensity_summary"] = intensity_summary(scored)
        tables["dashboard_formatting_locations"] = dashboard_formatting_locations()

        for name, df in db_recent_forecast_outputs(conn).items():
            tables[name] = df
        tables["live_prediction_cache_summary"] = live_prediction_cache_summary(conn)

        report["cold_start_health"] = cold_start_health(conn, suite, args)
        report["label_builder_method"] = report["cold_start_health"]["label_examples"].get("label_builder_method")
        report["has_mu_intensity_debug_columns"] = bool(
            not tables["intensity_summary"].empty
            and tables["intensity_summary"]["column"].astype(str).str.contains(r"mu_").any()
        )
        report["n_recent_model_forecasts_2h"] = int(
            query_one(conn, "SELECT COUNT(*) FROM model_forecasts WHERE forecasted_at >= now() - INTERVAL '2 hours'")[0] or 0
        ) if table_exists(conn, "model_forecasts") else 0
        report["n_recent_model_outcomes_24h"] = int(
            query_one(conn, "SELECT COUNT(*) FROM model_outcomes WHERE resolved_at >= now() - INTERVAL '24 hours'")[0] or 0
        ) if table_exists(conn, "model_outcomes") else 0

    file_names = {
        "all_probability_column_summary": "all_probability_column_summary.csv",
        "probability_summary_by_model_horizon": "probability_summary_by_model_horizon.csv",
        "probability_summary_by_current_ebike_bucket": "probability_summary_by_current_ebike_bucket.csv",
        "saturation_examples": "saturation_examples.csv",
        "zero_ebike_saturation_examples": "zero_ebike_saturation_examples.csv",
        "active_vs_suffixed_column_mismatch": "active_vs_suffixed_column_mismatch.csv",
        "probability_identity_checks": "probability_identity_checks.csv",
        "distribution_integrity_report": "distribution_integrity_report.csv",
        "feature_extremes": "feature_extremes.csv",
        "intensity_summary": "intensity_summary.csv",
        "dashboard_formatting_locations": "dashboard_formatting_locations.csv",
        "db_recent_forecast_summary": "db_recent_forecast_summary.csv",
        "db_recent_forecast_by_inventory": "db_recent_forecast_by_inventory.csv",
        "db_zero_current_saturation": "db_zero_current_saturation.csv",
        "db_recent_forecast_sources": "db_recent_forecast_sources.csv",
        "live_prediction_cache_summary": "live_prediction_cache_summary.csv",
    }
    for table_name, file_name in file_names.items():
        df = tables.get(table_name, pd.DataFrame())
        path = output_dir / file_name
        write_csv(df, path)
        output_paths[file_name] = str(path)

    report["likely_causes"] = rank_likely_causes(report, tables)
    report_path = output_dir / "diagnostic_report.json"
    write_json(report, report_path)
    output_paths["diagnostic_report.json"] = str(report_path)

    markdown = format_markdown_report(report, tables, output_paths)
    markdown_path = output_dir / "diagnostic_report.md"
    markdown_path.write_text(markdown)
    output_paths["diagnostic_report.md"] = str(markdown_path)

    print(markdown)
    print("Generated files:")
    for name, path in sorted(output_paths.items()):
        print(f"{name}: {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnose Divvy probability saturation without mutating the database.")
    parser.add_argument("--label-history-hours", type=int, default=24)
    parser.add_argument("--label-anchor-every-min", type=int, default=10)
    parser.add_argument("--label-max-source-rows", type=int, default=500_000)
    args = parser.parse_args(argv)
    try:
        return run(args)
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
