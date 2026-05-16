from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Iterable

import duckdb
import numpy as np
import pandas as pd

from . import config, db, predictor


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _ids_clause(ids: Iterable[str]) -> tuple[str, list[str]]:
    values = [str(value) for value in ids if value is not None]
    if not values:
        return "", []
    return ",".join(["?"] * len(values)), values


def _json_text(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return json.dumps(value, separators=(",", ":"), sort_keys=True)
    if isinstance(value, np.generic):
        value = value.item()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return json.dumps(value, separators=(",", ":"), sort_keys=True, default=str)


def all_station_candidates(conn: duckdb.DuckDBPyConnection, max_staleness_minutes: int = 10) -> pd.DataFrame:
    now = _utc_now()
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
        WHERE s.lat IS NOT NULL
          AND s.lon IS NOT NULL
          AND l.last_reported >= ? - (? * INTERVAL '1 minute')
        """,
        [now, int(max_staleness_minutes)],
    ).df()


def _value(row: pd.Series, base: str, horizon: int, model_key: str):
    value = row.get(f"{base}_{horizon}m_{model_key}")
    if value is None or (not isinstance(value, (dict, list)) and pd.isna(value)):
        if model_key == row.get("active_model_key"):
            value = row.get(f"{base}_{horizon}m")
    return value


def _count_ebikes_pmf_payload(row: pd.Series, horizon: int, model_key: str):
    """Build the cache JSON payload for ``p_count_ebikes``.

    Folds the full PMF (length ``capacity+1``) into the bucketed dict under the
    ``__full__`` key, keeping the legacy bucket keys (``"0".."5_plus"``) so
    existing consumers are unaffected. Returns ``None`` if no data is available.
    """
    bucket = _value(row, "p_count_ebikes", int(horizon), model_key)
    full = _value(row, "p_count_ebikes_full", int(horizon), model_key)
    bucket_dict = bucket if isinstance(bucket, dict) else None
    full_list = list(full) if isinstance(full, (list, tuple)) and len(full) > 0 else None
    if bucket_dict is None and full_list is None:
        return None
    out: dict = dict(bucket_dict) if bucket_dict is not None else {}
    if full_list is not None:
        out["__full__"] = full_list
    return out


def _prediction_rows_from_scored(
    scored: pd.DataFrame,
    suite: predictor.FittedModelSuite,
    selected_model_keys: list[str],
    horizons: tuple[int, ...],
    as_of: datetime,
) -> list[tuple]:
    rows = []
    active_key = suite.active_key
    model_by_key = suite.models
    for _, row in scored.iterrows():
        for model_key in selected_model_keys:
            if model_key not in predictor.MODEL_KEYS:
                continue
            model = model_by_key.get(model_key)
            for horizon in horizons:
                p = _value(row, "p_has_ebike", int(horizon), model_key)
                if p is None or pd.isna(p):
                    continue
                rows.append((
                    as_of,
                    model_key,
                    (model.model_version if model else predictor.MODEL_SPECS[model_key]["version"]),
                    (model.artifact_id if model else None),
                    active_key,
                    str(row["station_id"]),
                    int(horizon),
                    float(p),
                    _maybe_float(_value(row, "p_zero", int(horizon), model_key)),
                    _maybe_float(_value(row, "p_appears", int(horizon), model_key)),
                    _maybe_float(_value(row, "p_survives", int(horizon), model_key)),
                    _maybe_float(_value(row, "expected_ebikes", int(horizon), model_key)),
                    _maybe_float(_value(row, "expected_total_bikes", int(horizon), model_key)),
                    _json_text(_count_ebikes_pmf_payload(row, int(horizon), model_key)),
                    _json_text(_value(row, "p_count_total", int(horizon), model_key)),
                    _maybe_float(_value(row, "p_capacity_violation", int(horizon), model_key)),
                    _maybe_float(_value(row, "p_dock_constrained_arrival", int(horizon), model_key)),
                    _maybe_float(_value(row, "reliable_probability_lcb", int(horizon), model_key)) or float(p),
                    "global",
                    row.get("feature_snapshot_id"),
                    _maybe_float(row.get("data_age_minutes")),
                ))
    return rows


def _write_prediction_rows(conn: duckdb.DuckDBPyConnection, rows: list[tuple]) -> None:
    if rows:
        conn.executemany(
            """
            INSERT OR REPLACE INTO live_station_predictions (
              as_of, model_key, model_version, artifact_id, active_model_key,
              station_id, horizon_minutes, p_has_ebike, p_zero, p_appears,
              p_survives, expected_ebikes, expected_total_bikes,
              p_count_ebikes_json, p_count_total_json, p_capacity_violation,
              p_dock_constrained_arrival, reliable_probability_lcb,
              calibration_group, feature_snapshot_id, data_age_minutes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
    retention = _utc_now() - timedelta(hours=int(config.LIVE_PREDICTION_RETENTION_HOURS))
    conn.execute("DELETE FROM live_station_predictions WHERE as_of < ?", [retention])


def refresh_live_station_predictions(
    conn: duckdb.DuckDBPyConnection,
    model_keys: Iterable[str] | None = None,
    active_only: bool = True,
    horizons: tuple[int, ...] = predictor.HORIZONS,
    max_staleness_minutes: int = 10,
) -> dict:
    db.init_schema(conn)
    candidates = all_station_candidates(conn, max_staleness_minutes=max_staleness_minutes)
    if candidates.empty:
        return {"status": "no_fresh_stations", "rows_written": 0, "stations_scored": 0}
    if model_keys is None:
        suite = predictor.get_availability_model_suite(conn)
        active_key = suite.active_key
        selected_model_keys = [active_key] if active_only else list(predictor.MODEL_KEYS)
    else:
        selected_model_keys = [str(key) for key in model_keys]
    scored, suite = predictor.score_candidates(
        conn,
        candidates,
        horizons=horizons,
        model_keys=tuple(selected_model_keys),
    )
    as_of = _utc_now()
    active_key = suite.active_key
    rows = _prediction_rows_from_scored(scored, suite, selected_model_keys, horizons, as_of)
    _write_prediction_rows(conn, rows)
    return {
        "status": "ok",
        "rows_written": len(rows),
        "stations_scored": int(len(scored)),
        "as_of": as_of.isoformat(),
        "active_model_key": active_key,
        "model_keys": selected_model_keys,
    }


def refresh_live_station_predictions_coexisting(
    model_keys: Iterable[str] | None = None,
    active_only: bool = True,
    horizons: tuple[int, ...] = predictor.HORIZONS,
    max_staleness_minutes: int = 10,
) -> dict:
    """Refresh live predictions without holding the writer lock while scoring."""
    with db.session(read_only=True, retries=60, retry_sleep=1.0) as read_conn:
        candidates = all_station_candidates(read_conn, max_staleness_minutes=max_staleness_minutes)
        if candidates.empty:
            return {"status": "no_fresh_stations", "rows_written": 0, "stations_scored": 0}
        if model_keys is None:
            suite = predictor.get_availability_model_suite(read_conn)
            active_key = suite.active_key
            selected_model_keys = [active_key] if active_only else list(predictor.MODEL_KEYS)
        else:
            selected_model_keys = [str(key) for key in model_keys]
        scored, suite = predictor.score_candidates(
            read_conn,
            candidates,
            horizons=horizons,
            model_keys=tuple(selected_model_keys),
        )
        as_of = _utc_now()
        active_key = suite.active_key
        rows = _prediction_rows_from_scored(scored, suite, selected_model_keys, horizons, as_of)
    with db.session(read_only=False, retries=180, retry_sleep=1.0) as write_conn:
        db.init_schema(write_conn)
        _write_prediction_rows(write_conn, rows)
        try:
            db.refresh_read_replica()
        except Exception:
            pass
    return {
        "status": "ok",
        "rows_written": len(rows),
        "stations_scored": int(len(scored)),
        "as_of": as_of.isoformat(),
        "active_model_key": active_key,
        "model_keys": selected_model_keys,
    }


def _maybe_float(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def latest_prediction_cache(
    conn: duckdb.DuckDBPyConnection,
    model_key: str | Iterable[str],
    station_ids: Iterable[str],
    horizons: tuple[int, ...] = predictor.HORIZONS,
    max_age_minutes: float = 5,
) -> pd.DataFrame:
    station_ids_list = [str(station_id) for station_id in station_ids]
    if not station_ids_list:
        return pd.DataFrame()
    if isinstance(model_key, str):
        model_keys = [model_key]
    else:
        model_keys = [str(key) for key in model_key]
    station_placeholders, station_params = _ids_clause(station_ids_list)
    model_placeholders, model_params = _ids_clause(model_keys)
    horizon_values = [int(h) for h in horizons]
    horizon_placeholders = ",".join(["?"] * len(horizon_values))
    now = _utc_now()
    return conn.execute(
        f"""
        WITH ranked AS (
          SELECT *,
                 ROW_NUMBER() OVER (
                   PARTITION BY model_key, station_id, horizon_minutes
                   ORDER BY as_of DESC
                 ) AS rn
          FROM live_station_predictions
          WHERE station_id IN ({station_placeholders})
            AND model_key IN ({model_placeholders})
            AND horizon_minutes IN ({horizon_placeholders})
            AND as_of >= ? - (? * INTERVAL '1 minute')
        )
        SELECT *
        FROM ranked
        WHERE rn = 1
        """,
        [*station_params, *model_params, *horizon_values, now, float(max_age_minutes)],
    ).df()


def cache_to_wide_candidates(
    candidates: pd.DataFrame,
    cache_rows: pd.DataFrame,
    *,
    active_model_key: str,
    search_radius_km: float | None = None,
) -> pd.DataFrame:
    if candidates.empty or cache_rows.empty:
        return pd.DataFrame()
    out = candidates.copy()
    out["active_model_key"] = active_model_key
    out["active_model_source"] = "live_prediction_cache"
    out["best_evaluated_model_key"] = None
    out["best_sota_model_key"] = None
    out["forecasted_at"] = pd.to_datetime(cache_rows["as_of"]).max().to_pydatetime()
    for _, row in cache_rows.iterrows():
        station_id = str(row["station_id"])
        model_key = str(row["model_key"])
        horizon = int(row["horizon_minutes"])
        mask = out["station_id"].astype(str) == station_id
        if not mask.any():
            continue
        suffix = f"_{horizon}m_{model_key}"
        p_has = _maybe_float(row.get("p_has_ebike"))
        p_zero = _maybe_float(row.get("p_zero"))
        p_appears = _maybe_float(row.get("p_appears"))
        p_survives = _maybe_float(row.get("p_survives"))
        if p_has is not None and model_key in predictor.SOTA_PRIMARY_MODEL_KEYS:
            current_ebikes = int(_maybe_float(out.loc[mask, "num_ebikes_available"].iloc[0]) or 0)
            model_meta = {
                "model_key": model_key,
                "label": predictor.MODEL_SPECS.get(model_key, {}).get("label", model_key),
                "artifact_id": row.get("artifact_id"),
                "method": row.get("model_version"),
                "usable": True,
            }
            cache_is_bootstrap = predictor.is_bootstrap_or_fallback(model_meta)
            p_guarded = predictor.apply_cold_start_probability_guard(
                p_has,
                current_ebikes=current_ebikes,
                horizon_minutes=horizon,
                model_meta=model_meta,
                n_resolved=0 if cache_is_bootstrap else 1000,
            )
            if cache_is_bootstrap:
                station_row = out.loc[mask].iloc[0].copy()
                station_row["num_ebikes_available"] = current_ebikes
                p_guarded = predictor.cap_zero_current_without_inbound_support(station_row, p_guarded)
            p_has = p_guarded
            p_zero = 1.0 - p_guarded
            p_appears = p_guarded if current_ebikes <= 0 else np.nan
            p_survives = p_guarded if current_ebikes > 0 else np.nan
        values = {
            f"p_has_ebike{suffix}": p_has,
            f"p_zero{suffix}": p_zero,
            f"p_appears{suffix}": p_appears,
            f"p_survives{suffix}": p_survives,
            f"expected_ebikes{suffix}": row.get("expected_ebikes"),
            f"expected_total_bikes{suffix}": row.get("expected_total_bikes"),
            f"p_count_ebikes{suffix}": _json_dict(row.get("p_count_ebikes_json")),
            f"p_count_total{suffix}": _json_dict(row.get("p_count_total_json")),
            f"p_capacity_violation{suffix}": row.get("p_capacity_violation"),
            f"p_dock_constrained_arrival{suffix}": row.get("p_dock_constrained_arrival"),
            f"model_version{suffix}": row.get("model_version"),
            f"model_label{suffix}": predictor.MODEL_SPECS.get(model_key, {}).get("label", model_key),
            f"sample_size{suffix}": None,
        }
        for col, value in values.items():
            assign_value = np.nan if value is None else value
            if col not in out.columns:
                if isinstance(assign_value, (dict, list, str)):
                    out[col] = pd.Series([None] * len(out), index=out.index, dtype="object")
                else:
                    out[col] = np.nan
            elif isinstance(assign_value, (dict, list, str)) and out[col].dtype != "object":
                out[col] = out[col].astype("object")
            out.loc[mask, col] = [assign_value] * int(mask.sum())
        lcb_col = f"reliable_probability_lcb_{model_key}"
        if lcb_col not in out.columns:
            out[lcb_col] = np.nan
        current = out.loc[mask, lcb_col]
        row_lcb = _maybe_float(row.get("reliable_probability_lcb")) or p_has
        if row_lcb is not None and p_has is not None:
            row_lcb = min(row_lcb, p_has)
        if row_lcb is not None:
            out.loc[mask, lcb_col] = current.fillna(row_lcb)
        if model_key == active_model_key:
            for base in [
                "p_has_ebike",
                "p_zero",
                "p_appears",
                "p_survives",
                "expected_ebikes",
                "expected_total_bikes",
                "p_count_ebikes",
                "p_count_total",
                "p_capacity_violation",
                "p_dock_constrained_arrival",
                "sample_size",
            ]:
                source = f"{base}{suffix}"
                target = f"{base}_{horizon}m"
                if source in out.columns:
                    out.loc[mask, target] = out.loc[mask, source]
            out.loc[mask, "model_version"] = row.get("model_version")
            out.loc[mask, "data_age_minutes"] = row.get("data_age_minutes")
    latest_report = pd.to_datetime(out["last_reported"], errors="coerce")
    if "data_age_minutes" not in out.columns:
        out["data_age_minutes"] = np.nan
    age = (pd.Timestamp(_utc_now()) - latest_report).dt.total_seconds() / 60.0
    out["data_age_minutes"] = pd.to_numeric(out["data_age_minutes"], errors="coerce").fillna(age)
    out["confidence"] = np.where(out["data_age_minutes"].fillna(999.0) <= 10.0, "high", "low")
    out = predictor.apply_arrival_time_scores(
        out,
        active_model_key=active_model_key,
        search_radius_km=search_radius_km,
    )
    return out.sort_values(
        ["walk_adjusted_score", "rank_probability", "distance_km", "num_ebikes_available"],
        ascending=[False, False, True, False],
    )


def _json_dict(value):
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return None


def cache_health(conn: duckdb.DuckDBPyConnection) -> dict:
    now = _utc_now()
    try:
        row = conn.execute(
            """
            SELECT MAX(as_of), COUNT(DISTINCT station_id), COUNT(*)
            FROM live_station_predictions
            """
        ).fetchone()
    except Exception:
        return {"status": "unknown", "latest_as_of": None, "age_minutes": None, "stations_cached": 0, "rows_cached": 0}
    latest = row[0] if row else None
    age = (now - latest).total_seconds() / 60.0 if latest else None
    return {
        "status": "fresh" if age is not None and age <= config.CACHE_MAX_AGE_MINUTES else "stale",
        "latest_as_of": latest.isoformat() if latest else None,
        "age_minutes": age,
        "stations_cached": int(row[1] or 0) if row else 0,
        "rows_cached": int(row[2] or 0) if row else 0,
    }
