from __future__ import annotations

import argparse
import json
import math
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Iterable

import duckdb
import numpy as np
import pandas as pd

from . import db, decision_metrics, predictor

LOCAL_TZ = predictor.LOCAL_TZ
HORIZONS = predictor.HORIZONS


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _as_float(value, default: float | None = None) -> float | None:
    if value is None or pd.isna(value):
        return default
    return float(value)


def _as_int(value, default: int | None = None) -> int | None:
    if value is None or pd.isna(value):
        return default
    return int(value)


def _has_value(value) -> bool:
    if value is None:
        return False
    try:
        return not bool(pd.isna(value))
    except (TypeError, ValueError):
        return True


def _json_text(value) -> str | None:
    if not _has_value(value):
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, np.generic):
        value = value.item()
    try:
        return json.dumps(value, separators=(",", ":"), sort_keys=True)
    except TypeError:
        return None


def _has_column(conn: duckdb.DuckDBPyConnection, table_name: str, column_name: str) -> bool:
    try:
        row = conn.execute(
            """
            SELECT COUNT(*)
            FROM information_schema.columns
            WHERE table_name = ? AND column_name = ?
            """,
            [table_name, column_name],
        ).fetchone()
        return bool(row and row[0])
    except Exception:
        return False


def _select_column(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    alias: str,
    column_name: str,
    fallback_sql: str,
) -> str:
    if _has_column(conn, table_name, column_name):
        return f"{alias}.{column_name} AS {column_name}"
    return f"{fallback_sql} AS {column_name}"


def _forecast_value(row: pd.Series, base: str, horizon: int, model_key: str):
    suffix = f"_{horizon}m_{model_key}"
    value = row.get(f"{base}{suffix}")
    if not _has_value(value) and model_key == row.get("active_model_key", predictor.ACTIVE_MODEL_KEY):
        value = row.get(f"{base}_{horizon}m")
    return value


def _distribution_dict(value) -> dict[str, float] | None:
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
            out[str(key)] = max(0.0, p)
    total = sum(out.values())
    if total <= 0:
        return None
    return {key: value / total for key, value in out.items()}


def _count_log_prob(distribution, observed: int | None) -> float | None:
    dist = _distribution_dict(distribution)
    if dist is None or observed is None:
        return None
    key = str(int(observed)) if int(observed) <= 4 else "5_plus"
    return float(-math.log(max(1e-9, dist.get(key, 0.0))))


def _count_crps(distribution, observed: int | None) -> float | None:
    dist = _distribution_dict(distribution)
    if dist is None or observed is None:
        return None
    obs = int(observed)
    cdf = 0.0
    total = 0.0
    for key, bucket_value in [("0", 0), ("1", 1), ("2", 2), ("3", 3), ("4", 4), ("5_plus", 5)]:
        cdf += float(dist.get(key, 0.0))
        target_cdf = 1.0 if obs <= bucket_value else 0.0
        total += (cdf - target_cdf) ** 2
    return float(total)


def _clip_prob(series: pd.Series) -> pd.Series:
    return series.astype(float).clip(0.001, 0.999)


def _log_loss(y: pd.Series, p: pd.Series) -> float:
    probs = _clip_prob(p)
    truth = y.astype(float)
    return float(-(truth * np.log(probs) + (1.0 - truth) * np.log(1.0 - probs)).mean())


def _brier(y: pd.Series, p: pd.Series) -> float:
    probs = p.astype(float)
    truth = y.astype(float)
    return float(((probs - truth) ** 2).mean())


def _rank_loss(metrics: dict) -> float | None:
    if metrics.get("brier_score") is None or metrics.get("log_loss") is None:
        return None
    return float(metrics["brier_score"] + 0.05 * metrics["log_loss"])


def query_place_key(lat: float, lon: float, precision: int = 3) -> str:
    return f"{round(float(lat), precision):.{precision}f},{round(float(lon), precision):.{precision}f}"


def init_schema(conn: duckdb.DuckDBPyConnection) -> None:
    db.init_schema(conn)


def log_prediction_query(
    conn: duckdb.DuckDBPyConnection,
    *,
    request_id: str,
    source: str,
    lat: float,
    lon: float,
    near_radius_km: float,
    search_radius_km: float,
    query_label: str | None = None,
    candidate_count: int | None = None,
    best_station_id: str | None = None,
    queried_at: datetime | str | None = None,
    place_key: str | None = None,
) -> str:
    init_schema(conn)
    place_key = place_key or query_place_key(lat, lon)
    if queried_at is None:
        queried = _utc_now()
    else:
        queried = pd.Timestamp(queried_at).to_pydatetime()
        if queried.tzinfo is not None:
            queried = queried.astimezone(timezone.utc).replace(tzinfo=None)
    conn.execute(
        """
        INSERT INTO prediction_queries (
          request_id, source, queried_at, query_label, query_place_key,
          lat, lon, near_radius_km, search_radius_km, candidate_count,
          best_station_id
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (request_id) DO UPDATE SET
          source = excluded.source,
          query_label = excluded.query_label,
          query_place_key = excluded.query_place_key,
          lat = excluded.lat,
          lon = excluded.lon,
          near_radius_km = excluded.near_radius_km,
          search_radius_km = excluded.search_radius_km,
          candidate_count = excluded.candidate_count,
          best_station_id = excluded.best_station_id
        """,
        [
            request_id,
            source,
            queried,
            query_label,
            place_key,
            lat,
            lon,
            near_radius_km,
            search_radius_km,
            candidate_count,
            best_station_id,
        ],
    )
    return place_key


def log_forecasts(
    conn: duckdb.DuckDBPyConnection,
    scored: pd.DataFrame,
    *,
    user_lat: float | None,
    user_lon: float | None,
    request_id: str | None = None,
    source: str = "api",
    recommended_station_ids: Iterable[str] | None = None,
    recommended_station_ids_by_model: dict[str, Iterable[str] | str] | None = None,
    query_place_key: str | None = None,
    query_label: str | None = None,
    horizons: tuple[int, ...] = HORIZONS,
    active_model_key: str | None = None,
    active_model_source: str | None = None,
    best_evaluated_model_key: str | None = None,
) -> int:
    """Persist forecast rows so later observations can score the model."""
    init_schema(conn)
    if scored.empty:
        return 0

    recommended = set(str(station_id) for station_id in (recommended_station_ids or []))
    by_model: dict[str, set[str]] = {}
    for key, values in (recommended_station_ids_by_model or {}).items():
        if isinstance(values, str):
            by_model[str(key)] = {values}
        else:
            by_model[str(key)] = {str(v) for v in values}
    active_model_key = active_model_key or (
        str(scored["active_model_key"].dropna().iloc[0])
        if "active_model_key" in scored and scored["active_model_key"].notna().any()
        else predictor.ACTIVE_MODEL_KEY
    )
    active_model_source = active_model_source or (
        str(scored["active_model_source"].dropna().iloc[0])
        if "active_model_source" in scored and scored["active_model_source"].notna().any()
        else None
    )
    best_evaluated_model_key = best_evaluated_model_key or (
        str(scored["best_evaluated_model_key"].dropna().iloc[0])
        if "best_evaluated_model_key" in scored and scored["best_evaluated_model_key"].notna().any()
        else None
    )
    forecasted_at = _utc_now()
    rows = []
    for _, row in scored.iterrows():
        station_id = str(row["station_id"])
        row_forecasted_at = row.get("forecasted_at")
        if row_forecasted_at is not None and not pd.isna(row_forecasted_at):
            forecasted = pd.Timestamp(row_forecasted_at).to_pydatetime().replace(tzinfo=None)
        else:
            forecasted = forecasted_at
        for horizon in horizons:
            for model_key in predictor.MODEL_KEYS:
                suffix = f"_{horizon}m_{model_key}"
                p = _as_float(_forecast_value(row, "p_has_ebike", horizon, model_key))
                if p is None:
                    continue
                p_zero = _as_float(_forecast_value(row, "p_zero", horizon, model_key))
                p_appears = _as_float(_forecast_value(row, "p_appears", horizon, model_key))
                p_survives = _as_float(_forecast_value(row, "p_survives", horizon, model_key))
                model_spec = predictor.MODEL_SPECS[model_key]
                p_arrival = _as_float(row.get(f"p_arrival_{model_key}"))
                if p_arrival is None and model_key == active_model_key:
                    p_arrival = _as_float(row.get("p_arrival"))
                lcb = _as_float(row.get(f"reliable_probability_lcb_{model_key}"))
                if lcb is None and model_key == active_model_key:
                    lcb = _as_float(row.get("reliable_probability_lcb"))
                recommended_rank = _as_int(row.get(f"recommended_rank_{model_key}"))
                model_recommended = station_id in by_model.get(model_key, set())
                active_recommended = model_key == active_model_key and station_id in recommended
                if active_recommended:
                    decision_role = "active_best_practical"
                elif model_recommended or recommended_rank == 1:
                    decision_role = "model_best_practical"
                elif station_id in recommended:
                    decision_role = "highest_probability"
                else:
                    decision_role = "candidate"
                rows.append((
                    str(uuid.uuid4()),
                    request_id,
                    source,
                    model_key,
                    row.get(f"model_label{suffix}") or model_spec["label"],
                    row.get(f"model_version{suffix}") or model_spec["version"],
                    row.get("baseline_version"),
                    query_place_key,
                    query_label,
                    station_id,
                    row.get("name"),
                    forecasted,
                    forecasted + timedelta(minutes=horizon),
                    horizon,
                    user_lat,
                    user_lon,
                    _as_float(row.get("lat")),
                    _as_float(row.get("lon")),
                    _as_float(row.get("distance_km")),
                    _as_int(row.get("num_ebikes_available"), 0),
                    float(p),
                    p_zero,
                    p_appears,
                    p_survives,
                    row.get("confidence"),
                    station_id in recommended,
                    _as_float(_forecast_value(row, "expected_ebikes", horizon, model_key)),
                    _as_float(_forecast_value(row, "expected_total_bikes", horizon, model_key)),
                    _json_text(_forecast_value(row, "p_count_ebikes", horizon, model_key)),
                    _json_text(_forecast_value(row, "p_count_total", horizon, model_key)),
                    _as_float(_forecast_value(row, "p_capacity_violation", horizon, model_key)),
                    _as_float(_forecast_value(row, "p_dock_constrained_arrival", horizon, model_key)),
                    _as_float(_forecast_value(row, "expected_ebike_departures", horizon, model_key)),
                    _as_float(_forecast_value(row, "expected_classic_departures", horizon, model_key)),
                    _as_float(_forecast_value(row, "expected_ebike_arrivals", horizon, model_key)),
                    _as_float(_forecast_value(row, "expected_classic_arrivals", horizon, model_key)),
                    _as_float(
                        row.get(f"walk_adjusted_score_{model_key}")
                        if _has_value(row.get(f"walk_adjusted_score_{model_key}"))
                        else row.get("walk_adjusted_score")
                    ),
                    _as_float(row.get("arrival_time_minutes")),
                    p_arrival,
                    lcb,
                    row.get(f"feature_snapshot_id_{model_key}") or row.get("feature_snapshot_id"),
                    decision_role,
                    recommended_rank,
                    active_model_key,
                    active_model_source,
                    best_evaluated_model_key,
                ))

    if not rows:
        return 0
    conn.executemany(
        """
        INSERT OR IGNORE INTO model_forecasts (
          forecast_id, request_id, source, model_key, model_label,
          model_version, baseline_version, query_place_key, query_label,
          station_id, station_name, forecasted_at, target_at, horizon_minutes,
          user_lat, user_lon, station_lat, station_lon, distance_km,
          current_ebikes, p_has_ebike, p_zero, p_appears, p_survives,
          confidence, is_recommended, expected_ebikes, expected_total_bikes,
          p_count_ebikes_json, p_count_total_json, p_capacity_violation,
          p_dock_constrained_arrival, expected_ebike_departures,
          expected_classic_departures, expected_ebike_arrivals,
          expected_classic_arrivals, walk_adjusted_score, arrival_time_minutes,
          p_arrival, reliable_probability_lcb, feature_snapshot_id, decision_role,
          recommended_rank, active_model_key, active_model_source, best_evaluated_model_key
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    return len(rows)


def resolve_due_outcomes(
    conn: duckdb.DuckDBPyConnection,
    *,
    tolerance_minutes: int = 3,
    limit: int = 5000,
    max_age_hours: int = 6,
) -> int:
    """Attach the first observed station state at or after each forecast target.

    Forecasts whose ``target_at`` is older than ``max_age_hours`` are skipped:
    once the station-status tolerance window has closed, no later poll can
    satisfy them, so retrying every tick just blocks the resolver from
    reaching newer forecasts that *are* resolvable.
    """
    init_schema(conn)
    now = _utc_now()
    due = conn.execute(
        """
        SELECT
          f.forecast_id,
          f.request_id,
          COALESCE(f.model_key, 'logistic') AS model_key,
          f.station_id,
          f.horizon_minutes,
          f.target_at,
          f.p_count_ebikes_json,
          f.decision_role,
          f.recommended_rank,
          f.distance_km,
          f.arrival_time_minutes,
          f.p_arrival,
          f.reliable_probability_lcb,
          f.walk_adjusted_score
        FROM model_forecasts f
        LEFT JOIN model_outcomes o USING (forecast_id)
        WHERE o.forecast_id IS NULL
          AND f.target_at <= ?
          AND f.target_at >= ?
        ORDER BY f.target_at DESC
        LIMIT ?
        """,
        [now, now - timedelta(hours=max_age_hours), limit],
    ).fetchall()
    if not due:
        return 0

    rows = []
    recommendation_rows = []
    resolved_at = _utc_now()
    for (
        forecast_id,
        request_id,
        model_key,
        station_id,
        horizon,
        target_at,
        p_count_ebikes,
        decision_role,
        recommended_rank,
        distance_km,
        arrival_time_minutes,
        p_arrival,
        reliable_probability_lcb,
        walk_adjusted_score,
    ) in due:
        latest_ok = target_at + timedelta(minutes=tolerance_minutes)
        observed = conn.execute(
            """
            SELECT last_reported, num_ebikes_available, num_bikes_available, num_docks_available
            FROM station_status
            WHERE station_id = ?
              AND last_reported >= ?
              AND last_reported <= ?
            ORDER BY last_reported
            LIMIT 1
            """,
            [station_id, target_at, latest_ok],
        ).fetchone()
        if observed is None:
            continue
        observed_at, observed_ebikes, observed_total_bikes, observed_docks = observed
        observed_ebikes_int = _as_int(observed_ebikes, 0)
        observed_hit = bool((observed_ebikes or 0) >= 1)
        status_age = (observed_at - target_at).total_seconds() / 60.0
        rows.append((
            forecast_id,
            station_id,
            int(horizon),
            target_at,
            observed_at,
            observed_ebikes_int,
            observed_hit,
            resolved_at,
            _as_int(observed_total_bikes),
            _as_int(observed_docks),
            status_age,
            _count_log_prob(p_count_ebikes, observed_ebikes_int),
            _count_crps(p_count_ebikes, observed_ebikes_int),
        ))
        if request_id and decision_role and decision_role != "candidate":
            realized = (1.0 if observed_hit else 0.0) - 0.05 * float(distance_km or 0.0)
            oracle = 1.0
            regret = max(0.0, oracle - realized)
            recommendation_rows.append((
                request_id,
                model_key,
                station_id,
                decision_role,
                _as_int(recommended_rank),
                _as_float(distance_km),
                _as_int(arrival_time_minutes),
                _as_float(p_arrival),
                _as_float(reliable_probability_lcb),
                _as_float(walk_adjusted_score),
                target_at,
                observed_at,
                observed_ebikes_int,
                observed_hit,
                realized,
                oracle,
                regret,
                resolved_at,
            ))

    if not rows:
        return 0
    conn.executemany(
        """
        INSERT OR IGNORE INTO model_outcomes (
          forecast_id, station_id, horizon_minutes, target_at, observed_at,
          observed_ebikes, observed_has_ebike, resolved_at, observed_total_bikes,
          observed_docks, status_age_minutes, count_log_prob, crps
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    if recommendation_rows:
        conn.executemany(
            """
            INSERT OR IGNORE INTO recommendation_outcomes (
              request_id, model_key, station_id, decision_role, recommended_rank,
              distance_km, arrival_time_minutes, p_arrival, reliable_probability_lcb,
              walk_adjusted_score, target_at, observed_at, observed_ebikes,
              observed_has_ebike, realized_utility, oracle_utility,
              distance_adjusted_regret, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            recommendation_rows,
        )
    return len(rows)


def emit_self_evaluation_forecasts(
    conn: duckdb.DuckDBPyConnection,
    *,
    station_sample_size: int,
    tick_index: int,
    horizons: tuple[int, ...] = HORIZONS,
) -> int:
    """Score a rotating shard of stations and log the forecasts.

    Keeps the prediction → outcome → metric → selection loop fed when no
    API traffic is arriving. Tagged ``source='self_eval'`` with NULL
    ``request_id`` so the existing guard at the top of the
    ``recommendation_outcomes`` insert in :func:`resolve_due_outcomes`
    leaves them out of distance-adjusted-regret accounting; they still
    flow into ``model_outcomes`` and ``model_metrics`` unchanged.
    """
    init_schema(conn)
    if station_sample_size <= 0:
        return 0
    total_row = conn.execute(
        "SELECT COUNT(*) FROM stations WHERE lat IS NOT NULL AND lon IS NOT NULL"
    ).fetchone()
    total = int(total_row[0] if total_row else 0)
    if total <= 0:
        return 0
    offset = (int(tick_index) * int(station_sample_size)) % max(total, 1)
    candidates = conn.execute(
        """
        WITH latest AS (
          SELECT station_id, num_bikes_available, num_ebikes_available,
                 num_docks_available, last_reported, is_renting
          FROM (
            SELECT *, ROW_NUMBER() OVER (
              PARTITION BY station_id ORDER BY last_reported DESC
            ) AS rn
            FROM station_status
            WHERE last_reported >= now() - INTERVAL '30 minutes'
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
          CAST(0.0 AS DOUBLE) AS distance_km
        FROM stations s
        JOIN latest l USING (station_id)
        WHERE s.lat IS NOT NULL AND s.lon IS NOT NULL
        ORDER BY s.station_id
        LIMIT ? OFFSET ?
        """,
        [int(station_sample_size), offset],
    ).df()
    if candidates.empty:
        return 0
    scored, _suite = predictor.score_candidates(conn, candidates, horizons=horizons)
    if scored.empty:
        return 0
    return log_forecasts(
        conn,
        scored,
        user_lat=None,
        user_lon=None,
        request_id=None,
        source="self_eval",
        recommended_station_ids=[],
        recommended_station_ids_by_model={},
        query_place_key=None,
        query_label="self_eval",
        horizons=horizons,
    )


def _joined_forecasts(conn: duckdb.DuckDBPyConnection, window_hours: int) -> pd.DataFrame:
    arrival_expr = _select_column(conn, "model_forecasts", "f", "arrival_time_minutes", "CAST(NULL AS DOUBLE)")
    p_arrival_expr = _select_column(conn, "model_forecasts", "f", "p_arrival", "f.p_has_ebike")
    lcb_expr = _select_column(conn, "model_forecasts", "f", "reliable_probability_lcb", "f.p_has_ebike")
    decision_role_expr = _select_column(
        conn,
        "model_forecasts",
        "f",
        "decision_role",
        "CASE WHEN COALESCE(f.is_recommended, false) THEN 'active_best_practical' ELSE 'candidate' END",
    )
    recommended_rank_expr = _select_column(
        conn,
        "model_forecasts",
        "f",
        "recommended_rank",
        "CASE WHEN COALESCE(f.is_recommended, false) THEN 1 ELSE NULL END",
    )
    active_model_expr = _select_column(conn, "model_forecasts", "f", "active_model_key", "CAST(NULL AS TEXT)")
    active_source_expr = _select_column(conn, "model_forecasts", "f", "active_model_source", "CAST(NULL AS TEXT)")
    best_model_expr = _select_column(conn, "model_forecasts", "f", "best_evaluated_model_key", "CAST(NULL AS TEXT)")
    expected_ebikes_expr = _select_column(conn, "model_forecasts", "f", "expected_ebikes", "CAST(NULL AS DOUBLE)")
    expected_total_expr = _select_column(conn, "model_forecasts", "f", "expected_total_bikes", "CAST(NULL AS DOUBLE)")
    p_count_expr = _select_column(conn, "model_forecasts", "f", "p_count_ebikes_json", "CAST(NULL AS JSON)")
    capacity_violation_expr = _select_column(conn, "model_forecasts", "f", "p_capacity_violation", "CAST(NULL AS DOUBLE)")
    dock_constrained_expr = _select_column(conn, "model_forecasts", "f", "p_dock_constrained_arrival", "CAST(NULL AS DOUBLE)")
    observed_total_expr = _select_column(conn, "model_outcomes", "o", "observed_total_bikes", "CAST(NULL AS INTEGER)")
    observed_docks_expr = _select_column(conn, "model_outcomes", "o", "observed_docks", "CAST(NULL AS INTEGER)")
    count_log_expr = _select_column(conn, "model_outcomes", "o", "count_log_prob", "CAST(NULL AS DOUBLE)")
    crps_expr = _select_column(conn, "model_outcomes", "o", "crps", "CAST(NULL AS DOUBLE)")
    return conn.execute(
        f"""
        SELECT
          f.forecast_id,
          COALESCE(f.model_key, 'logistic') AS model_key,
          COALESCE(f.model_label, f.model_version) AS model_label,
          f.model_version,
          f.source,
          f.query_place_key,
          f.query_label,
          f.station_id,
          f.station_name,
          f.forecasted_at,
          f.target_at,
          f.horizon_minutes,
          f.distance_km,
          {arrival_expr},
          f.current_ebikes,
          f.p_has_ebike,
          {p_arrival_expr},
          {lcb_expr},
          {expected_ebikes_expr},
          {expected_total_expr},
          {p_count_expr},
          {capacity_violation_expr},
          {dock_constrained_expr},
          f.confidence,
          f.is_recommended,
          {decision_role_expr},
          {recommended_rank_expr},
          {active_model_expr},
          {active_source_expr},
          {best_model_expr},
          o.observed_at,
          o.observed_ebikes,
          {observed_total_expr},
          {observed_docks_expr},
          o.observed_has_ebike,
          {count_log_expr},
          {crps_expr}
        FROM model_forecasts f
        JOIN model_outcomes o USING (forecast_id)
        WHERE f.forecasted_at >= ? - (? * INTERVAL '1 hour')
        """,
        [_utc_now(), window_hours],
    ).df()


def _metric_row(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "n": 0,
            "brier_score": None,
            "log_loss": None,
            "rank_loss": None,
            "observed_rate": None,
            "mean_prediction": None,
            "count_log_loss": None,
            "crps": None,
            "ece": None,
            "recommended_hit_rate": None,
            "distance_adjusted_regret": None,
            "decision_rank_loss": None,
            "mean_expected_ebikes": None,
            "mean_observed_ebikes": None,
            "capacity_violation_rate": None,
            "dock_constrained_arrival_rate": None,
        }
    y = df["observed_has_ebike"].astype(float)
    p = df["p_has_ebike"].astype(float)
    row = {
        "n": int(len(df)),
        "brier_score": _brier(y, p),
        "log_loss": _log_loss(y, p),
        "observed_rate": float(y.mean()),
        "mean_prediction": float(p.mean()),
    }
    row["ece"] = decision_metrics.ece_score(y, p)
    if "count_log_prob" in df and df["count_log_prob"].notna().any():
        row["count_log_loss"] = float(pd.to_numeric(df["count_log_prob"], errors="coerce").mean())
    else:
        row["count_log_loss"] = None
    if "crps" in df and df["crps"].notna().any():
        row["crps"] = float(pd.to_numeric(df["crps"], errors="coerce").mean())
    else:
        row["crps"] = None
    if "expected_ebikes" in df and df["expected_ebikes"].notna().any():
        row["mean_expected_ebikes"] = float(pd.to_numeric(df["expected_ebikes"], errors="coerce").mean())
    else:
        row["mean_expected_ebikes"] = None
    if "observed_ebikes" in df and df["observed_ebikes"].notna().any():
        row["mean_observed_ebikes"] = float(pd.to_numeric(df["observed_ebikes"], errors="coerce").mean())
    else:
        row["mean_observed_ebikes"] = None
    if "p_capacity_violation" in df and df["p_capacity_violation"].notna().any():
        row["capacity_violation_rate"] = float(pd.to_numeric(df["p_capacity_violation"], errors="coerce").mean())
    else:
        row["capacity_violation_rate"] = None
    if "p_dock_constrained_arrival" in df and df["p_dock_constrained_arrival"].notna().any():
        row["dock_constrained_arrival_rate"] = float(
            pd.to_numeric(df["p_dock_constrained_arrival"], errors="coerce").mean()
        )
    else:
        row["dock_constrained_arrival_rate"] = None
    row["rank_loss"] = _rank_loss(row)
    recommended = decision_metrics.recommended_precision(df)
    row["recommended_hit_rate"] = recommended["hit_rate"] if recommended else None
    if "distance_adjusted_regret" in df and df["distance_adjusted_regret"].notna().any():
        row["distance_adjusted_regret"] = float(pd.to_numeric(df["distance_adjusted_regret"], errors="coerce").mean())
    else:
        row["distance_adjusted_regret"] = None
    row["decision_rank_loss"] = decision_metrics.decision_rank_loss(row)
    return row


def performance_summary(
    conn: duckdb.DuckDBPyConnection,
    *,
    window_hours: int = 24,
    resolve: bool = True,
    initialize_schema: bool = True,
) -> dict:
    if initialize_schema:
        init_schema(conn)
    resolved_now = resolve_due_outcomes(conn) if resolve else 0
    joined = _joined_forecasts(conn, window_hours)
    if joined.empty:
        return {
            "window_hours": window_hours,
            "resolved_now": resolved_now,
            "overall": _metric_row(joined),
            "by_horizon": [],
            "by_model": [],
            "by_model_horizon": [],
            "by_horizon_model": [],
            "model_leaderboard": [],
            "by_place_model": [],
            "calibration": [],
            "coherence": None,
            "decision_metrics": {},
            "recommended_precision": None,
            "active_model": None,
            "best_current_model": None,
            "best_sota_model": None,
            "active_equals_best": None,
            "worst_station_hours": [],
        }

    joined["observed_has_ebike"] = joined["observed_has_ebike"].astype(bool)
    joined["p_has_ebike"] = joined["p_has_ebike"].astype(float).clip(0.001, 0.999)

    by_horizon = []
    for horizon, group in joined.groupby("horizon_minutes"):
        row = _metric_row(group)
        row["horizon_minutes"] = int(horizon)
        by_horizon.append(row)

    by_model = []
    for model_key, group in joined.groupby("model_key"):
        row = _metric_row(group)
        row["model_key"] = model_key
        row["model_label"] = group["model_label"].dropna().iloc[0] if group["model_label"].notna().any() else model_key
        by_model.append(row)
    by_model = sorted(
        by_model,
        key=lambda r: math.inf if r.get("decision_rank_loss") is None else r["decision_rank_loss"],
    )
    for rank, row in enumerate(by_model, start=1):
        row["rank"] = rank

    by_model_horizon = []
    for (model_key, horizon), group in joined.groupby(["model_key", "horizon_minutes"]):
        row = _metric_row(group)
        row["model_key"] = model_key
        row["model_label"] = group["model_label"].dropna().iloc[0] if group["model_label"].notna().any() else model_key
        row["horizon_minutes"] = int(horizon)
        by_model_horizon.append(row)
    by_model_horizon = sorted(
        by_model_horizon,
        key=lambda r: (r["horizon_minutes"], math.inf if r.get("decision_rank_loss") is None else r["decision_rank_loss"]),
    )

    by_place_model = []
    place_groups = joined.dropna(subset=["query_place_key"]).groupby(["query_place_key", "model_key"])
    for (place_key, model_key), group in place_groups:
        if len(group) < 2:
            continue
        row = _metric_row(group)
        row["query_place_key"] = place_key
        label_values = group["query_label"].dropna()
        row["query_label"] = label_values.iloc[-1] if not label_values.empty else place_key
        row["model_key"] = model_key
        row["model_label"] = group["model_label"].dropna().iloc[0] if group["model_label"].notna().any() else model_key
        by_place_model.append(row)
    by_place_model = sorted(
        by_place_model,
        key=lambda r: (r["query_place_key"], math.inf if r.get("rank_loss") is None else r["rank_loss"]),
    )

    bands = pd.IntervalIndex.from_breaks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], closed="right")
    calibration = []
    joined["probability_band"] = pd.cut(joined["p_has_ebike"], bands)
    for band, group in joined.groupby("probability_band", observed=True):
        calibration.append({
            "probability_band": f"{band.left:.1f}-{band.right:.1f}",
            "n": int(len(group)),
            "mean_prediction": float(group["p_has_ebike"].mean()),
            "observed_hit_rate": float(group["observed_has_ebike"].astype(float).mean()),
        })

    recommended = joined[joined["is_recommended"] == True]  # noqa: E712
    recommended_precision = None
    if not recommended.empty:
        recommended_precision = {
            "n": int(len(recommended)),
            "hit_rate": float(recommended["observed_has_ebike"].astype(float).mean()),
            "mean_prediction": float(recommended["p_has_ebike"].mean()),
        }

    ts = pd.to_datetime(joined["forecasted_at"], utc=True)
    joined["local_hour"] = ts.dt.tz_convert(LOCAL_TZ).dt.hour
    worst = []
    grouped = joined.groupby(["model_key", "station_id", "station_name", "horizon_minutes", "local_hour"], dropna=False)
    for (model_key, station_id, station_name, horizon, hour), group in grouped:
        if len(group) < 3:
            continue
        row = _metric_row(group)
        row.update({
            "model_key": model_key,
            "station_id": station_id,
            "station_name": station_name,
            "horizon_minutes": int(horizon),
            "local_hour": int(hour),
        })
        worst.append(row)
    worst = sorted(
        worst,
        key=lambda r: (math.inf if r["brier_score"] is None else r["brier_score"]),
        reverse=True,
    )[:10]

    overall = _metric_row(joined)
    coherence = {
        "capacity_violation_rate": overall.get("capacity_violation_rate"),
        "dock_constrained_arrival_rate": overall.get("dock_constrained_arrival_rate"),
        "count_log_loss": overall.get("count_log_loss"),
        "crps": overall.get("crps"),
        "mean_expected_ebikes": overall.get("mean_expected_ebikes"),
        "mean_observed_ebikes": overall.get("mean_observed_ebikes"),
    }
    best_current_model = by_model[0] if by_model else None
    best_sota_model = next((row for row in by_model if row.get("model_key") in predictor.SOTA_PRIMARY_MODEL_KEYS), None)
    active_key = None
    if "active_model_key" in joined and joined["active_model_key"].notna().any():
        active_key = str(joined["active_model_key"].dropna().iloc[-1])
    else:
        active_key = predictor.ACTIVE_MODEL_KEY
    active_model = next((row for row in by_model if row.get("model_key") == active_key), None)
    active_payload = None
    if active_model:
        active_payload = dict(active_model)
        active_payload["active_model_key"] = active_key
        active_payload["active_model_source"] = (
            str(joined["active_model_source"].dropna().iloc[-1])
            if "active_model_source" in joined and joined["active_model_source"].notna().any()
            else None
        )
    decision_summary = {
        "recommended_precision": recommended_precision,
        "active_recommendation_hit_rate": (
            float(joined[joined["decision_role"] == "active_best_practical"]["observed_has_ebike"].astype(float).mean())
            if "decision_role" in joined and not joined[joined["decision_role"] == "active_best_practical"].empty
            else None
        ),
        "model_specific_top_hit_rate": (
            float(joined[joined["decision_role"] == "model_best_practical"]["observed_has_ebike"].astype(float).mean())
            if "decision_role" in joined and not joined[joined["decision_role"] == "model_best_practical"].empty
            else None
        ),
    }

    return {
        "window_hours": window_hours,
        "resolved_now": resolved_now,
        "overall": overall,
        "by_horizon": by_horizon,
        "by_model": by_model,
        "by_model_horizon": by_model_horizon,
        "by_horizon_model": by_model_horizon,
        "model_leaderboard": by_model,
        "by_place_model": by_place_model,
        "calibration": calibration,
        "coherence": coherence,
        "decision_metrics": decision_summary,
        "recommended_precision": recommended_precision,
        "active_model": active_payload,
        "best_current_model": best_current_model,
        "best_sota_model": best_sota_model,
        "active_equals_best": bool(active_key == (best_current_model or {}).get("model_key")) if best_current_model else None,
        "worst_station_hours": worst,
    }


def multi_bike_performance_summary(
    conn: duckdb.DuckDBPyConnection,
    *,
    window_hours: int = 24,
    plan_sizes: tuple[int, ...] = (2, 3, 4, 5),
    sources: tuple[str, ...] = ("api_multi_bike",),
    initialize_schema: bool = True,
) -> dict:
    """Per-plan-size leaderboard for multi-bike-plan requests.

    For each multi-bike request we count the recommended stops to derive the
    plan size, then aggregate per-(plan_size, model) metrics over **all**
    candidate forecasts that landed in those requests (not just the picked
    stops — that pool is too small for stable Brier estimates).

    Returns a dict keyed by plan size:
        {
          "window_hours": int,
          "by_plan_size": {
              k: {
                "plan_size": k,
                "n_requests": int,
                "model_leaderboard": [_metric_row + {model_key, model_label, rank}, ...],
              }
              for k in plan_sizes
          },
        }
    Plan sizes with no requests in the window are still returned with empty
    leaderboards so the UI can show a "no data yet" tile.
    """
    if initialize_schema:
        init_schema(conn)
    placeholders = ",".join("?" for _ in sources)
    requested = tuple(int(k) for k in plan_sizes)

    requests = conn.execute(
        f"""
        SELECT
          request_id,
          COUNT(DISTINCT CASE WHEN is_recommended THEN station_id END) AS plan_size
        FROM model_forecasts
        WHERE source IN ({placeholders})
          AND forecasted_at > now() - (? * INTERVAL '1 hour')
        GROUP BY request_id
        """,
        list(sources) + [window_hours],
    ).df()
    if not requests.empty:
        requests["plan_size"] = pd.to_numeric(requests["plan_size"], errors="coerce").fillna(0).astype(int)

    by_plan_size: dict[int, dict] = {}
    if requests.empty:
        for k in requested:
            by_plan_size[int(k)] = {"plan_size": int(k), "n_requests": 0, "model_leaderboard": []}
        return {"window_hours": window_hours, "by_plan_size": by_plan_size}

    request_ids_in_scope = requests.loc[requests["plan_size"].isin(requested), "request_id"].tolist()
    if not request_ids_in_scope:
        for k in requested:
            n_req = int((requests["plan_size"] == k).sum())
            by_plan_size[int(k)] = {"plan_size": int(k), "n_requests": n_req, "model_leaderboard": []}
        return {"window_hours": window_hours, "by_plan_size": by_plan_size}

    placeholders_ids = ",".join("?" for _ in request_ids_in_scope)
    joined = conn.execute(
        f"""
        SELECT
          f.request_id,
          f.model_key,
          COALESCE(f.model_label, f.model_version) AS model_label,
          f.p_has_ebike,
          o.observed_has_ebike,
          o.observed_ebikes,
          o.count_log_prob,
          o.crps,
          f.expected_ebikes,
          f.p_capacity_violation,
          f.p_dock_constrained_arrival,
          f.is_recommended
        FROM model_forecasts f
        JOIN model_outcomes o USING (forecast_id)
        WHERE f.request_id IN ({placeholders_ids})
        """,
        request_ids_in_scope,
    ).df()
    if joined.empty:
        for k in requested:
            n_req = int((requests["plan_size"] == k).sum())
            by_plan_size[int(k)] = {"plan_size": int(k), "n_requests": n_req, "model_leaderboard": []}
        return {"window_hours": window_hours, "by_plan_size": by_plan_size}

    joined = joined.merge(requests, on="request_id", how="left")
    joined["observed_has_ebike"] = joined["observed_has_ebike"].astype(bool)
    joined["p_has_ebike"] = joined["p_has_ebike"].astype(float).clip(0.001, 0.999)

    for k in requested:
        slice_df = joined[joined["plan_size"] == k]
        n_req = int((requests["plan_size"] == k).sum())
        if slice_df.empty:
            by_plan_size[int(k)] = {"plan_size": int(k), "n_requests": n_req, "model_leaderboard": []}
            continue
        rows = []
        for model_key, group in slice_df.groupby("model_key"):
            row = _metric_row(group)
            row["model_key"] = model_key
            row["model_label"] = (
                group["model_label"].dropna().iloc[0]
                if group["model_label"].notna().any()
                else model_key
            )
            rows.append(row)
        rows.sort(key=lambda r: math.inf if r.get("rank_loss") is None else r["rank_loss"])
        for rank, row in enumerate(rows, start=1):
            row["rank"] = rank
        by_plan_size[int(k)] = {
            "plan_size": int(k),
            "n_requests": n_req,
            "model_leaderboard": rows,
        }

    return {"window_hours": window_hours, "by_plan_size": by_plan_size}


def count_performance_summary(
    conn: duckdb.DuckDBPyConnection,
    *,
    window_hours: int = 24,
    initialize_schema: bool = True,
) -> dict:
    """Per-model leaderboard for *count* prediction (not just binary has-bike).

    Sorted by count NLL (lower = better). Models that don't emit a count PMF
    (random_forest, gradient_boosting, logistic, empirical, stg_ncde_inventory)
    have NLL/CRPS = None and are dropped from the leaderboard. MAE on
    expected_ebikes vs observed_ebikes is computed for any model that emits
    expected_ebikes — this lets us judge point-prediction quality even for
    models that don't ship a full PMF.
    """
    if initialize_schema:
        init_schema(conn)
    df = conn.execute(
        """
        SELECT
          f.model_key,
          COALESCE(f.model_label, f.model_version) AS model_label,
          f.expected_ebikes,
          o.observed_ebikes,
          o.count_log_prob,
          o.crps
        FROM model_forecasts f JOIN model_outcomes o USING (forecast_id)
        WHERE f.forecasted_at > now() - (? * INTERVAL '1 hour')
        """,
        [window_hours],
    ).df()
    if df.empty:
        return {"window_hours": window_hours, "model_leaderboard": []}

    rows = []
    for model_key, group in df.groupby("model_key"):
        with_pmf = group.dropna(subset=["count_log_prob"])
        with_expected = group.dropna(subset=["expected_ebikes", "observed_ebikes"])
        n = int(len(group))
        n_pmf = int(len(with_pmf))
        n_mae = int(len(with_expected))
        nll = float(with_pmf["count_log_prob"].mean()) if n_pmf else None
        crps = float(with_pmf["crps"].mean()) if n_pmf else None
        mae = (
            float((with_expected["expected_ebikes"] - with_expected["observed_ebikes"]).abs().mean())
            if n_mae
            else None
        )
        rmse = (
            float(((with_expected["expected_ebikes"] - with_expected["observed_ebikes"]) ** 2).mean() ** 0.5)
            if n_mae
            else None
        )
        rows.append({
            "model_key": model_key,
            "model_label": (
                group["model_label"].dropna().iloc[0]
                if group["model_label"].notna().any()
                else model_key
            ),
            "n": n,
            "n_with_pmf": n_pmf,
            "n_with_expected": n_mae,
            "count_nll": nll,
            "crps": crps,
            "mae_expected": mae,
            "rmse_expected": rmse,
            "mean_observed_ebikes": float(group["observed_ebikes"].mean()),
        })
    rows.sort(
        key=lambda r: (
            math.inf if r["count_nll"] is None else r["count_nll"],
            math.inf if r["mae_expected"] is None else r["mae_expected"],
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return {"window_hours": window_hours, "model_leaderboard": rows}


def threshold_k_performance_summary(
    conn: duckdb.DuckDBPyConnection,
    *,
    window_hours: int = 24,
    k_values: tuple[int, ...] = (1, 2, 3, 5),
    initialize_schema: bool = True,
) -> dict:
    """For each k, evaluate P(observed_ebikes >= k) Brier per model.

    The threshold probability is derived from p_count_ebikes_json
    (PMF bins "0".."4" plus "5_plus"). Only models that emit a PMF appear in
    the leaderboard. Returned shape mirrors per_horizon_performance_summary:
        {
          "window_hours": int,
          "k_values": [int, ...],
          "by_k": {k: {"k": k, "n_total": int, "model_leaderboard": [...]}},
        }
    """
    if initialize_schema:
        init_schema(conn)
    df = conn.execute(
        """
        SELECT
          f.model_key,
          COALESCE(f.model_label, f.model_version) AS model_label,
          f.p_count_ebikes_json,
          o.observed_ebikes
        FROM model_forecasts f JOIN model_outcomes o USING (forecast_id)
        WHERE f.forecasted_at > now() - (? * INTERVAL '1 hour')
          AND f.p_count_ebikes_json IS NOT NULL
        """,
        [window_hours],
    ).df()
    if df.empty:
        return {"window_hours": window_hours, "k_values": list(k_values), "by_k": {}}

    parsed = []
    for record in df.to_dict(orient="records"):
        try:
            pmf = json.loads(record["p_count_ebikes_json"])
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        # Bins are "0".."4" and "5_plus". Sum >= k.
        bins: dict[int, float] = {}
        plus_bin = 0.0
        for key, val in pmf.items():
            if key == "5_plus":
                plus_bin = float(val)
                continue
            try:
                bins[int(key)] = float(val)
            except (TypeError, ValueError):
                continue
        for k in k_values:
            if k <= 4:
                p_ge_k = sum(p for j, p in bins.items() if j >= k) + plus_bin
            else:
                # k=5: only the 5_plus bin contributes.
                p_ge_k = plus_bin if k == 5 else 0.0
            parsed.append({
                "model_key": record["model_key"],
                "model_label": record["model_label"],
                "k": k,
                "p_ge_k": float(min(max(p_ge_k, 0.0), 1.0)),
                "y_ge_k": int(int(record["observed_ebikes"] or 0) >= k),
            })
    if not parsed:
        return {"window_hours": window_hours, "k_values": list(k_values), "by_k": {}}

    parsed_df = pd.DataFrame(parsed)
    by_k: dict[int, dict] = {}
    for k in k_values:
        slice_df = parsed_df[parsed_df["k"] == k]
        if slice_df.empty:
            by_k[int(k)] = {"k": int(k), "n_total": 0, "model_leaderboard": []}
            continue
        rows = []
        for model_key, group in slice_df.groupby("model_key"):
            n = int(len(group))
            y = group["y_ge_k"].astype(float)
            p = group["p_ge_k"].astype(float).clip(0.001, 0.999)
            rows.append({
                "model_key": model_key,
                "model_label": (
                    group["model_label"].dropna().iloc[0]
                    if group["model_label"].notna().any()
                    else model_key
                ),
                "n": n,
                "brier_score": float(((y - p) ** 2).mean()),
                "log_loss": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
                "observed_rate": float(y.mean()),
                "mean_prediction": float(p.mean()),
            })
        rows.sort(key=lambda r: r["brier_score"])
        for rank, row in enumerate(rows, start=1):
            row["rank"] = rank
        by_k[int(k)] = {
            "k": int(k),
            "n_total": int(len(slice_df)),
            "model_leaderboard": rows,
        }
    return {"window_hours": window_hours, "k_values": list(k_values), "by_k": by_k}


def open_dock_performance_summary(
    conn: duckdb.DuckDBPyConnection,
    *,
    window_hours: int = 24,
    initialize_schema: bool = True,
) -> dict:
    """Per-model leaderboard for the *parking-side* prediction.

    Uses (1 - p_capacity_violation) as the model's predicted P(open dock at t)
    and (observed_docks > 0) as ground truth. Only models that emit
    p_capacity_violation contribute (currently the SOTA + inventory-world
    family).

    This is the dual of the binary "has eBike" benchmark and exposes a half
    of the user journey (returning a bike) that the existing tabs miss.
    The (1 - p_capacity_violation) framing is a proxy for the eventual
    first-class p_has_open_dock target flagged in the cdg_nmip TODO.
    """
    if initialize_schema:
        init_schema(conn)
    df = conn.execute(
        """
        SELECT
          f.model_key,
          COALESCE(f.model_label, f.model_version) AS model_label,
          f.p_capacity_violation,
          o.observed_docks
        FROM model_forecasts f JOIN model_outcomes o USING (forecast_id)
        WHERE f.forecasted_at > now() - (? * INTERVAL '1 hour')
          AND f.p_capacity_violation IS NOT NULL
          AND o.observed_docks IS NOT NULL
        """,
        [window_hours],
    ).df()
    if df.empty:
        return {"window_hours": window_hours, "n_total": 0, "open_rate": None, "model_leaderboard": []}
    df["p_open"] = (1.0 - df["p_capacity_violation"].astype(float)).clip(0.001, 0.999)
    df["y_open"] = (df["observed_docks"].fillna(0).astype(int) > 0).astype(int)
    open_rate = float(df["y_open"].mean())

    rows = []
    for model_key, group in df.groupby("model_key"):
        n = int(len(group))
        y = group["y_open"].astype(float)
        p = group["p_open"].astype(float)
        rows.append({
            "model_key": model_key,
            "model_label": (
                group["model_label"].dropna().iloc[0]
                if group["model_label"].notna().any()
                else model_key
            ),
            "n": n,
            "brier_score": float(((y - p) ** 2).mean()),
            "log_loss": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
            "observed_rate": float(y.mean()),
            "mean_prediction": float(p.mean()),
        })
    rows.sort(key=lambda r: r["brier_score"])
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return {
        "window_hours": window_hours,
        "n_total": int(len(df)),
        "open_rate": open_rate,
        "model_leaderboard": rows,
    }


def topk_recommendation_summary(
    conn: duckdb.DuckDBPyConnection,
    *,
    window_hours: int = 24,
    k_values: tuple[int, ...] = (1, 3, 5),
    initialize_schema: bool = True,
) -> dict:
    """For each multi-bike-plan request, did the model's top-k recommended
    stations include the actually-best station (most observed_ebikes)?

    The "best" station per request is determined among the candidates that
    have a resolved outcome. A model gets credit for top-k if any of its
    top-k recommended_rank stations is the argmax-observed station.

    Limited to source LIKE 'api%' (real recommendation requests) so we don't
    score self_eval forecasts where there's no decision being made.
    """
    if initialize_schema:
        init_schema(conn)
    df = conn.execute(
        """
        SELECT
          f.request_id,
          f.model_key,
          COALESCE(f.model_label, f.model_version) AS model_label,
          f.station_id,
          f.recommended_rank,
          o.observed_ebikes,
          o.observed_has_ebike,
          f.horizon_minutes
        FROM model_forecasts f JOIN model_outcomes o USING (forecast_id)
        WHERE f.forecasted_at > now() - (? * INTERVAL '1 hour')
          AND f.source LIKE 'api%'
          AND f.request_id IS NOT NULL
          AND f.horizon_minutes = 10
        """,
        [window_hours],
    ).df()
    if df.empty:
        return {"window_hours": window_hours, "k_values": list(k_values), "model_leaderboard": []}

    # For each request, find the actually-best station(s) (max observed_ebikes).
    best_per_request: dict[str, set[str]] = {}
    for request_id, group in df.groupby("request_id"):
        observed = group.drop_duplicates(subset=["station_id"])[["station_id", "observed_ebikes"]]
        if observed.empty:
            continue
        max_obs = observed["observed_ebikes"].max()
        if pd.isna(max_obs):
            continue
        best_per_request[request_id] = set(
            str(s) for s in observed.loc[observed["observed_ebikes"] == max_obs, "station_id"]
        )
    if not best_per_request:
        return {"window_hours": window_hours, "k_values": list(k_values), "model_leaderboard": []}

    rows = []
    for model_key, group in df.groupby("model_key"):
        per_k_hits = {k: 0 for k in k_values}
        n_requests = 0
        label = (
            group["model_label"].dropna().iloc[0]
            if group["model_label"].notna().any()
            else model_key
        )
        for request_id, request_group in group.groupby("request_id"):
            best = best_per_request.get(request_id)
            if not best:
                continue
            ranked = request_group.dropna(subset=["recommended_rank"]).sort_values("recommended_rank")
            if ranked.empty:
                continue
            n_requests += 1
            for k in k_values:
                topk_stations = set(str(s) for s in ranked.head(k)["station_id"])
                if topk_stations & best:
                    per_k_hits[k] += 1
        if n_requests == 0:
            continue
        row = {
            "model_key": model_key,
            "model_label": label,
            "n_requests": n_requests,
        }
        for k in k_values:
            row[f"top{k}_hit_rate"] = per_k_hits[k] / n_requests
        rows.append(row)
    rows.sort(
        key=lambda r: -r.get(f"top{k_values[0]}_hit_rate", 0.0)
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return {
        "window_hours": window_hours,
        "k_values": list(k_values),
        "model_leaderboard": rows,
    }


def survival_calibration_summary(
    conn: duckdb.DuckDBPyConnection,
    *,
    window_hours: int = 24,
    initialize_schema: bool = True,
) -> dict:
    """Empty-station survival-curve calibration per (horizon, model).

    For forecasts where current_ebikes == 0, treat (1 - p_has_ebike) as the
    survival probability "still empty at horizon t". Aggregate per
    (horizon, model) so the dashboard can plot predicted-vs-observed
    survival curves and read off where each model's arrival modeling
    diverges from reality.
    """
    if initialize_schema:
        init_schema(conn)
    df = conn.execute(
        """
        SELECT
          f.model_key,
          COALESCE(f.model_label, f.model_version) AS model_label,
          f.horizon_minutes,
          f.p_has_ebike,
          o.observed_has_ebike
        FROM model_forecasts f JOIN model_outcomes o USING (forecast_id)
        WHERE f.forecasted_at > now() - (? * INTERVAL '1 hour')
          AND f.current_ebikes = 0
        """,
        [window_hours],
    ).df()
    if df.empty:
        return {"window_hours": window_hours, "by_horizon_model": [], "horizons": []}

    df["observed_still_empty"] = (~df["observed_has_ebike"].astype(bool)).astype(float)
    df["predicted_still_empty"] = (1.0 - df["p_has_ebike"].astype(float)).clip(0.001, 0.999)
    df["brier"] = (df["observed_still_empty"] - df["predicted_still_empty"]) ** 2

    rows = []
    horizons = sorted({int(h) for h in df["horizon_minutes"].dropna().unique()})
    for (model_key, horizon), group in df.groupby(["model_key", "horizon_minutes"]):
        n = int(len(group))
        rows.append({
            "model_key": model_key,
            "model_label": (
                group["model_label"].dropna().iloc[0]
                if group["model_label"].notna().any()
                else model_key
            ),
            "horizon_minutes": int(horizon),
            "n": n,
            "predicted_still_empty": float(group["predicted_still_empty"].mean()),
            "observed_still_empty": float(group["observed_still_empty"].mean()),
            "brier": float(group["brier"].mean()),
        })
    rows.sort(key=lambda r: (r["model_key"], r["horizon_minutes"]))
    return {
        "window_hours": window_hours,
        "horizons": horizons,
        "by_horizon_model": rows,
    }


def per_horizon_performance_summary(
    conn: duckdb.DuckDBPyConnection,
    *,
    window_hours: int = 24,
    initialize_schema: bool = True,
) -> dict:
    """Per-(horizon, model) leaderboard across all resolved forecasts.

    Used by the dashboard's "By horizon" benchmark to show how each model's
    skill degrades (or doesn't) as the prediction horizon stretches. Same
    metric set as ``performance_summary`` but grouped by horizon_minutes.

    Returns:
        {
          "window_hours": int,
          "horizons": [int, ...],          # sorted, distinct
          "by_horizon": {
              horizon_minutes: {
                "horizon_minutes": int,
                "model_leaderboard": [_metric_row + {model_key, model_label, rank}, ...],
              }
          },
        }
    """
    if initialize_schema:
        init_schema(conn)
    joined = _joined_forecasts(conn, window_hours)
    if joined.empty:
        return {"window_hours": window_hours, "horizons": [], "by_horizon": {}}

    joined["observed_has_ebike"] = joined["observed_has_ebike"].astype(bool)
    joined["p_has_ebike"] = joined["p_has_ebike"].astype(float).clip(0.001, 0.999)

    by_horizon: dict[int, dict] = {}
    horizons_sorted = sorted({int(h) for h in joined["horizon_minutes"].dropna().unique()})
    for horizon in horizons_sorted:
        slice_df = joined[joined["horizon_minutes"] == horizon]
        if slice_df.empty:
            continue
        rows = []
        for model_key, group in slice_df.groupby("model_key"):
            row = _metric_row(group)
            row["model_key"] = model_key
            row["model_label"] = (
                group["model_label"].dropna().iloc[0]
                if group["model_label"].notna().any()
                else model_key
            )
            rows.append(row)
        rows.sort(key=lambda r: math.inf if r.get("rank_loss") is None else r["rank_loss"])
        for rank, row in enumerate(rows, start=1):
            row["rank"] = rank
        by_horizon[int(horizon)] = {
            "horizon_minutes": int(horizon),
            "model_leaderboard": rows,
        }

    return {
        "window_hours": window_hours,
        "horizons": horizons_sorted,
        "by_horizon": by_horizon,
    }


def empty_station_performance_summary(
    conn: duckdb.DuckDBPyConnection,
    *,
    window_hours: int = 24,
    initialize_schema: bool = True,
) -> dict:
    """Per-model leaderboard restricted to forecasts where current_ebikes == 0.

    This is a fundamentally harder problem than the full distribution: at an
    empty station, P(has eBike at horizon) reduces to "will a bike arrive?",
    which models that lean on autocorrelation can't fake. Models that
    actually represent arrival rates (empirical, ZINB-based, graph flow)
    have a chance to differentiate themselves here.

    Returns the same shape as ``performance_summary`` (single leaderboard)
    so the dashboard renderer can reuse the slim-leaderboard helper.
    """
    if initialize_schema:
        init_schema(conn)
    joined = _joined_forecasts(conn, window_hours)
    if joined.empty:
        return {
            "window_hours": window_hours,
            "n_total": 0,
            "appearance_rate": None,
            "model_leaderboard": [],
        }
    joined = joined[joined["current_ebikes"].fillna(-1).astype(int) == 0].copy()
    if joined.empty:
        return {
            "window_hours": window_hours,
            "n_total": 0,
            "appearance_rate": None,
            "model_leaderboard": [],
        }
    joined["observed_has_ebike"] = joined["observed_has_ebike"].astype(bool)
    joined["p_has_ebike"] = joined["p_has_ebike"].astype(float).clip(0.001, 0.999)
    appearance_rate = float(joined["observed_has_ebike"].astype(float).mean())

    rows = []
    for model_key, group in joined.groupby("model_key"):
        row = _metric_row(group)
        row["model_key"] = model_key
        row["model_label"] = (
            group["model_label"].dropna().iloc[0]
            if group["model_label"].notna().any()
            else model_key
        )
        rows.append(row)
    rows.sort(key=lambda r: math.inf if r.get("rank_loss") is None else r["rank_loss"])
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return {
        "window_hours": window_hours,
        "n_total": int(len(joined)),
        "appearance_rate": appearance_rate,
        "model_leaderboard": rows,
    }


def best_performing_model(
    conn: duckdb.DuckDBPyConnection,
    window_hours: int = 24,
    min_n: int = 30,
    eligible_model_keys: Iterable[str] | None = None,
    metric: str = "decision_rank_loss",
) -> dict:
    try:
        init_schema(conn)
    except Exception:
        pass
    joined = _joined_forecasts(conn, window_hours)
    eligible = set(eligible_model_keys or predictor.MODEL_KEYS)
    if joined.empty:
        return {
            "best_model_key": None,
            "best_model_label": None,
            "rank": None,
            "n": 0,
            "metric": metric,
            "metric_value": None,
            "window_hours": window_hours,
        }
    rows = []
    for model_key, group in joined.groupby("model_key"):
        if model_key not in eligible:
            continue
        row = _metric_row(group)
        if int(row.get("n") or 0) < int(min_n):
            continue
        row["model_key"] = model_key
        row["model_label"] = group["model_label"].dropna().iloc[0] if group["model_label"].notna().any() else model_key
        rows.append(row)
    if not rows and metric == "decision_rank_loss":
        return best_performing_model(
            conn,
            window_hours=window_hours,
            min_n=min_n,
            eligible_model_keys=eligible_model_keys,
            metric="rank_loss",
        )
    rows = sorted(rows, key=lambda r: math.inf if r.get(metric) is None else float(r[metric]))
    if not rows:
        return {
            "best_model_key": None,
            "best_model_label": None,
            "rank": None,
            "n": 0,
            "metric": metric,
            "metric_value": None,
            "window_hours": window_hours,
        }
    best = rows[0]
    return {
        "best_model_key": best.get("model_key"),
        "best_model_label": best.get("model_label"),
        "rank": 1,
        "n": int(best.get("n") or 0),
        "metric": metric,
        "metric_value": best.get(metric),
        "brier_score": best.get("brier_score"),
        "log_loss": best.get("log_loss"),
        "ece": best.get("ece"),
        "decision_rank_loss": best.get("decision_rank_loss"),
        "count_log_loss": best.get("count_log_loss"),
        "crps": best.get("crps"),
        "recommended_hit_rate": best.get("recommended_hit_rate"),
        "distance_adjusted_regret": best.get("distance_adjusted_regret"),
        "capacity_violation_rate": best.get("capacity_violation_rate"),
        "observed_rate": best.get("observed_rate"),
        "mean_prediction": best.get("mean_prediction"),
        "window_hours": window_hours,
    }


def snapshot_metrics(conn: duckdb.DuckDBPyConnection, *, window_hours: int = 24) -> int:
    summary = performance_summary(conn, window_hours=window_hours)
    computed_at = _utc_now()
    rows = []

    def add_row(group_key: str, group_value: str, horizon: int | None, metrics: dict) -> None:
        rows.append((
            str(uuid.uuid4()),
            predictor.MODEL_VERSION,
            computed_at,
            window_hours,
            horizon,
            group_key,
            group_value,
            int(metrics.get("n") or 0),
            metrics.get("brier_score"),
            metrics.get("log_loss"),
            metrics.get("rank_loss"),
            metrics.get("observed_rate"),
            metrics.get("mean_prediction"),
            metrics.get("count_log_loss"),
            metrics.get("crps"),
            metrics.get("ece"),
            metrics.get("recommended_hit_rate"),
            metrics.get("distance_adjusted_regret"),
            metrics.get("decision_rank_loss"),
            bool(metrics.get("model_key") == (summary.get("active_model") or {}).get("model_key")),
            bool(metrics.get("model_key") == (summary.get("best_current_model") or {}).get("model_key")),
        ))

    add_row("overall", "all", None, summary["overall"])
    for row in summary["by_horizon"]:
        add_row("horizon", str(row["horizon_minutes"]), int(row["horizon_minutes"]), row)
    for row in summary.get("by_model", []):
        add_row("model", str(row["model_key"]), None, row)
    for row in summary.get("by_place_model", []):
        group_value = f"{row['query_place_key']}::{row['model_key']}"
        add_row("place_model", group_value, None, row)

    if not rows:
        return 0
    conn.executemany(
        """
        INSERT INTO model_metrics (
          metric_id, model_version, computed_at, window_hours, horizon_minutes,
          group_key, group_value, n, brier_score, log_loss, rank_loss, observed_rate,
          mean_prediction, count_log_loss, crps, ece, recommended_hit_rate,
          distance_adjusted_regret, decision_rank_loss, is_active_model, is_best_model
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    return len(rows)


def _all_station_candidates(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
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


def forecast_all_stations(conn: duckdb.DuckDBPyConnection, *, source: str = "evaluator") -> dict:
    init_schema(conn)
    candidates = _all_station_candidates(conn)
    if candidates.empty:
        return {"status": "no_stations", "forecasts_logged": 0, "stations_scored": 0}
    scored, fitted = predictor.score_candidates(conn, candidates)
    logged = log_forecasts(
        conn,
        scored,
        user_lat=None,
        user_lon=None,
        source=source,
        recommended_station_ids=[],
    )
    return {
        "status": "ok",
        "stations_scored": int(len(scored)),
        "forecasts_logged": int(logged),
        "model_method": fitted.method,
        "training_examples": fitted.n_examples,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Divvy model evaluation utilities")
    parser.add_argument("command", choices=["resolve", "summary", "snapshot", "forecast-all", "loop"])
    parser.add_argument("--window-hours", type=int, default=24)
    parser.add_argument("--interval-seconds", type=int, default=300)
    args = parser.parse_args(argv)

    with db.session(read_only=False) as conn:
        init_schema(conn)
        if args.command == "resolve":
            print(resolve_due_outcomes(conn))
        elif args.command == "summary":
            print(performance_summary(conn, window_hours=args.window_hours))
        elif args.command == "snapshot":
            print(snapshot_metrics(conn, window_hours=args.window_hours))
        elif args.command == "forecast-all":
            print(forecast_all_stations(conn))
        elif args.command == "loop":
            while True:
                result = forecast_all_stations(conn)
                resolved = resolve_due_outcomes(conn)
                metrics = snapshot_metrics(conn, window_hours=args.window_hours)
                print({"forecast": result, "resolved": resolved, "metrics": metrics})
                time.sleep(args.interval_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
