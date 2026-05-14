from __future__ import annotations

import math
import uuid
from datetime import datetime, timezone
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import requests

from . import config, live_cache, model_eval, predictor, tile, tile_predictor
from .display import display_probability


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2.0) ** 2
    )
    return 2.0 * 6371.0 * math.asin(math.sqrt(a))


def _clean(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.floating, float)):
        if not math.isfinite(float(value)):
            return None
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime().isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if pd.isna(value):
        return None
    return value


def _prob_pct(value: float | None) -> str:
    return display_probability(value)


def _model_predictions(row: pd.Series) -> list[dict]:
    predictions = []
    for model_key in predictor.MODEL_KEYS:
        spec = predictor.MODEL_SPECS[model_key]
        item = {
            "model_key": model_key,
            "label": spec["label"],
            "version": spec["version"],
            "p_arrival": _clean(row.get(f"p_arrival_{model_key}")),
            "arrival_time_minutes": _clean(row.get("arrival_time_minutes")),
            "reliable_probability_lcb": _clean(row.get(f"reliable_probability_lcb_{model_key}")),
            "walk_adjusted_score": _clean(row.get(f"walk_adjusted_score_{model_key}")),
            "rank_probability": _clean(row.get(f"rank_probability_{model_key}")),
            "recommended_rank": _clean(row.get(f"recommended_rank_{model_key}")),
        }
        for horizon in predictor.HORIZONS:
            item[f"p_has_ebike_{horizon}m"] = _clean(row.get(f"p_has_ebike_{horizon}m_{model_key}"))
            item[f"p_zero_{horizon}m"] = _clean(row.get(f"p_zero_{horizon}m_{model_key}"))
            item[f"p_appears_{horizon}m"] = _clean(row.get(f"p_appears_{horizon}m_{model_key}"))
            item[f"p_survives_{horizon}m"] = _clean(row.get(f"p_survives_{horizon}m_{model_key}"))
            item[f"p_learned_{horizon}m"] = _clean(row.get(f"p_learned_{horizon}m_{model_key}"))
            item[f"p_empirical_{horizon}m"] = _clean(row.get(f"p_empirical_{horizon}m_{model_key}"))
            item[f"expected_ebikes_{horizon}m"] = _clean(row.get(f"expected_ebikes_{horizon}m_{model_key}"))
            item[f"expected_total_bikes_{horizon}m"] = _clean(row.get(f"expected_total_bikes_{horizon}m_{model_key}"))
            item[f"p_count_ebikes_{horizon}m"] = _clean(row.get(f"p_count_ebikes_{horizon}m_{model_key}"))
            item[f"p_capacity_violation_{horizon}m"] = _clean(row.get(f"p_capacity_violation_{horizon}m_{model_key}"))
            item[f"p_dock_constrained_arrival_{horizon}m"] = _clean(row.get(f"p_dock_constrained_arrival_{horizon}m_{model_key}"))
            item[f"sample_size_{horizon}m"] = _clean(row.get(f"sample_size_{horizon}m_{model_key}"))
        predictions.append(item)
    return predictions


def _model_feature_snapshot(row: pd.Series) -> dict:
    return {
        "calendar": {
            "is_commute_hour": _clean(row.get("is_commute_hour")),
            "is_federal_holiday": _clean(row.get("is_federal_holiday")),
            "month_sin": _clean(row.get("month_sin")),
            "month_cos": _clean(row.get("month_cos")),
            "day_of_year_sin": _clean(row.get("day_of_year_sin")),
            "day_of_year_cos": _clean(row.get("day_of_year_cos")),
        },
        "station_history": {
            "station_same_hour_rate": _clean(row.get("station_same_hour_rate")),
            "nearby_same_hour_rate": _clean(row.get("nearby_same_hour_rate")),
            "station_neighbor_same_hour_rate": _clean(row.get("station_neighbor_same_hour_rate")),
            "station_neighbor_recent_ebikes": _clean(row.get("station_neighbor_recent_ebikes")),
            "station_neighbor_recent_zero_rate": _clean(row.get("station_neighbor_recent_zero_rate")),
            "trend_5m": _clean(row.get("trend_5m")),
            "trend_10m": _clean(row.get("trend_10m")),
            "trend_15m": _clean(row.get("trend_15m")),
            "churn_rate": _clean(row.get("churn_rate")),
        },
        "trip_flow": {
            "trip_departures_same_hour_10m": _clean(row.get("trip_departures_same_hour_10m")),
            "trip_arrivals_same_hour_10m": _clean(row.get("trip_arrivals_same_hour_10m")),
            "trip_net_arrivals_same_hour_10m": _clean(row.get("trip_net_arrivals_same_hour_10m")),
            "trip_ebike_arrival_share_same_hour": _clean(row.get("trip_ebike_arrival_share_same_hour")),
            "trip_recent_departures_30m": _clean(row.get("trip_recent_departures_30m")),
            "trip_recent_arrivals_30m": _clean(row.get("trip_recent_arrivals_30m")),
            "trip_recent_net_arrivals_30m": _clean(row.get("trip_recent_net_arrivals_30m")),
            "route_inbound_trips_same_hour": _clean(row.get("route_inbound_trips_same_hour")),
            "route_inbound_ebike_share_same_hour": _clean(row.get("route_inbound_ebike_share_same_hour")),
            "route_inbound_median_duration_minutes": _clean(row.get("route_inbound_median_duration_minutes")),
            "route_inbound_due_horizon": _clean(row.get("route_inbound_due_horizon")),
        },
        "weather": {
            "temperature_2m": _clean(row.get("weather_temperature_2m")),
            "relative_humidity_2m": _clean(row.get("weather_relative_humidity_2m")),
            "apparent_temperature": _clean(row.get("weather_apparent_temperature")),
            "precipitation": _clean(row.get("weather_precipitation")),
            "rain": _clean(row.get("weather_rain")),
            "snowfall": _clean(row.get("weather_snowfall")),
            "snow_depth": _clean(row.get("weather_snow_depth")),
            "cloud_cover": _clean(row.get("weather_cloud_cover")),
            "wind_speed_10m": _clean(row.get("weather_wind_speed_10m")),
            "wind_gusts_10m": _clean(row.get("weather_wind_gusts_10m")),
            "bad_conditions": _clean(row.get("weather_bad_conditions")),
        },
    }


def fetch_live_free_bikes(
    lat: float,
    lon: float,
    *,
    radius_km: float = 1.5,
) -> dict | None:
    """Fetch the current GBFS free-floating bike feed and return the closest bike."""
    resp = requests.get(
        config.FREE_BIKE_STATUS_URL,
        headers={"User-Agent": config.USER_AGENT, "Accept": "application/json"},
        timeout=config.REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    closest = None
    for bike in resp.json().get("data", {}).get("bikes", []):
        if bike.get("is_disabled") or bike.get("is_reserved"):
            continue
        blat, blon = bike.get("lat"), bike.get("lon")
        if blat is None or blon is None:
            continue
        distance = haversine_km(lat, lon, float(blat), float(blon))
        if distance > radius_km:
            continue
        item = {
            "bike_id": bike.get("bike_id"),
            "name": bike.get("name"),
            "lat": float(blat),
            "lon": float(blon),
            "distance_km": distance,
        }
        if closest is None or distance < closest["distance_km"]:
            closest = item
    return closest


def fetch_live_free_bike_ids() -> set[str]:
    """Return the set of bike_ids currently present in the GBFS free-floating feed.

    Used to filter out bikes whose ``free_bike_status`` rows look "current" by
    timestamp but that have actually been removed from the feed (the table has
    no removal events because of insert-on-change dedup).
    """
    resp = requests.get(
        config.FREE_BIKE_STATUS_URL,
        headers={"User-Agent": config.USER_AGENT, "Accept": "application/json"},
        timeout=config.REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return {
        str(bike["bike_id"])
        for bike in resp.json().get("data", {}).get("bikes", [])
        if bike.get("bike_id") is not None
    }


def _station_payload(row: pd.Series | None) -> dict | None:
    if row is None:
        return None
    p5 = _clean(row.get("p_has_ebike_5m"))
    p10 = _clean(row.get("p_has_ebike_10m"))
    p_arrival = _clean(row.get("p_arrival"))
    arrival_minutes = _clean(row.get("arrival_time_minutes"))
    current = _clean(row.get("num_ebikes_available")) or 0
    distance = _clean(row.get("distance_km"))
    confidence = _clean(row.get("confidence"))
    walk_score = _clean(row.get("walk_adjusted_score"))
    distance_score = _clean(row.get("distance_score"))
    reason_parts = [
        f"{_prob_pct(p_arrival)} calibrated arrival probability",
        f"{arrival_minutes or '?'} min arrival",
        f"active model {row.get('active_model_key') or predictor.ACTIVE_MODEL_KEY}",
        f"{int(current)} now",
    ]
    if distance is not None:
        reason_parts.append(f"{distance:.2f} km away")
    if confidence:
        reason_parts.append(f"{confidence} confidence")
    if row.get("prediction_cache_status"):
        reason_parts.append(str(row.get("prediction_cache_status")))

    return {
        "station_id": _clean(row.get("station_id")),
        "name": _clean(row.get("name")),
        "short_name": _clean(row.get("short_name")),
        "lat": _clean(row.get("lat")),
        "lon": _clean(row.get("lon")),
        "distance_km": distance,
        "capacity": _clean(row.get("capacity")),
        "current_ebikes": int(current),
        "current_bikes": _clean(row.get("num_bikes_available")),
        "current_docks": _clean(row.get("num_docks_available")),
        "last_reported": _clean(row.get("last_reported")),
        "data_age_minutes": _clean(row.get("data_age_minutes")),
        "p_has_ebike_5m": p5,
        "p_has_ebike_10m": p10,
        "p_has_ebike_15m": _clean(row.get("p_has_ebike_15m")),
        "p_has_ebike_20m": _clean(row.get("p_has_ebike_20m")),
        "p_arrival": p_arrival,
        "arrival_time_minutes": arrival_minutes,
        "reliable_probability_lcb": _clean(row.get("reliable_probability_lcb")),
        "active_model_key": _clean(row.get("active_model_key")),
        "prediction_cache_status": _clean(row.get("prediction_cache_status")),
        "p_zero_5m": _clean(row.get("p_zero_5m")),
        "p_zero_10m": _clean(row.get("p_zero_10m")),
        "p_zero_15m": _clean(row.get("p_zero_15m")),
        "p_zero_20m": _clean(row.get("p_zero_20m")),
        "p_appears_5m": _clean(row.get("p_appears_5m")),
        "p_appears_10m": _clean(row.get("p_appears_10m")),
        "p_survives_5m": _clean(row.get("p_survives_5m")),
        "p_survives_10m": _clean(row.get("p_survives_10m")),
        "expected_ebikes_5m": _clean(row.get("expected_ebikes_5m")),
        "expected_ebikes_10m": _clean(row.get("expected_ebikes_10m")),
        "expected_ebikes_15m": _clean(row.get("expected_ebikes_15m")),
        "expected_ebikes_20m": _clean(row.get("expected_ebikes_20m")),
        "expected_total_bikes_10m": _clean(row.get("expected_total_bikes_10m")),
        "p_count_ebikes_5m": _clean(row.get("p_count_ebikes_5m")),
        "p_count_ebikes_10m": _clean(row.get("p_count_ebikes_10m")),
        "p_count_total_10m": _clean(row.get("p_count_total_10m")),
        "p_capacity_violation_10m": _clean(row.get("p_capacity_violation_10m")),
        "p_dock_constrained_arrival_10m": _clean(row.get("p_dock_constrained_arrival_10m")),
        "expected_ebike_departures_10m": _clean(row.get("expected_ebike_departures_10m")),
        "expected_ebike_arrivals_10m": _clean(row.get("expected_ebike_arrivals_10m")),
        "sample_size_5m": _clean(row.get("sample_size_5m")),
        "sample_size_10m": _clean(row.get("sample_size_10m")),
        "confidence": confidence,
        "rank_score": _clean(row.get("rank_score")),
        "walk_adjusted_score": walk_score,
        "rank_probability": _clean(row.get("rank_probability")),
        "distance_score": distance_score,
        "current_count_score": _clean(row.get("current_count_score")),
        "distance_reference_km": _clean(row.get("distance_reference_km")),
        "model_predictions": _model_predictions(row),
        "model_features": _model_feature_snapshot(row),
        "reason": ", ".join(reason_parts),
    }


def _first_row(df: pd.DataFrame) -> pd.Series | None:
    if df.empty:
        return None
    return df.iloc[0]


def _model_payload(fitted: predictor.FittedModelSuite, prediction_cache_status: str) -> dict:
    active_meta = predictor.model_meta_from_fitted(fitted.active_key, fitted.active)
    active_warning = fitted.active.model_warning
    if fitted.active_key in predictor.SOTA_PRIMARY_MODEL_KEYS and predictor.is_bootstrap_or_fallback(active_meta):
        active_warning = "No trained SOTA artifact registered; using capped provisional probabilities."
    return {
        "active_model_key": fitted.active_key,
        "active_model_label": fitted.active.label,
        "active_model_display_label": predictor.model_display_label(active_meta),
        "active_artifact_id": fitted.active.artifact_id,
        "artifact_id": fitted.active.artifact_id,
        "active_model_source": fitted.active_source,
        "active_model_warning": active_warning,
        "best_evaluated_model_key": fitted.best_evaluated_model_key,
        "best_evaluated_model_label": predictor.MODEL_SPECS.get(fitted.best_evaluated_model_key or "", {}).get("label"),
        "best_usable_model_key": fitted.best_usable_model_key,
        "best_usable_model_label": predictor.MODEL_SPECS.get(fitted.best_usable_model_key or "", {}).get("label"),
        "best_sota_model_key": fitted.best_sota_model_key,
        "best_trained_sota_model_key": fitted.best_trained_sota_model_key,
        "best_baseline_model_key": fitted.best_baseline_model_key,
        "active_equals_best": bool(fitted.active_key == fitted.best_evaluated_model_key) if fitted.best_evaluated_model_key else None,
        "active_equals_best_evaluated": bool(fitted.active_key == fitted.best_evaluated_model_key) if fitted.best_evaluated_model_key else None,
        "active_equals_best_usable": bool(fitted.active_key == fitted.best_usable_model_key) if fitted.best_usable_model_key else None,
        "selection_metric": fitted.selection_metric,
        "selection_window_hours": fitted.selection_window_hours,
        "prediction_cache_status": prediction_cache_status,
        "model_warning": active_warning,
        "version": fitted.model_version,
        "baseline_version": predictor.BASELINE_VERSION,
        "method": fitted.method,
        "trained_at": fitted.trained_at.isoformat(),
        "training_examples": fitted.n_examples,
        "training_positive": fitted.n_positive,
        "training_negative": fitted.n_negative,
        "models": fitted.summary(),
    }


def recommend(
    conn: duckdb.DuckDBPyConnection,
    *,
    lat: float,
    lon: float,
    near_radius_km: float = 0.5,
    search_radius_km: float = 1.5,
    source: str = "api",
    query_label: str | None = None,
    log_forecasts: bool = True,
    log_query: bool = True,
    resolve_outcomes: bool = True,
    initialize_schema: bool = True,
    include_internal: bool = False,
) -> dict:
    if initialize_schema:
        model_eval.init_schema(conn)
    resolved_now = model_eval.resolve_due_outcomes(conn) if resolve_outcomes else 0
    request_id = str(uuid.uuid4())
    generated_at = _utc_now()

    candidates = predictor.station_candidates(conn, lat, lon, radius_km=search_radius_km)
    fitted = predictor.get_availability_model_suite(conn)
    prediction_cache_status = "miss"
    if candidates.empty:
        scored = candidates.copy()
    else:
        station_ids = candidates["station_id"].astype(str).tolist()
        active_key = fitted.active_key
        try:
            active_cache = live_cache.latest_prediction_cache(
                conn,
                active_key,
                station_ids,
                max_age_minutes=config.CACHE_MAX_AGE_MINUTES,
            )
            comparison_cache = live_cache.latest_prediction_cache(
                conn,
                predictor.MODEL_KEYS,
                station_ids,
                max_age_minutes=config.COMPARISON_CACHE_MAX_AGE_MINUTES,
            )
            cache_rows = comparison_cache
            required = len(set(station_ids)) * len(predictor.HORIZONS)
            active_complete = len(active_cache) >= required
            if active_complete:
                scored = live_cache.cache_to_wide_candidates(
                    candidates,
                    cache_rows,
                    active_model_key=active_key,
                    search_radius_km=search_radius_km,
                )
                prediction_cache_status = "fresh"
            else:
                scored, fitted = predictor.score_candidates(conn, candidates, search_radius_km=search_radius_km)
                prediction_cache_status = "stale_fallback" if len(active_cache) else "miss_fallback"
        except Exception:
            scored, fitted = predictor.score_candidates(conn, candidates, search_radius_km=search_radius_km)
            prediction_cache_status = "stale_fallback"

    free_bike = None
    free_bike_error = None
    try:
        free_bike = fetch_live_free_bikes(lat, lon, radius_km=search_radius_km)
    except requests.RequestException as exc:
        free_bike_error = str(exc)

    if free_bike is not None:
        try:
            tile_id = tile.tile_id_for(free_bike["lat"], free_bike["lon"])
            if tile_id is not None:
                free_bike["tile_id"] = tile_id
                free_bike["tile_dwell_probability"] = {
                    str(h): p
                    for h, p in tile_predictor.score_single_bike(
                        conn, str(free_bike["bike_id"]), tile_id
                    ).items()
                }
        except Exception as exc:
            free_bike["tile_dwell_error"] = str(exc)

    if scored.empty:
        place_key = model_eval.query_place_key(lat, lon)
        if log_query:
            place_key = model_eval.log_prediction_query(
                conn,
                request_id=request_id,
                source=source,
                lat=lat,
                lon=lon,
                near_radius_km=near_radius_km,
                search_radius_km=search_radius_km,
                query_label=query_label,
                candidate_count=0,
                best_station_id=None,
            )
        payload = {
            "request_id": request_id,
            "generated_at": generated_at.isoformat(),
            "location": {"lat": lat, "lon": lon},
            "query_label": query_label,
            "query_place_key": place_key,
            "near_radius_km": near_radius_km,
            "search_radius_km": search_radius_km,
            "candidate_count": 0,
            "active_model_key": fitted.active_key,
            "best_current_model": fitted.best_evaluated_model_key,
            "best_evaluated_model_key": fitted.best_evaluated_model_key,
            "best_usable_model_key": fitted.best_usable_model_key,
            "best_trained_sota_model_key": fitted.best_trained_sota_model_key,
            "active_equals_best": bool(fitted.active_key == fitted.best_evaluated_model_key) if fitted.best_evaluated_model_key else None,
            "active_equals_best_evaluated": bool(fitted.active_key == fitted.best_evaluated_model_key) if fitted.best_evaluated_model_key else None,
            "active_equals_best_usable": bool(fitted.active_key == fitted.best_usable_model_key) if fitted.best_usable_model_key else None,
            "prediction_cache_status": prediction_cache_status,
            "resolved_outcomes": resolved_now,
            "forecast_rows_logged": 0,
            "closest_docked_station_with_ebike": None,
            "closest_free_ebike": free_bike,
            "free_bike_error": free_bike_error,
            "best_station_5_10m": None,
            "best_practical_station_5_10m": None,
            "highest_probability_station_5_10m": None,
            "reliable_alternatives": [],
            "ranking_formula": predictor.ranking_formula(search_radius_km),
            "model": _model_payload(fitted, prediction_cache_status),
            "message": "No docked stations with live status were found inside the search radius.",
        }
        if include_internal:
            payload["_scored_for_logging"] = scored
            payload["_recommended_station_ids"] = []
        return payload
    scored["prediction_cache_status"] = prediction_cache_status

    nearest_current = scored[scored["num_ebikes_available"].fillna(0) >= 1].sort_values("distance_km")
    practical = scored.sort_values(
        ["walk_adjusted_score", "rank_probability", "distance_km", "num_ebikes_available"],
        ascending=[False, False, True, False],
    )
    highest_probability = scored.sort_values(
        ["rank_probability", "distance_km", "num_ebikes_available"],
        ascending=[False, True, False],
    )

    practical_row = _first_row(practical)
    probability_row = _first_row(highest_probability)
    recommended_ids = {
        str(row["station_id"])
        for row in [practical_row, probability_row]
        if row is not None
    }
    recommended_station_ids_by_model: dict[str, list[str]] = {}
    recommended_station_id_by_model: dict[str, str | None] = {}
    recommended_rank_by_model: dict[str, int | None] = {}
    model_top_rows: list[pd.Series] = []
    for model_key in predictor.MODEL_KEYS:
        score_col = f"walk_adjusted_score_{model_key}"
        if score_col not in scored.columns:
            recommended_station_ids_by_model[model_key] = []
            recommended_station_id_by_model[model_key] = None
            recommended_rank_by_model[model_key] = None
            continue
        model_sorted = scored.sort_values(
            [score_col, f"rank_probability_{model_key}", "distance_km", "num_ebikes_available"],
            ascending=[False, False, True, False],
        )
        top = _first_row(model_sorted)
        if top is None:
            recommended_station_ids_by_model[model_key] = []
            recommended_station_id_by_model[model_key] = None
            recommended_rank_by_model[model_key] = None
            continue
        station_id = str(top["station_id"])
        recommended_station_ids_by_model[model_key] = [station_id]
        recommended_station_id_by_model[model_key] = station_id
        recommended_rank_by_model[model_key] = int(top.get(f"recommended_rank_{model_key}") or 1)
        model_top_rows.append(top)
    place_key = model_eval.query_place_key(lat, lon)
    if log_query:
        place_key = model_eval.log_prediction_query(
            conn,
            request_id=request_id,
            source=source,
            lat=lat,
            lon=lon,
            near_radius_km=near_radius_km,
            search_radius_km=search_radius_km,
            query_label=query_label,
            candidate_count=int(len(scored)),
            best_station_id=str(practical_row["station_id"]) if practical_row is not None else None,
        )
    logged = 0
    if log_forecasts:
        logged = model_eval.log_forecasts(
            conn,
            scored,
            user_lat=lat,
            user_lon=lon,
            request_id=request_id,
            source=source,
            recommended_station_ids=recommended_ids,
            recommended_station_ids_by_model=recommended_station_ids_by_model,
            query_place_key=place_key,
            query_label=query_label,
            active_model_key=fitted.active_key,
            active_model_source=fitted.active_source,
            best_evaluated_model_key=fitted.best_evaluated_model_key,
        )

    reliable_ids = []
    reliable_rows = []
    for _, row in practical.head(10).iterrows():
        sid = str(row["station_id"])
        if sid not in reliable_ids:
            reliable_ids.append(sid)
            reliable_rows.append(row)
    for row in model_top_rows:
        sid = str(row["station_id"])
        if sid not in reliable_ids:
            reliable_ids.append(sid)
            reliable_rows.append(row)
    reliable = pd.DataFrame(reliable_rows) if reliable_rows else practical.head(0)
    near = practical[practical["distance_km"] <= near_radius_km]
    near_best = _station_payload(_first_row(near)) if not near.empty else None

    freshness = scored["data_age_minutes"].dropna()
    db_freshness_minutes = float(freshness.min()) if not freshness.empty else None

    payload = {
        "request_id": request_id,
        "generated_at": generated_at.isoformat(),
        "location": {"lat": lat, "lon": lon},
        "query_label": query_label,
        "query_place_key": place_key,
        "near_radius_km": near_radius_km,
        "search_radius_km": search_radius_km,
        "candidate_count": int(len(scored)),
        "active_model_key": fitted.active_key,
        "best_current_model": fitted.best_evaluated_model_key,
        "best_evaluated_model_key": fitted.best_evaluated_model_key,
        "best_usable_model_key": fitted.best_usable_model_key,
        "best_trained_sota_model_key": fitted.best_trained_sota_model_key,
        "active_equals_best": bool(fitted.active_key == fitted.best_evaluated_model_key) if fitted.best_evaluated_model_key else None,
        "active_equals_best_evaluated": bool(fitted.active_key == fitted.best_evaluated_model_key) if fitted.best_evaluated_model_key else None,
        "active_equals_best_usable": bool(fitted.active_key == fitted.best_usable_model_key) if fitted.best_usable_model_key else None,
        "resolved_outcomes": int(resolved_now),
        "forecast_rows_logged": int(logged),
        "db_freshness_minutes": db_freshness_minutes,
        "prediction_cache_status": prediction_cache_status,
        "closest_docked_station_with_ebike": _station_payload(_first_row(nearest_current)),
        "closest_free_ebike": free_bike,
        "free_bike_error": free_bike_error,
        "best_station_5_10m": _station_payload(practical_row),
        "best_practical_station_5_10m": _station_payload(practical_row),
        "highest_probability_station_5_10m": _station_payload(probability_row),
        "best_station_within_near_radius": near_best,
        "reliable_alternatives": [_station_payload(row) for _, row in reliable.iterrows()],
        "recommended_station_id_by_model": recommended_station_id_by_model,
        "recommended_rank_by_model": recommended_rank_by_model,
        "ranking_formula": predictor.ranking_formula(search_radius_km),
        "model": _model_payload(fitted, prediction_cache_status),
    }
    if include_internal:
        payload["_scored_for_logging"] = scored
        payload["_recommended_station_ids"] = list(recommended_ids)
        payload["_recommended_station_ids_by_model"] = recommended_station_ids_by_model
    return payload
