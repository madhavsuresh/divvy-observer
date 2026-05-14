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


PLAN_HORIZONS = (5, 10, 15, 20)
PLAN_WALK_PENALTY_PER_MINUTE = 0.01
PLAN_WALKING_SPEED_KMH = 4.8
PLAN_WALK_BUFFER_MIN = 1.0
PLAN_PMF_MAX_SUPPORT = 21
PLAN_MAX_TSP_STOPS = 7


def _plan_walk_minutes(distance_km: float) -> float:
    speed = PLAN_WALKING_SPEED_KMH
    minutes = max(0.0, float(distance_km)) / speed * 60.0
    return minutes + PLAN_WALK_BUFFER_MIN


def _horizon_for_walk_minutes(minutes: float | None) -> int:
    try:
        m = float(minutes) if minutes is not None else 10.0
    except (TypeError, ValueError):
        m = 10.0
    if not math.isfinite(m):
        return 10
    if m <= 7:
        return 5
    if m <= 12:
        return 10
    if m <= 17:
        return 15
    return 20


def _normalize_pmf(arr: np.ndarray, *, max_support: int = PLAN_PMF_MAX_SUPPORT) -> np.ndarray:
    arr = np.where(np.isfinite(arr), arr, 0.0)
    arr = np.clip(arr, 0.0, None)
    if arr.size > max_support:
        head = arr[: max_support - 1].copy()
        tail = float(arr[max_support - 1 :].sum())
        arr = np.concatenate([head, [tail]])
    total = float(arr.sum())
    if total <= 0:
        result = np.zeros(max(arr.size, 1), dtype=float)
        result[0] = 1.0
        return result
    return arr / total


def _convolve_pmf(a: np.ndarray, b: np.ndarray, *, max_support: int = PLAN_PMF_MAX_SUPPORT) -> np.ndarray:
    return _normalize_pmf(np.convolve(a, b), max_support=max_support)


def _tail_prob_ge_k(pmf: np.ndarray, k: int) -> float:
    if k <= 0:
        return 1.0
    if k >= pmf.size:
        return 0.0
    return float(pmf[k:].sum())


def _poisson_pmf_truncated(mean: float, max_count: int) -> np.ndarray:
    """Poisson PMF on [0, max_count] with the tail mass placed at max_count."""
    max_count = max(0, int(max_count))
    if max_count == 0:
        return np.array([1.0])
    mean = max(0.0, float(mean)) if math.isfinite(float(mean)) else 0.0
    if mean <= 1e-9:
        out = np.zeros(max_count + 1)
        out[0] = 1.0
        return out
    probs = np.zeros(max_count + 1)
    p = math.exp(-min(mean, 700.0))
    probs[0] = p
    running = p
    for k in range(1, max_count):
        p = p * mean / k
        probs[k] = p
        running += p
    probs[max_count] = max(0.0, 1.0 - running)
    total = probs.sum()
    if total <= 0:
        out = np.zeros(max_count + 1)
        out[0] = 1.0
        return out
    return probs / total


def _station_count_pmf(row: Any, horizon_minutes: int) -> tuple[np.ndarray, bool]:
    """Return ``(pmf, degraded)``.

    Preference order:
      1. ``p_count_ebikes_{h}m["__full__"]`` (cached uncollapsed PMF).
      2. ``p_count_ebikes_full_{h}m`` (direct uncollapsed PMF on fresh scores).
      3. Poisson(``expected_ebikes_{h}m``) truncated at capacity — much better
         than the legacy ``5_plus``-collapsed bucket dict for k > 1.
      4. Bucket dict (last resort; produces inflated tails for large k).
      5. Point mass at the current observed count.
    Cases (3)-(5) set ``degraded=True``.
    """
    def _get(key: str, default=None):
        if isinstance(row, dict):
            return row.get(key, default)
        return row.get(key, default)

    pmf_dict = _get(f"p_count_ebikes_{int(horizon_minutes)}m")
    full = None
    if isinstance(pmf_dict, dict) and isinstance(pmf_dict.get("__full__"), (list, tuple)):
        full = pmf_dict["__full__"]
    else:
        direct_full = _get(f"p_count_ebikes_full_{int(horizon_minutes)}m")
        if isinstance(direct_full, (list, tuple)):
            full = direct_full

    if full is not None and len(full) > 0:
        arr = np.array([float(x) if x is not None else 0.0 for x in full], dtype=float)
        return _normalize_pmf(arr), False

    capacity_raw = _get("capacity")
    try:
        capacity = int(capacity_raw) if capacity_raw is not None and not pd.isna(capacity_raw) else 0
    except (TypeError, ValueError):
        capacity = 0
    cap_for_support = max(min(capacity if capacity > 0 else 20, 20), 1)

    expected_raw = _get(f"expected_ebikes_{int(horizon_minutes)}m")
    expected: float | None = None
    try:
        if expected_raw is not None and not pd.isna(expected_raw):
            expected = float(expected_raw)
            if not math.isfinite(expected):
                expected = None
    except (TypeError, ValueError):
        expected = None

    if expected is None and isinstance(pmf_dict, dict):
        # Derive a mean from the bucket dict (5_plus midpoint ~= 5.5 is a rough
        # but harmless anchor). Used only when no expected_ebikes is available.
        try:
            expected = sum(
                float(pmf_dict.get(str(i)) or 0.0) * i for i in range(5)
            ) + float(pmf_dict.get("5_plus") or 0.0) * 5.5
        except (TypeError, ValueError):
            expected = None

    if expected is not None:
        return _poisson_pmf_truncated(expected, cap_for_support), True

    if isinstance(pmf_dict, dict):
        arr = np.array(
            [float(pmf_dict.get(str(i)) or 0.0) for i in range(5)]
            + [float(pmf_dict.get("5_plus") or 0.0)],
            dtype=float,
        )
        return _normalize_pmf(arr), True

    current_raw = _get("num_ebikes_available")
    try:
        current = int(current_raw) if current_raw is not None and not pd.isna(current_raw) else 0
    except (TypeError, ValueError):
        current = 0
    idx = max(0, min(current, max(capacity, current, 1)))
    arr = np.zeros(idx + 1, dtype=float)
    arr[idx] = 1.0
    return _normalize_pmf(arr), True


def _free_bike_p_stays(bike: dict, horizon_minutes: int) -> float:
    p_stays_raw = bike.get("p_stays") or {}
    p = p_stays_raw.get(int(horizon_minutes))
    if p is None:
        p = p_stays_raw.get(str(horizon_minutes))
    if p is None:
        p = 0.5
    try:
        return float(min(1.0, max(0.0, float(p))))
    except (TypeError, ValueError):
        return 0.5


def _free_bike_count_pmf(bike: dict, horizon_minutes: int) -> np.ndarray:
    p = _free_bike_p_stays(bike, horizon_minutes)
    return np.array([1.0 - p, p], dtype=float)


def fetch_free_bikes_with_persistence(
    conn: duckdb.DuckDBPyConnection,
    lat: float,
    lon: float,
    *,
    radius_km: float = 1.5,
) -> list[dict]:
    """Enumerate free-floating ebikes within ``radius_km``, enriched with per-horizon ``p_stays``.

    Wraps ``tile_predictor.score_tiles`` so the full disk is scored in one DB call.
    """
    tile_ids = tile.tile_ids_in_disk(lat, lon, radius_km)
    if not tile_ids:
        return []
    try:
        live_bike_ids: set[str] | None = fetch_live_free_bike_ids()
    except Exception:
        live_bike_ids = None
    try:
        state_df, _ = tile_predictor.score_tiles(conn, tile_ids, live_bike_ids=live_bike_ids)
    except Exception:
        return []
    if state_df is None or state_df.empty:
        return []
    bikes: list[dict] = []
    seen_ids: set[str] = set()
    for _, row in state_df.iterrows():
        row_tile_id = row.get("tile_id")
        for bike in row.get("bikes") or []:
            bike_id = bike.get("bike_id")
            if bike_id is None:
                continue
            bike_id_str = str(bike_id)
            if bike_id_str in seen_ids:
                continue
            blat = bike.get("lat")
            blon = bike.get("lon")
            if blat is None or blon is None:
                continue
            distance = haversine_km(lat, lon, float(blat), float(blon))
            if distance > radius_km:
                continue
            p_stays_raw = bike.get("p_stays") or {}
            p_stays = {}
            for h, p in p_stays_raw.items():
                try:
                    p_stays[int(h)] = float(p)
                except (TypeError, ValueError):
                    continue
            seen_ids.add(bike_id_str)
            bikes.append({
                "bike_id": bike_id_str,
                "name": bike.get("name"),
                "lat": float(blat),
                "lon": float(blon),
                "distance_km": distance,
                "tile_id": row_tile_id,
                "dwell_seconds_so_far": bike.get("dwell_seconds_so_far"),
                "p_stays": p_stays,
                "p_stays_source": "tile_predictor",
            })
    return bikes


def _candidate_from_station(row: pd.Series) -> dict | None:
    sid = row.get("station_id")
    if sid is None:
        return None
    lat = row.get("lat")
    lon = row.get("lon")
    try:
        lat_f = float(lat)
        lon_f = float(lon)
    except (TypeError, ValueError):
        return None
    if math.isnan(lat_f) or math.isnan(lon_f):
        return None
    distance_km_raw = row.get("distance_km")
    try:
        distance_km = float(distance_km_raw) if distance_km_raw is not None else 0.0
    except (TypeError, ValueError):
        distance_km = 0.0
    walk_min = _plan_walk_minutes(distance_km)
    horizon = _horizon_for_walk_minutes(walk_min)
    pmf, degraded = _station_count_pmf(row, horizon)
    expected = float(np.dot(np.arange(pmf.size), pmf))
    p_has = float(pmf[1:].sum()) if pmf.size > 1 else 0.0
    return {
        "stop_type": "station",
        "station_id": str(sid),
        "name": row.get("name") or "Station",
        "lat": lat_f,
        "lon": lon_f,
        "distance_from_user_km": distance_km,
        "walk_minutes_from_user": walk_min,
        "horizon_minutes": int(horizon),
        "pmf": pmf,
        "expected_pickup": expected,
        "p_has_ebike_at_horizon": p_has,
        "current_ebikes": int(row.get("num_ebikes_available") or 0),
        "capacity": int(row.get("capacity") or 0),
        "degraded": degraded,
    }


def _candidate_from_free_bike(bike: dict) -> dict:
    distance_km = float(bike.get("distance_km") or 0.0)
    walk_min = _plan_walk_minutes(distance_km)
    horizon = _horizon_for_walk_minutes(walk_min)
    pmf = _free_bike_count_pmf(bike, horizon)
    p_stays = float(pmf[1])
    return {
        "stop_type": "free_bike",
        "bike_id": bike.get("bike_id"),
        "name": bike.get("name") or f"Free bike {bike.get('bike_id')}",
        "lat": float(bike["lat"]),
        "lon": float(bike["lon"]),
        "distance_from_user_km": distance_km,
        "walk_minutes_from_user": walk_min,
        "horizon_minutes": int(horizon),
        "tile_id": bike.get("tile_id"),
        "pmf": pmf,
        "expected_pickup": p_stays,
        "p_has_ebike_at_horizon": p_stays,
        "p_stays_source": bike.get("p_stays_source") or "tile_predictor",
        "degraded": False,
    }


def _strip_internal(stop: dict) -> dict:
    out = dict(stop)
    out.pop("pmf", None)
    return out


def _build_plan_greedy(
    candidates: list[dict],
    *,
    k: int,
    target_probability: float,
    user_lat: float,
    user_lon: float,
    walk_penalty: float,
    max_stops: int,
    min_stops: int = 0,
    ignore_walk_penalty: bool = False,
) -> dict:
    """Greedy plan builder.

    Phase 1: add stops by marginal probability lift until ``P(>=k) >= target_probability``
    or no positive lift remains.

    Phase 2 (robustness): if fewer than ``min_stops`` stops were chosen, keep adding
    stops by marginal *expected pickup* per added walking minute. This gives the
    user backup options (and surfaces free bikes "along the way") even when a
    single high-capacity station already satisfies the target on paper.
    """
    if not candidates:
        return {
            "stops": [],
            "success_probability": 0.0,
            "expected_bikes": 0.0,
            "feasible": False,
            "convolved_pmf": [1.0],
        }
    remaining = list(candidates)
    plan_pmf = np.array([1.0])
    plan_stops: list[dict] = []
    last_lat = user_lat
    last_lon = user_lon
    target_met = False
    while remaining and len(plan_stops) < max_stops:
        current_p = _tail_prob_ge_k(plan_pmf, k)
        if current_p >= target_probability:
            target_met = True
            break
        best_delta = -math.inf
        best_idx = -1
        best_new_pmf: np.ndarray | None = None
        best_added_walk = 0.0
        for idx, stop in enumerate(remaining):
            added_walk = _plan_walk_minutes(
                haversine_km(last_lat, last_lon, stop["lat"], stop["lon"])
            )
            new_pmf = _convolve_pmf(plan_pmf, stop["pmf"])
            new_p = _tail_prob_ge_k(new_pmf, k)
            lift = new_p - current_p
            penalty = 0.0 if ignore_walk_penalty else walk_penalty * added_walk
            delta = lift - penalty
            if delta > best_delta:
                best_delta = delta
                best_idx = idx
                best_new_pmf = new_pmf
                best_added_walk = added_walk
        if best_idx < 0 or best_new_pmf is None:
            break
        if best_delta <= 0 and current_p > 0:
            break
        chosen = remaining.pop(best_idx)
        chosen_clean = _strip_internal(chosen)
        chosen_clean["walk_minutes_from_previous"] = best_added_walk
        plan_stops.append(chosen_clean)
        plan_pmf = best_new_pmf
        last_lat = chosen["lat"]
        last_lon = chosen["lon"]

    # Phase 2: pick robust backup stops up to min_stops. The probability target
    # is already met, so prefer stops the user would naturally encounter — closest
    # to them — to surface free bikes and short-walk alternatives "along the way".
    # Score: expected_pickup / (walk_from_user_minutes)^1.5  (proximity dominates).
    while remaining and len(plan_stops) < min(min_stops, max_stops):
        best_score = -math.inf
        best_idx = -1
        best_added_walk = 0.0
        best_new_pmf: np.ndarray | None = None
        for idx, stop in enumerate(remaining):
            added_walk = _plan_walk_minutes(
                haversine_km(last_lat, last_lon, stop["lat"], stop["lon"])
            )
            walk_from_user = max(1.5, float(stop.get("walk_minutes_from_user") or 1.5))
            expected = max(0.05, float(stop.get("expected_pickup") or 0.0))
            score = expected / (walk_from_user ** 1.5)
            if score > best_score:
                best_score = score
                best_idx = idx
                best_added_walk = added_walk
                best_new_pmf = _convolve_pmf(plan_pmf, stop["pmf"])
        if best_idx < 0 or best_new_pmf is None:
            break
        chosen = remaining.pop(best_idx)
        chosen_clean = _strip_internal(chosen)
        chosen_clean["walk_minutes_from_previous"] = best_added_walk
        plan_stops.append(chosen_clean)
        plan_pmf = best_new_pmf
        last_lat = chosen["lat"]
        last_lon = chosen["lon"]

    final_p = _tail_prob_ge_k(plan_pmf, k)
    expected_bikes = float(min(k, np.dot(np.arange(plan_pmf.size), plan_pmf)))
    return {
        "stops": plan_stops,
        "success_probability": final_p,
        "expected_bikes": expected_bikes,
        "feasible": final_p >= target_probability,
        "convolved_pmf": plan_pmf.tolist(),
        "target_met_before_min_stops": bool(target_met),
    }


def _tsp_order(stops: list[dict], user_lat: float, user_lon: float) -> list[dict]:
    if not stops:
        return stops
    if len(stops) > PLAN_MAX_TSP_STOPS:
        prev_lat, prev_lon = user_lat, user_lon
        ordered: list[dict] = []
        for stop in stops:
            stop = dict(stop)
            stop["walk_minutes_from_previous"] = _plan_walk_minutes(
                haversine_km(prev_lat, prev_lon, stop["lat"], stop["lon"])
            )
            ordered.append(stop)
            prev_lat, prev_lon = stop["lat"], stop["lon"]
        return ordered
    from itertools import permutations

    indices = list(range(len(stops)))
    best_total = math.inf
    best_perm: tuple[int, ...] | None = None
    for perm in permutations(indices):
        total = 0.0
        prev_lat, prev_lon = user_lat, user_lon
        for i in perm:
            total += _plan_walk_minutes(
                haversine_km(prev_lat, prev_lon, stops[i]["lat"], stops[i]["lon"])
            )
            prev_lat = stops[i]["lat"]
            prev_lon = stops[i]["lon"]
        if total < best_total:
            best_total = total
            best_perm = perm
    if best_perm is None:
        return stops
    ordered = []
    prev_lat, prev_lon = user_lat, user_lon
    for i in best_perm:
        stop = dict(stops[i])
        stop["walk_minutes_from_previous"] = _plan_walk_minutes(
            haversine_km(prev_lat, prev_lon, stop["lat"], stop["lon"])
        )
        ordered.append(stop)
        prev_lat = stop["lat"]
        prev_lon = stop["lon"]
    return ordered


def _plan_summary(plan: dict, *, k: int, walk_penalty: float) -> dict:
    stops = plan["stops"]
    total_walk_min = sum(stop.get("walk_minutes_from_previous", 0.0) for stop in stops)
    total_distance_km = 0.0
    if stops:
        prev_lat = stops[0]["lat"]
        prev_lon = stops[0]["lon"]
        total_distance_km = stops[0].get("distance_from_user_km", 0.0)
        for stop in stops[1:]:
            total_distance_km += haversine_km(prev_lat, prev_lon, stop["lat"], stop["lon"])
            prev_lat = stop["lat"]
            prev_lon = stop["lon"]
    return {
        "k": k,
        "stops": stops,
        "n_stops": len(stops),
        "success_probability": float(plan["success_probability"]),
        "expected_bikes": float(plan["expected_bikes"]),
        "total_walking_minutes": float(total_walk_min),
        "total_walking_km": float(total_distance_km),
        "feasible": bool(plan["feasible"]),
        "utility": float(plan["success_probability"] - walk_penalty * total_walk_min),
        "convolved_pmf": plan["convolved_pmf"],
    }


def plan_multi_bike_acquisition(
    conn: duckdb.DuckDBPyConnection,
    *,
    lat: float,
    lon: float,
    k: int,
    near_radius_km: float = 0.5,
    search_radius_km: float = 1.5,
    plan_horizon_minutes: int = 10,
    target_success_probability: float = 0.85,
    query_label: str | None = None,
    source: str = "api",
    include_internal: bool = False,
) -> dict:
    """Build a plan to acquire ``k`` ebikes near (lat, lon), favoring high success probability."""
    k = int(max(1, min(10, int(k))))
    search_radius_km = float(min(3.0, max(0.1, search_radius_km)))
    near_radius_km = float(min(search_radius_km, max(0.05, near_radius_km)))
    request_id = str(uuid.uuid4())
    generated_at = _utc_now()

    base = recommend(
        conn,
        lat=lat,
        lon=lon,
        near_radius_km=near_radius_km,
        search_radius_km=search_radius_km,
        source=source,
        query_label=query_label,
        log_forecasts=False,
        log_query=False,
        resolve_outcomes=False,
        initialize_schema=False,
        include_internal=True,
    )
    scored = base.get("_scored_for_logging")
    if scored is None:
        scored = pd.DataFrame()

    station_stops: list[dict] = []
    if isinstance(scored, pd.DataFrame) and not scored.empty:
        for _, row in scored.iterrows():
            stop = _candidate_from_station(row)
            if stop is not None:
                station_stops.append(stop)

    free_bikes: list[dict] = []
    free_bike_error: str | None = base.get("free_bike_error")
    try:
        free_bikes = fetch_free_bikes_with_persistence(conn, lat, lon, radius_km=search_radius_km)
    except Exception as exc:
        free_bike_error = str(exc)
    free_bike_stops = [_candidate_from_free_bike(bike) for bike in free_bikes]

    all_candidates = sorted(
        station_stops + free_bike_stops,
        key=lambda s: (-float(s["expected_pickup"]), float(s["distance_from_user_km"])),
    )

    max_stops = max(2 * k, 3)

    # Strategy A: marginal-lift greedy over all candidates (probability-aware).
    plan_a = _build_plan_greedy(
        all_candidates,
        k=k,
        target_probability=target_success_probability,
        user_lat=lat,
        user_lon=lon,
        walk_penalty=PLAN_WALK_PENALTY_PER_MINUTE,
        max_stops=max_stops,
    )
    plan_a["stops"] = _tsp_order(plan_a["stops"], lat, lon)

    # Strategy B: probability-only greedy (no walk penalty).
    plan_b = _build_plan_greedy(
        all_candidates,
        k=k,
        target_probability=target_success_probability,
        user_lat=lat,
        user_lon=lon,
        walk_penalty=PLAN_WALK_PENALTY_PER_MINUTE,
        max_stops=max_stops,
        ignore_walk_penalty=True,
    )
    plan_b["stops"] = _tsp_order(plan_b["stops"], lat, lon)

    # Strategy C: shortest-walk over the top-(2k) candidates.
    short_walk_pool = [dict(s) for s in all_candidates[: min(max_stops, len(all_candidates))]]
    short_walk_pool.sort(key=lambda s: s["distance_from_user_km"])
    plan_c_pmf = np.array([1.0])
    for stop in short_walk_pool:
        plan_c_pmf = _convolve_pmf(plan_c_pmf, stop["pmf"])
    plan_c_p = _tail_prob_ge_k(plan_c_pmf, k)
    plan_c_stops_ordered = _tsp_order([_strip_internal(s) for s in short_walk_pool], lat, lon)
    plan_c = {
        "stops": plan_c_stops_ordered,
        "success_probability": plan_c_p,
        "expected_bikes": float(min(k, np.dot(np.arange(plan_c_pmf.size), plan_c_pmf))),
        "feasible": plan_c_p >= target_success_probability,
        "convolved_pmf": plan_c_pmf.tolist(),
    }

    # Strategy D: free-bikes-first. Greedy over only the free-bike candidates
    # (sorted by proximity), then top up with stations if probability target
    # not reached. Lets a constellation of nearby free bikes form a valid plan
    # without being blocked by a single dominant far station.
    free_sorted = sorted(free_bike_stops, key=lambda s: s["distance_from_user_km"])
    station_sorted = sorted(station_stops, key=lambda s: s["distance_from_user_km"])
    plan_d_pmf = np.array([1.0])
    plan_d_stops: list[dict] = []
    last_lat_d, last_lon_d = lat, lon
    for stop in free_sorted:
        if len(plan_d_stops) >= max_stops:
            break
        if _tail_prob_ge_k(plan_d_pmf, k) >= target_success_probability:
            break
        added_walk = _plan_walk_minutes(haversine_km(last_lat_d, last_lon_d, stop["lat"], stop["lon"]))
        plan_d_pmf = _convolve_pmf(plan_d_pmf, stop["pmf"])
        chosen = _strip_internal(stop)
        chosen["walk_minutes_from_previous"] = added_walk
        plan_d_stops.append(chosen)
        last_lat_d, last_lon_d = stop["lat"], stop["lon"]
    for stop in station_sorted:
        if len(plan_d_stops) >= max_stops:
            break
        if _tail_prob_ge_k(plan_d_pmf, k) >= target_success_probability:
            break
        added_walk = _plan_walk_minutes(haversine_km(last_lat_d, last_lon_d, stop["lat"], stop["lon"]))
        plan_d_pmf = _convolve_pmf(plan_d_pmf, stop["pmf"])
        chosen = _strip_internal(stop)
        chosen["walk_minutes_from_previous"] = added_walk
        plan_d_stops.append(chosen)
        last_lat_d, last_lon_d = stop["lat"], stop["lon"]
    plan_d_p = _tail_prob_ge_k(plan_d_pmf, k)
    plan_d_stops_ordered = _tsp_order(plan_d_stops, lat, lon)
    plan_d = {
        "stops": plan_d_stops_ordered,
        "success_probability": plan_d_p,
        "expected_bikes": float(min(k, np.dot(np.arange(plan_d_pmf.size), plan_d_pmf))),
        "feasible": plan_d_p >= target_success_probability,
        "convolved_pmf": plan_d_pmf.tolist(),
    }

    plan_a_summary = _plan_summary(plan_a, k=k, walk_penalty=PLAN_WALK_PENALTY_PER_MINUTE)
    plan_b_summary = _plan_summary(plan_b, k=k, walk_penalty=PLAN_WALK_PENALTY_PER_MINUTE)
    plan_c_summary = _plan_summary(plan_c, k=k, walk_penalty=PLAN_WALK_PENALTY_PER_MINUTE)
    plan_d_summary = _plan_summary(plan_d, k=k, walk_penalty=PLAN_WALK_PENALTY_PER_MINUTE)
    plan_a_summary["strategy"] = "marginal_lift"
    plan_b_summary["strategy"] = "probability_only"
    plan_c_summary["strategy"] = "shortest_walk"
    plan_d_summary["strategy"] = "free_bikes_first"

    # Pick the primary by utility (probability minus walking cost). Plans that
    # do not even meet the probability target rank below feasible plans.
    candidate_plans = [plan_a_summary, plan_b_summary, plan_c_summary, plan_d_summary]

    def _plan_score(plan: dict) -> tuple[int, float, float]:
        return (
            int(bool(plan["feasible"])),
            float(plan["utility"]),
            -float(plan["total_walking_minutes"]),
        )

    candidate_plans.sort(key=_plan_score, reverse=True)
    primary_summary = candidate_plans[0]
    alternatives = candidate_plans[1:]

    plan_station_ids = [
        stop["station_id"]
        for stop in primary_summary["stops"]
        if stop.get("stop_type") == "station" and stop.get("station_id")
    ]
    selected_bike_ids = {
        stop.get("bike_id")
        for stop in primary_summary["stops"]
        if stop.get("stop_type") == "free_bike" and stop.get("bike_id") is not None
    }
    selected_free_bikes = [
        {
            "bike_id": stop.get("bike_id"),
            "name": stop.get("name"),
            "lat": stop.get("lat"),
            "lon": stop.get("lon"),
            "tile_id": stop.get("tile_id"),
            "p_stays_source": stop.get("p_stays_source"),
        }
        for stop in primary_summary["stops"]
        if stop.get("stop_type") == "free_bike"
    ]
    unused_free_bikes = sorted(
        [
            {
                "bike_id": stop.get("bike_id"),
                "name": stop.get("name"),
                "lat": stop.get("lat"),
                "lon": stop.get("lon"),
                "distance_km": stop.get("distance_from_user_km"),
                "walk_minutes_from_user": stop.get("walk_minutes_from_user"),
                "horizon_minutes": stop.get("horizon_minutes"),
                "p_stays": stop.get("p_has_ebike_at_horizon"),
                "tile_id": stop.get("tile_id"),
                "p_stays_source": stop.get("p_stays_source"),
            }
            for stop in free_bike_stops
            if stop.get("bike_id") not in selected_bike_ids
        ],
        key=lambda b: (b["distance_km"] or 0.0),
    )

    payload = {
        "request_id": request_id,
        "generated_at": generated_at.isoformat(),
        "location": {"lat": lat, "lon": lon},
        "k": k,
        "near_radius_km": near_radius_km,
        "search_radius_km": search_radius_km,
        "plan_horizon_minutes": plan_horizon_minutes,
        "target_success_probability": target_success_probability,
        "candidate_count": len(all_candidates),
        "station_candidate_count": len(station_stops),
        "free_bike_candidate_count": len(free_bike_stops),
        "plan": primary_summary,
        "alternative_plans": alternatives,
        "selected_free_bikes": selected_free_bikes,
        "unused_free_bikes": unused_free_bikes,
        "recommended_station_ids": plan_station_ids,
        "ranking_formula": predictor.ranking_formula(search_radius_km),
        "model": base.get("model") or {},
        "query_label": query_label,
        "query_place_key": base.get("query_place_key"),
        "free_bike_error": free_bike_error,
        "feasible": primary_summary["feasible"],
        "unmet_demand": (
            max(0, k - int(math.floor(primary_summary["expected_bikes"])))
            if not primary_summary["feasible"]
            else 0
        ),
        "prediction_cache_status": base.get("prediction_cache_status"),
        "db_freshness_minutes": base.get("db_freshness_minutes"),
        "active_model_key": base.get("active_model_key"),
    }
    if include_internal:
        payload["_scored_for_logging"] = scored if isinstance(scored, pd.DataFrame) else pd.DataFrame()
        payload["_recommended_station_ids"] = plan_station_ids
        payload["_recommended_station_ids_by_model"] = base.get("_recommended_station_ids_by_model") or {}
    return payload
