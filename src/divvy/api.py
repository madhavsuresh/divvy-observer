from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Annotated

import duckdb
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, model_validator

from . import automation, db, disabled_predictor, forecast_queue, live_cache, model_eval, predictor, recommendations, service_state, tile, tile_predictor


class RecommendationRequest(BaseModel):
    lat: float = Field(..., ge=-90.0, le=90.0)
    lon: float = Field(..., ge=-180.0, le=180.0)
    near_radius_km: float = Field(0.5, gt=0.0, le=5.0)
    search_radius_km: float = Field(1.5, gt=0.0, le=10.0)
    place_label: str | None = Field(None, max_length=240)

    @model_validator(mode="after")
    def _radius_order(self) -> "RecommendationRequest":
        if self.search_radius_km < self.near_radius_km:
            raise ValueError("search_radius_km must be greater than or equal to near_radius_km")
        return self


class BacktestRequest(BaseModel):
    history_hours: int = Field(24 * 7, ge=2, le=24 * 90)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield


app = FastAPI(
    title="Divvy Prediction Service",
    version="0.1.0",
    description="Server-side Divvy eBike availability recommendations and model evaluation.",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    try:
        with db.session(read_only=True) as conn:
            row = conn.execute(
                """
                SELECT
                  COUNT(*) AS station_status_rows,
                  COUNT(DISTINCT station_id) AS stations_with_status,
                  MIN(last_reported) AS first_reported,
                  MAX(last_reported) AS last_reported
                FROM station_status
                """
            ).fetchone()
    except duckdb.Error as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "ok": True,
        "db_path": str(db.config.DB_PATH),
        "station_status_rows": int(row[0] or 0),
        "stations_with_status": int(row[1] or 0),
        "first_reported": row[2].isoformat() if row and row[2] else None,
        "last_reported": row[3].isoformat() if row and row[3] else None,
    }


@app.post("/api/v1/recommendations")
def recommendation_endpoint(payload: RecommendationRequest) -> dict:
    try:
        with db.session(read_only=True) as conn:
            result = recommendations.recommend(
                conn,
                lat=payload.lat,
                lon=payload.lon,
                near_radius_km=payload.near_radius_km,
                search_radius_km=payload.search_radius_km,
                query_label=payload.place_label,
                source="api",
                log_forecasts=False,
                log_query=False,
                resolve_outcomes=False,
                initialize_schema=False,
                include_internal=True,
            )
    except duckdb.Error as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    scored = result.pop("_scored_for_logging", None)
    recommended_ids = result.pop("_recommended_station_ids", [])
    recommended_ids_by_model = result.pop("_recommended_station_ids_by_model", {})
    result["forecast_rows_logged"] = 0
    result["forecast_logging_mode"] = "queued_for_collector"
    if scored is not None:
        try:
            queue_result = forecast_queue.enqueue_recommendation_forecasts(
                scored,
                request_id=result["request_id"],
                source="api",
                user_lat=payload.lat,
                user_lon=payload.lon,
                near_radius_km=payload.near_radius_km,
                search_radius_km=payload.search_radius_km,
                query_label=payload.place_label,
                query_place_key=result.get("query_place_key"),
                candidate_count=int(result.get("candidate_count") or 0),
                best_station_id=(result.get("best_practical_station_5_10m") or {}).get("station_id"),
                recommended_station_ids=recommended_ids,
                recommended_station_ids_by_model=recommended_ids_by_model,
                active_model_key=(result.get("model") or {}).get("active_model_key"),
                active_model_source=(result.get("model") or {}).get("active_model_source"),
                best_evaluated_model_key=(result.get("model") or {}).get("best_evaluated_model_key"),
            )
            result["forecast_rows_queued"] = queue_result["queued_forecast_rows"]
        except Exception as exc:
            result["forecast_rows_queued"] = 0
            result["forecast_queue_error"] = str(exc)
    return result


@app.get("/api/v1/free_ebike/tile_state")
def free_ebike_tile_state(
    lat: Annotated[float, Query(ge=-90.0, le=90.0)],
    lon: Annotated[float, Query(ge=-180.0, le=180.0)],
    radius_km: Annotated[float, Query(gt=0.0, le=3.0)] = 0.5,
) -> dict:
    """Tiled view of the area: free-ebike persistence + dock predictions per H3 res-9 cell."""
    tile_ids = tile.tile_ids_in_disk(lat, lon, radius_km)
    if not tile_ids:
        return {"as_of": None, "center": {"lat": lat, "lon": lon}, "radius_km": radius_km, "tiles": []}

    try:
        live_bike_ids: set[str] | None = recommendations.fetch_live_free_bike_ids()
    except Exception:
        live_bike_ids = None

    try:
        with db.session(read_only=True) as conn:
            state_df, score_df = tile_predictor.score_tiles(
                conn, tile_ids, live_bike_ids=live_bike_ids
            )
    except duckdb.Error as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if state_df.empty:
        return {"as_of": None, "center": {"lat": lat, "lon": lon}, "radius_km": radius_km, "tiles": []}

    now = tile_predictor._utc_now()

    score_by_tile: dict[str, dict[int, dict]] = {}
    for _, srow in score_df.iterrows():
        score_by_tile.setdefault(srow["tile_id"], {})[int(srow["horizon_minutes"])] = srow.to_dict()

    tiles_payload = []
    for _, row in state_df.iterrows():
        tid = row["tile_id"]
        center = tile.tile_center(tid)
        boundary = tile.tile_boundary(tid)
        per_horizon = score_by_tile.get(tid, {})
        current_free = int(row["current_free_ebikes"])

        bikes_payload = [
            {
                "bike_id": bike["bike_id"],
                "name": bike["name"],
                "lat": bike["lat"],
                "lon": bike["lon"],
                "dwell_seconds_so_far": bike["dwell_seconds_so_far"],
                "p_stays": bike["p_stays"],
            }
            for bike in row["bikes"]
        ]
        reserved_bikes_payload = [
            {
                "bike_id": bike["bike_id"],
                "name": bike["name"],
                "lat": bike["lat"],
                "lon": bike["lon"],
                "dwell_seconds_so_far": bike["dwell_seconds_so_far"],
            }
            for bike in row.get("reserved_bikes", [])
        ]

        stations_payload = []
        for st in row["stations"]:
            stations_payload.append({
                "station_id": st["station_id"],
                "name": st["name"],
                "lat": st["lat"],
                "lon": st["lon"],
                "capacity": st["capacity"],
                "num_ebikes_available": st["num_ebikes_available"],
                "num_bikes_available": st["num_bikes_available"],
                "num_docks_available": st["num_docks_available"],
                "last_reported": st["last_reported"].isoformat() if st["last_reported"] else None,
            })

        horizons_payload = {}
        dock_predictions_as_of = None
        for h in predictor.HORIZONS:
            srow = per_horizon.get(int(h))
            if not srow:
                continue
            if srow.get("dock_predictions_as_of") is not None:
                dock_predictions_as_of = srow["dock_predictions_as_of"]
            horizons_payload[str(int(h))] = {
                "free": {
                    "p_has_bike": float(srow["free_p_has_bike"]),
                    "p_survives": float(srow["free_p_survives"]),
                    "p_appears": float(srow["free_p_appears"]),
                    "expected_count": float(srow["free_expected_count"]),
                    "expected_arrivals": float(srow["free_expected_arrivals"]),
                    "expected_departures": float(srow["free_expected_departures"]),
                    "expected_disabilities": float(srow["free_expected_disabilities"]),
                    "disability_rate_per_min": float(srow["free_disability_rate_per_min"]),
                    "p_count": dict(srow["free_p_count"]),
                },
                "docked": {
                    "p_any_has_ebike": float(srow["dock_p_any_has_ebike"]),
                    "expected_count": float(srow["dock_expected_count"]),
                    "per_station": list(srow["dock_per_station"]),
                },
                "combined": {
                    "p_any_ebike": float(srow["combined_p_any_ebike"]),
                    "total_expected_ebikes": float(srow["combined_total_expected_ebikes"]),
                },
            }

        tiles_payload.append({
            "tile_id": tid,
            "tile_center": {"lat": float(center[0]), "lon": float(center[1])},
            "tile_boundary": [[float(b[0]), float(b[1])] for b in boundary],
            "current_free_ebikes": current_free,
            "current_reserved_free_ebikes": int(row["current_reserved_free_ebikes"]),
            "n_stations_in_tile": int(row["n_stations_in_tile"]),
            "current_docked_ebikes": int(row["current_docked_ebikes"]),
            "churn_rate_5m": int(row["churn_rate_5m"]),
            "churn_rate_30m": int(row["churn_rate_30m"]),
            "reservation_events_5m": int(row["reservation_events_5m"]),
            "reservation_events_30m": int(row["reservation_events_30m"]),
            "last_change_at": row["last_change_at"].isoformat() if row["last_change_at"] else None,
            "horizons": horizons_payload,
            "bikes": bikes_payload,
            "reserved_bikes": reserved_bikes_payload,
            "stations": stations_payload,
            "dock_predictions_as_of": dock_predictions_as_of.isoformat() if dock_predictions_as_of else None,
        })

    return {
        "as_of": now.isoformat(),
        "center": {"lat": lat, "lon": lon},
        "radius_km": radius_km,
        "notes": {
            "dock_reservations": (
                "Divvy's station_status GBFS feed does not expose a per-station reserved count, "
                "so current_docked_reserved_ebikes is not available. Reservation counts here cover "
                "free-floating ebikes only."
            ),
            "free_floating_reservations": (
                "Reserved bikes drop out of the free_bike_status feed for the duration of the hold "
                "(~5 min). current_reserved_free_ebikes and reserved_bikes are derived from our "
                "historical record (rows with is_reserved=true within the last 10 min). "
                "reservation_events_5m/30m count observed false→true transitions, which Divvy reports "
                "only intermittently, so absolute counts under-represent true reservation volume."
            ),
            "disability_hazard_included": (
                "Free-bike p_stays / p_survives / expected_count include the per-bike hazard of "
                "becoming disabled (battery / damage) while in the tile, so 'still rideable here in "
                "H minutes' captures both ride-departure and disability drains. The disability share "
                "is reported separately as expected_disabilities and disability_rate_per_min. The "
                "underlying is_disabled flag is sparsely reported by Divvy, so this term is near "
                "zero in practice today but will scale up automatically as upstream coverage improves."
            ),
        },
        "tiles": tiles_payload,
    }


@app.get("/api/v1/disabled_bike/tile_state")
def disabled_bike_tile_state(
    lat: Annotated[float, Query(ge=-90.0, le=90.0)],
    lon: Annotated[float, Query(ge=-180.0, le=180.0)],
    radius_km: Annotated[float, Query(gt=0.0, le=3.0)] = 0.5,
) -> dict:
    """Tiled maintenance view: current disabled bikes + repair-time forecasts per H3 res-9 cell.

    Surfaces ``free_bike_status.is_disabled`` (sparse, individual-bike) and
    ``station_status.num_bikes_disabled`` (regularly non-zero, count-only)
    together. The audience is ops planning, not riders — use
    ``/api/v1/free_ebike/tile_state`` for the rider-side view.
    """
    tile_ids = tile.tile_ids_in_disk(lat, lon, radius_km)
    if not tile_ids:
        return {
            "as_of": None,
            "center": {"lat": lat, "lon": lon},
            "radius_km": radius_km,
            "tiles": [],
        }

    try:
        with db.session(read_only=True) as conn:
            state_df, score_df = disabled_predictor.score_tiles_disability(
                conn, tile_ids
            )
    except duckdb.Error as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if state_df.empty:
        return {
            "as_of": None,
            "center": {"lat": lat, "lon": lon},
            "radius_km": radius_km,
            "tiles": [],
        }

    now = disabled_predictor._utc_now()

    score_by_tile: dict[str, dict[float, dict]] = {}
    for _, srow in score_df.iterrows():
        score_by_tile.setdefault(srow["tile_id"], {})[float(srow["horizon_hours"])] = srow.to_dict()

    tiles_payload = []
    for _, row in state_df.iterrows():
        tid = row["tile_id"]
        center = tile.tile_center(tid)
        boundary = tile.tile_boundary(tid)
        per_horizon = score_by_tile.get(tid, {})

        disabled_free_bikes_payload = [
            {
                "bike_id": bike["bike_id"],
                "name": bike["name"],
                "lat": bike["lat"],
                "lon": bike["lon"],
                "disabled_since": bike["disabled_since"].isoformat() if bike["disabled_since"] else None,
                "dwell_seconds_so_far": bike["dwell_seconds_so_far"],
            }
            for bike in row["disabled_free_bikes"]
        ]

        disabled_stations_payload = [
            {
                "station_id": st["station_id"],
                "name": st["name"],
                "lat": st["lat"],
                "lon": st["lon"],
                "capacity": st["capacity"],
                "num_bikes_disabled": st["num_bikes_disabled"],
                "num_docks_disabled": st["num_docks_disabled"],
                "num_bikes_available": st["num_bikes_available"],
                "num_ebikes_available": st["num_ebikes_available"],
                "disability_events_24h": st["disability_events_24h"],
                "repair_events_24h": st["repair_events_24h"],
                "bike_hours_disabled_24h": st["bike_hours_disabled_24h"],
                "last_reported": st["last_reported"].isoformat() if st["last_reported"] else None,
            }
            for st in row["disabled_stations"]
        ]

        horizons_payload = {}
        for h, srow in per_horizon.items():
            horizons_payload[str(h)] = {
                "p_any_repair": float(srow["p_any_repair"]),
                "p_all_repaired": float(srow["p_all_repaired"]),
                "expected_repairs": float(srow["expected_repairs"]),
                "rate_per_hour": float(srow["rate_per_hour"]),
                "n_disabled": int(srow["n_disabled"]),
            }

        tiles_payload.append({
            "tile_id": tid,
            "tile_center": {"lat": float(center[0]), "lon": float(center[1])},
            "tile_boundary": [[float(b[0]), float(b[1])] for b in boundary],
            "current_disabled_free_ebikes": int(row["current_disabled_free_ebikes"]),
            "current_disabled_docked_bikes": int(row["current_disabled_docked_bikes"]),
            "current_disabled_docks": int(row["current_disabled_docks"]),
            "disability_events_24h": int(row["disability_events_24h"]),
            "repair_events_24h": int(row["repair_events_24h"]),
            "free_disability_events_24h": int(row["free_disability_events_24h"]),
            "free_repair_events_24h": int(row["free_repair_events_24h"]),
            "dock_bike_disability_events_24h": int(row["dock_bike_disability_events_24h"]),
            "dock_bike_repair_events_24h": int(row["dock_bike_repair_events_24h"]),
            "dock_disability_events_24h": int(row["dock_disability_events_24h"]),
            "dock_repair_events_24h": int(row["dock_repair_events_24h"]),
            "bike_hours_disabled_24h": float(row["bike_hours_disabled_24h"]),
            "median_free_dwell_minutes": float(row["median_free_dwell_minutes"]),
            "p90_free_dwell_minutes": float(row["p90_free_dwell_minutes"]),
            "n_stations_in_tile": int(row["n_stations_in_tile"]),
            "last_disability_event_at": row["last_disability_event_at"].isoformat() if row["last_disability_event_at"] else None,
            "horizons": horizons_payload,
            "disabled_free_bikes": disabled_free_bikes_payload,
            "disabled_stations": disabled_stations_payload,
        })

    return {
        "as_of": now.isoformat(),
        "center": {"lat": lat, "lon": lon},
        "radius_km": radius_km,
        "horizons_hours": list(disabled_predictor.HORIZONS_HOURS),
        "notes": {
            "docked_bike_id_unknown": (
                "Divvy's GBFS station_status feed exposes num_bikes_disabled as a per-station "
                "count, not a list of bike_ids. Per-bike disability dwell times are therefore "
                "only available for free-floating bikes (current_disabled_free_ebikes). For "
                "docked bikes we report aggregate bike_hours_disabled_24h instead — the integral "
                "of num_bikes_disabled over time, in bike-hours."
            ),
            "free_floating_undercount": (
                "Divvy reports the free_bike_status.is_disabled flag only intermittently — "
                "operators likely toggle bikes between states without the public feed always "
                "reflecting it — so the free-floating disability counts here under-count true "
                "volume the same way reservation_events does. The docked side is more complete."
            ),
            "repair_model": (
                "Repair-time horizons assume a constant-hazard Poisson model with rate fit from "
                "observed repair events at the matching (tile, hour-of-day) over the last 28 "
                "days, EB-shrunk toward the global rate. It can't capture early-vs-late repair "
                "propensity differences (e.g. truck routes that prefer just-broken bikes)."
            ),
        },
        "tiles": tiles_payload,
    }


@app.get("/api/v1/model/performance")
def model_performance(
    window_hours: Annotated[int, Query(ge=1, le=24 * 30)] = 24,
) -> dict:
    try:
        with db.session(read_only=True) as conn:
            health_payload = predictor.model_health(conn)
            performance = model_eval.performance_summary(
                conn,
                window_hours=window_hours,
                resolve=False,
                initialize_schema=False,
            )
    except duckdb.Error as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "model": health_payload,
        "performance": performance,
        "best_current_model": performance.get("best_current_model"),
        "active_model": performance.get("active_model"),
    }


@app.get("/api/v1/model/best")
def best_model_endpoint(
    window_hours: Annotated[int, Query(ge=1, le=24 * 30)] = 24,
    eligible: Annotated[str, Query(pattern="^(sota|all)$")] = "sota",
) -> dict:
    keys = predictor.SOTA_PRIMARY_MODEL_KEYS if eligible == "sota" else predictor.MODEL_KEYS
    try:
        with db.session(read_only=True) as conn:
            return model_eval.best_performing_model(
                conn,
                window_hours=window_hours,
                eligible_model_keys=keys,
            )
    except duckdb.Error as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/v1/system/status")
def system_status_endpoint() -> dict:
    try:
        with db.session(read_only=True) as conn:
            return service_state.system_status(conn, initialize_schema=False)
    except duckdb.Error as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/v1/system/jobs")
def system_jobs_endpoint() -> dict:
    try:
        with db.session(read_only=True) as conn:
            return {"jobs": service_state.latest_job_status(conn)}
    except duckdb.Error as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/v1/system/cache")
def system_cache_endpoint() -> dict:
    try:
        with db.session(read_only=True) as conn:
            return live_cache.cache_health(conn)
    except duckdb.Error as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/v1/automation/run-once")
def automation_run_once_endpoint(
    job: Annotated[str, Query(min_length=1)],
) -> dict:
    try:
        return automation.run_once(job)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except duckdb.Error as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/v1/automation/health")
def automation_health_endpoint() -> dict:
    try:
        return automation.health_payload()
    except duckdb.Error as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/v1/model/backtest")
def model_backtest(payload: BacktestRequest) -> dict:
    try:
        with db.session(read_only=True) as conn:
            result = predictor.backtest(conn, history_hours=payload.history_hours)
    except duckdb.Error as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return result
