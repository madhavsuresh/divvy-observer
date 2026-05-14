from __future__ import annotations

import json
import logging
import math
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import duckdb
import numpy as np
import pandas as pd

from . import config, model_eval, predictor

log = logging.getLogger("divvy.forecast_queue")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _subdir(name: str) -> Path:
    path = config.FORECAST_QUEUE_DIR / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _json_value(value):
    if isinstance(value, pd.Timestamp):
        if value.tzinfo is not None:
            value = value.tz_convert("UTC").tz_localize(None)
        return value.isoformat()
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            value = value.astimezone(timezone.utc).replace(tzinfo=None)
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _scored_records(scored: pd.DataFrame) -> list[dict]:
    records = []
    for record in scored.to_dict(orient="records"):
        records.append({key: _json_value(value) for key, value in record.items()})
    return records


def _has_value(value) -> bool:
    if value is None:
        return False
    try:
        return not bool(pd.isna(value))
    except (TypeError, ValueError):
        return True


def forecast_row_count(scored: pd.DataFrame) -> int:
    count = 0
    for _, row in scored.iterrows():
        for horizon in predictor.HORIZONS:
            for model_key in predictor.MODEL_KEYS:
                suffix = f"_{horizon}m_{model_key}"
                p = row.get(f"p_has_ebike{suffix}")
                active_model_key = row.get("active_model_key") or predictor.ACTIVE_MODEL_KEY
                if not _has_value(p) and model_key == active_model_key:
                    p = row.get(f"p_has_ebike_{horizon}m")
                if _has_value(p):
                    count += 1
    return count


def enqueue_recommendation_forecasts(
    scored: pd.DataFrame,
    *,
    request_id: str,
    source: str,
    user_lat: float,
    user_lon: float,
    near_radius_km: float,
    search_radius_km: float,
    query_label: str | None,
    query_place_key: str | None,
    candidate_count: int,
    best_station_id: str | None,
    recommended_station_ids: Iterable[str] | None,
    recommended_station_ids_by_model: dict[str, Iterable[str] | str] | None = None,
    active_model_key: str | None = None,
    active_model_source: str | None = None,
    best_evaluated_model_key: str | None = None,
) -> dict:
    """Queue recommendation forecast writes for the collector process.

    The API can safely create small JSON files while the collector remains the
    only DuckDB writer. Files are atomically renamed into `pending`.
    """
    config.ensure_dirs()
    pending = _subdir("pending")
    tmp = _subdir("tmp")
    queued_at = _utc_now()
    forecast_rows = forecast_row_count(scored)
    active_model_key = active_model_key or (
        str(scored["active_model_key"].dropna().iloc[0])
        if "active_model_key" in scored and scored["active_model_key"].notna().any()
        else predictor.ACTIVE_MODEL_KEY
    )
    payload = {
        "kind": "recommendation_forecasts",
        "schema_version": 2,
        "queued_at": queued_at.isoformat(),
        "request_id": request_id,
        "source": source,
        "active_model_key": active_model_key,
        "active_model_source": active_model_source,
        "best_evaluated_model_key": best_evaluated_model_key,
        "query": {
            "lat": user_lat,
            "lon": user_lon,
            "near_radius_km": near_radius_km,
            "search_radius_km": search_radius_km,
            "query_label": query_label,
            "query_place_key": query_place_key,
            "candidate_count": candidate_count,
            "best_station_id": best_station_id,
        },
        "recommended_station_ids": [str(station_id) for station_id in (recommended_station_ids or [])],
        "recommended_station_ids_by_model": {
            str(key): ([str(values)] if isinstance(values, str) else [str(value) for value in values])
            for key, values in (recommended_station_ids_by_model or {}).items()
        },
        "scored_rows": _scored_records(scored),
        "forecast_row_count": forecast_rows,
    }
    filename = f"{queued_at.strftime('%Y%m%dT%H%M%S%f')}-{request_id or uuid.uuid4()}.json"
    tmp_path = tmp / f"{filename}.tmp"
    final_path = pending / filename
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"))
    os.replace(tmp_path, final_path)
    return {
        "queued_forecast_rows": forecast_rows,
        "queue_file": final_path.name,
    }


def _load_payload(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _process_payload(conn: duckdb.DuckDBPyConnection, payload: dict) -> int:
    if payload.get("kind") != "recommendation_forecasts":
        return 0
    query = payload.get("query") or {
        "lat": payload.get("user_lat"),
        "lon": payload.get("user_lon"),
        "near_radius_km": payload.get("near_radius_km"),
        "search_radius_km": payload.get("search_radius_km"),
        "query_label": payload.get("query_label"),
        "query_place_key": payload.get("query_place_key"),
        "candidate_count": payload.get("candidate_count"),
        "best_station_id": payload.get("best_station_id"),
    }
    request_id = payload.get("request_id") or str(uuid.uuid4())
    source = payload.get("source") or "api"
    if query.get("lat") is None or query.get("lon") is None:
        return 0
    place_key = query.get("query_place_key") or model_eval.query_place_key(query["lat"], query["lon"])
    model_eval.log_prediction_query(
        conn,
        request_id=request_id,
        source=source,
        lat=float(query["lat"]),
        lon=float(query["lon"]),
        near_radius_km=float(query.get("near_radius_km") or 0.5),
        search_radius_km=float(query.get("search_radius_km") or 1.5),
        query_label=query.get("query_label"),
        candidate_count=query.get("candidate_count"),
        best_station_id=query.get("best_station_id"),
        queried_at=payload.get("queued_at"),
        place_key=place_key,
    )
    scored = pd.DataFrame(payload.get("scored_rows") or [])
    if scored.empty:
        return 0
    return model_eval.log_forecasts(
        conn,
        scored,
        user_lat=float(query["lat"]),
        user_lon=float(query["lon"]),
        request_id=request_id,
        source=source,
        recommended_station_ids=payload.get("recommended_station_ids") or [],
        recommended_station_ids_by_model=payload.get("recommended_station_ids_by_model") or {},
        query_place_key=place_key,
        query_label=query.get("query_label"),
        active_model_key=payload.get("active_model_key"),
        active_model_source=payload.get("active_model_source"),
        best_evaluated_model_key=payload.get("best_evaluated_model_key"),
    )


def drain_forecast_queue(conn: duckdb.DuckDBPyConnection, *, limit: int | None = None) -> dict:
    pending = _subdir("pending")
    processing = _subdir("processing")
    failed = _subdir("failed")
    files = sorted(pending.glob("*.json"))
    if limit is not None:
        files = files[:limit]
    result = {
        "files_processed": 0,
        "forecast_rows_logged": 0,
        "files_failed": 0,
    }
    for path in files:
        work_path = processing / path.name
        try:
            os.replace(path, work_path)
            payload = _load_payload(work_path)
            logged = _process_payload(conn, payload)
            work_path.unlink(missing_ok=True)
            result["files_processed"] += 1
            result["forecast_rows_logged"] += int(logged)
        except Exception:
            result["files_failed"] += 1
            failed_path = failed / work_path.name
            try:
                os.replace(work_path, failed_path)
            except OSError:
                pass
            log.exception("failed to drain forecast queue file %s", path.name)
    return result
