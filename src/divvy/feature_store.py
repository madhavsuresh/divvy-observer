from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

import duckdb
import numpy as np
import pandas as pd


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _json_clean(value):
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
    if isinstance(value, float) and not np.isfinite(value):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def build_online_feature_rows(
    conn: duckdb.DuckDBPyConnection,
    candidates: pd.DataFrame,
    now,
    horizons,
    user_lat: float | None = None,
    user_lon: float | None = None,
) -> pd.DataFrame:
    """Compatibility wrapper around predictor's online feature construction."""
    del user_lat, user_lon
    from . import predictor

    if candidates.empty:
        return candidates.copy()
    station_ids = candidates["station_id"].astype(str).tolist()
    rates = predictor._history_rates_for_candidates(conn, station_ids, now)
    trends = predictor._latest_trends(conn, station_ids, now)
    graph_static = predictor._station_graph_static_features(conn)
    live_neighbor = predictor._live_neighbor_features(conn, station_ids, now)
    live_inflight = predictor._live_inflight_features(conn, station_ids, now)
    free_density = predictor._free_floating_density_features(conn, candidates, now)
    base = (
        candidates
        .merge(rates, on="station_id", how="left")
        .merge(trends, on="station_id", how="left")
        .merge(graph_static, on="station_id", how="left")
        .merge(live_neighbor, on="station_id", how="left")
        .merge(live_inflight, on="station_id", how="left")
        .merge(free_density, on="station_id", how="left")
    )
    base = predictor._add_inventory_features(base)
    for column, default in [
        ("station_same_hour_rate", 0.35),
        ("nearby_same_hour_rate", 0.35),
        ("station_neighbor_same_hour_rate", 0.35),
        ("station_neighbor_count_500m", 0.0),
        ("station_neighbor_capacity_500m", 0.0),
        ("station_neighbor_recent_ebikes", 0.0),
        ("station_neighbor_recent_zero_rate", 1.0),
        ("trend_5m", 0.0),
        ("trend_10m", 0.0),
        ("trend_15m", 0.0),
        ("churn_rate", 0.0),
        ("live_inflight_ebike_due_5m", 0.0),
        ("live_inflight_ebike_due_10m", 0.0),
        ("live_inflight_ebike_due_15m", 0.0),
        ("live_inflight_ebike_due_20m", 0.0),
        ("live_inflight_classic_due_5m", 0.0),
        ("live_inflight_classic_due_10m", 0.0),
        ("live_inflight_classic_due_15m", 0.0),
        ("live_inflight_classic_due_20m", 0.0),
        ("free_floating_density_300m", 0.0),
        ("free_floating_density_500m", 0.0),
        ("free_floating_density_1000m", 0.0),
    ]:
        if column not in base.columns:
            base[column] = default
        base[column] = pd.to_numeric(base[column], errors="coerce").fillna(default)
    latest_report = pd.to_datetime(base["last_reported"], errors="coerce")
    base["status_age_minutes"] = (pd.Timestamp(now) - latest_report).dt.total_seconds() / 60.0
    base["station_closed_penalty_flag"] = (~base.get("is_renting", pd.Series(True, index=base.index)).fillna(True).astype(bool)).astype(int)
    base["stale_status_penalty_flag"] = (base["status_age_minutes"].fillna(999.0) > 10.0).astype(int)

    rows = []
    for horizon in horizons:
        frame = base.copy()
        frame["forecasted_at"] = now
        frame["target_at"] = pd.Timestamp(now) + pd.Timedelta(minutes=int(horizon))
        frame["horizon_minutes"] = int(horizon)
        frame["current_ebikes_clipped"] = frame["num_ebikes_available"].fillna(0).clip(0, 6)
        frame["current_bucket"] = frame["num_ebikes_available"].fillna(0).map(predictor.current_bucket)
        rows.append(frame)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out = predictor.add_temporal_features(out, "forecasted_at")
    out = predictor.add_calendar_features(out, "forecasted_at")
    out = predictor._add_trip_features(conn, out, now=now)
    out = predictor._add_weather_features(conn, out, "forecasted_at")
    out = predictor._fill_feature_defaults(out, predictor.CALENDAR_FEATURE_COLUMNS)
    out = predictor._fill_feature_defaults(out, predictor.TRIP_FEATURE_COLUMNS)
    out = predictor._fill_feature_defaults(out, predictor.LIVE_INFLIGHT_FEATURE_COLUMNS)
    out = predictor._fill_feature_defaults(out, predictor.FREE_FLOATING_FEATURE_COLUMNS)
    out = predictor._fill_feature_defaults(out, predictor.STATUS_QUALITY_FEATURE_COLUMNS)
    out = predictor._weather_defaults(out)
    for column in predictor.FEATURE_COLUMNS:
        if column not in out.columns:
            out[column] = 0.0
    out[predictor.FEATURE_COLUMNS] = out[predictor.FEATURE_COLUMNS].fillna(0.0)
    return out


def build_training_feature_rows(conn: duckdb.DuckDBPyConnection, examples: pd.DataFrame) -> pd.DataFrame:
    del conn
    return examples.copy()


def add_free_floating_features(
    conn: duckdb.DuckDBPyConnection,
    rows: pd.DataFrame,
    now,
) -> pd.DataFrame:
    out = rows.copy()
    out["free_floating_nearby_count"] = 0.0
    out["free_floating_density_1km"] = 0.0
    out["free_floating_density_300m"] = 0.0
    out["free_floating_density_500m"] = 0.0
    out["free_floating_density_1000m"] = 0.0
    try:
        if out.empty:
            return out
        fetched = pd.Timestamp(now).to_pydatetime()
        free = conn.execute(
            """
            SELECT lat, lon
            FROM free_bike_status
            WHERE fetched_at >= ? - INTERVAL '15 minutes'
              AND fetched_at <= ?
              AND COALESCE(is_reserved, false) = false
              AND COALESCE(is_disabled, false) = false
            """,
            [fetched, fetched],
        ).df()
        if free.empty or "lat" not in out or "lon" not in out:
            return out
        from .predictor import _haversine_np

        counts = []
        counts_300 = []
        counts_500 = []
        for _, row in out.iterrows():
            distances = _haversine_np(float(row["lat"]), float(row["lon"]), free["lat"].to_numpy(), free["lon"].to_numpy())
            counts_300.append(float((distances <= 0.3).sum()))
            counts_500.append(float((distances <= 0.5).sum()))
            counts.append(float((distances <= 1.0).sum()))
        out["free_floating_nearby_count"] = counts
        out["free_floating_density_1km"] = out["free_floating_nearby_count"] / np.pi
        out["free_floating_density_300m"] = pd.Series(counts_300, index=out.index) / (np.pi * 0.3 * 0.3)
        out["free_floating_density_500m"] = pd.Series(counts_500, index=out.index) / (np.pi * 0.5 * 0.5)
        out["free_floating_density_1000m"] = out["free_floating_density_1km"]
    except Exception:
        pass
    return out


def materialize_feature_snapshot(
    conn: duckdb.DuckDBPyConnection,
    rows: pd.DataFrame,
    model_key: str,
    request_id: str | None = None,
) -> pd.DataFrame:
    from . import db

    db.init_schema(conn)
    if rows.empty:
        return pd.DataFrame(columns=["feature_snapshot_id", "station_id", "horizon_minutes"])
    insert_rows = []
    mapping = []
    created = _utc_now()
    for _, row in rows.iterrows():
        snapshot_id = str(uuid.uuid4())
        station_id = str(row.get("station_id"))
        horizon = int(row.get("horizon_minutes") or 0)
        feature_json = json.dumps(
            {str(k): _json_clean(v) for k, v in row.to_dict().items()},
            separators=(",", ":"),
            sort_keys=True,
        )
        insert_rows.append((
            snapshot_id,
            request_id,
            model_key,
            station_id,
            pd.Timestamp(row.get("forecasted_at") or row.get("anchor_ts") or created).to_pydatetime(),
            horizon,
            feature_json,
            pd.Timestamp(row.get("last_reported")).to_pydatetime() if row.get("last_reported") is not None else None,
            created,
        ))
        mapping.append({
            "feature_snapshot_id": snapshot_id,
            "station_id": station_id,
            "horizon_minutes": horizon,
        })
    conn.executemany(
        """
        INSERT OR IGNORE INTO model_feature_snapshots (
          feature_snapshot_id, request_id, model_key, station_id, anchor_ts,
          horizon_minutes, feature_json, status_reported_at, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        insert_rows,
    )
    return pd.DataFrame(mapping)
