from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import altair as alt
import duckdb
import pandas as pd
import pydeck as pdk
import requests
import streamlit as st
from streamlit_geolocation import streamlit_geolocation

from divvy import analytics, config, db, disabled_predictor, live_cache, model_eval, model_registry, model_selection, predictor, recommendations, service_state, tile, tile_predictor
from divvy.display import display_probability

DOW_LABELS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]


@st.cache_data(ttl=60)
def _stations() -> pd.DataFrame:
    with db.session(read_only=True) as conn:
        return analytics.station_options(conn)


@st.cache_data(ttl=120)
def _meta(station_id: str) -> dict:
    with db.session(read_only=True) as conn:
        return analytics.station_meta(conn, station_id)


@st.cache_data(ttl=120)
def _availability(station_id: str, start, end, require_renting: bool) -> pd.DataFrame:
    with db.session(read_only=True) as conn:
        return analytics.availability_heatmap(
            conn, station_id, start=start, end=end, require_renting=require_renting
        )


@st.cache_data(ttl=120)
def _range(station_id: str, start, end) -> pd.DataFrame:
    with db.session(read_only=True) as conn:
        return analytics.range_by_hour(conn, station_id, start=start, end=end)


@st.cache_data(ttl=120)
def _churn(station_id: str, start, end) -> pd.DataFrame:
    with db.session(read_only=True) as conn:
        return analytics.churn_by_hour(conn, station_id, start=start, end=end)


@st.cache_data(ttl=120)
def _ts(station_id: str, start, end, resample) -> pd.DataFrame:
    with db.session(read_only=True) as conn:
        return analytics.time_series(
            conn, station_id, start=start, end=end, resample_minutes=resample
        )


@st.cache_data(ttl=60)
def _ebike_summary(station_id: str, start, end) -> dict:
    with db.session(read_only=True) as conn:
        return analytics.ebike_summary(conn, station_id, start=start, end=end)


@st.cache_data(ttl=15)
def _collector_health() -> dict:
    pending_dir = config.FORECAST_QUEUE_DIR / "pending"
    failed_dir = config.FORECAST_QUEUE_DIR / "failed"
    pending_files = len(list(pending_dir.glob("*.json"))) if pending_dir.exists() else 0
    failed_files = len(list(failed_dir.glob("*.json"))) if failed_dir.exists() else 0
    replica_mtime = None
    if config.READ_DB_PATH.exists():
        replica_mtime = datetime.fromtimestamp(config.READ_DB_PATH.stat().st_mtime, tz=timezone.utc).replace(tzinfo=None)

    payload = {
        "pending_queue_files": pending_files,
        "failed_queue_files": failed_files,
        "read_replica_updated_at": replica_mtime,
    }
    try:
        with db.session(read_only=True) as conn:
            latest_tick = conn.execute(
                """
                SELECT
                  ticked_at,
                  station_payload_count,
                  free_bike_payload_count,
                  station_rows_inserted,
                  free_bike_events_inserted,
                  forecast_queue_files_processed,
                  forecast_rows_logged,
                  forecast_queue_files_failed,
                  outcomes_resolved,
                  metrics_rows_snapshotted
                FROM collector_ticks
                ORDER BY ticked_at DESC
                LIMIT 1
                """
            ).fetchone()
            recent = conn.execute(
                """
                SELECT
                  COUNT(*) AS ticks_10m,
                  COALESCE(SUM(station_rows_inserted), 0) AS station_rows_10m,
                  COALESCE(SUM(free_bike_events_inserted), 0) AS free_events_10m,
                  COALESCE(SUM(forecast_rows_logged), 0) AS forecasts_10m,
                  COALESCE(SUM(outcomes_resolved), 0) AS outcomes_10m
                FROM collector_ticks
                WHERE ticked_at >= now() - INTERVAL '10 minutes'
                """
            ).fetchone()
            freshness = conn.execute(
                """
                SELECT
                  (SELECT MAX(last_reported) FROM station_status) AS latest_station_reported,
                  (SELECT MAX(fetched_at) FROM station_status) AS latest_station_fetch,
                  (SELECT MAX(fetched_at) FROM free_bike_status) AS latest_free_bike_fetch
                """
            ).fetchone()
    except duckdb.Error as exc:
        payload["error"] = str(exc)
        return payload

    if latest_tick:
        payload.update({
            "latest_tick_at": latest_tick[0],
            "station_payload_count": int(latest_tick[1] or 0),
            "free_bike_payload_count": int(latest_tick[2] or 0),
            "station_rows_inserted": int(latest_tick[3] or 0),
            "free_bike_events_inserted": int(latest_tick[4] or 0),
            "forecast_queue_files_processed": int(latest_tick[5] or 0),
            "forecast_rows_logged": int(latest_tick[6] or 0),
            "forecast_queue_files_failed": int(latest_tick[7] or 0),
            "outcomes_resolved": int(latest_tick[8] or 0),
            "metrics_rows_snapshotted": int(latest_tick[9] or 0),
        })
    if recent:
        payload.update({
            "ticks_10m": int(recent[0] or 0),
            "station_rows_10m": int(recent[1] or 0),
            "free_events_10m": int(recent[2] or 0),
            "forecasts_10m": int(recent[3] or 0),
            "outcomes_10m": int(recent[4] or 0),
        })
    if freshness:
        payload.update({
            "latest_station_reported": freshness[0],
            "latest_station_fetch": freshness[1],
            "latest_free_bike_fetch": freshness[2],
        })
    return payload


@st.cache_data(ttl=30)
def _system_status_payload() -> dict:
    with db.session(read_only=True) as conn:
        status = service_state.system_status(conn, initialize_schema=False)
        try:
            selection = model_selection.latest_selection_state(conn)
        except Exception:
            selection = {}
        try:
            cache = live_cache.cache_health(conn)
        except Exception:
            cache = {}
        return {"status": status, "selection": selection, "cache": cache}


@st.cache_data(ttl=60)
def _prediction_recommendation(
    lat: float,
    lon: float,
    near_radius_km: float,
    search_radius_km: float,
    query_label: str | None,
) -> dict:
    with db.session(read_only=True) as conn:
        return recommendations.recommend(
            conn,
            lat=lat,
            lon=lon,
            near_radius_km=near_radius_km,
            search_radius_km=search_radius_km,
            query_label=query_label,
            source="streamlit",
            log_forecasts=False,
            log_query=False,
            resolve_outcomes=False,
            initialize_schema=False,
        )


@st.cache_data(ttl=60)
def _prediction_multi_bike_plan(
    lat: float,
    lon: float,
    k: int,
    near_radius_km: float,
    search_radius_km: float,
    plan_horizon_minutes: int,
    target_success_probability: float,
    query_label: str | None,
) -> dict:
    with db.session(read_only=True) as conn:
        return recommendations.plan_multi_bike_acquisition(
            conn,
            lat=lat,
            lon=lon,
            k=int(k),
            near_radius_km=near_radius_km,
            search_radius_km=search_radius_km,
            plan_horizon_minutes=plan_horizon_minutes,
            target_success_probability=target_success_probability,
            query_label=query_label,
            source="streamlit_multi_bike",
        )


@st.cache_data(ttl=120)
def _prediction_performance(window_hours: int) -> dict:
    with db.session(read_only=True) as conn:
        return model_eval.performance_summary(
            conn,
            window_hours=window_hours,
            resolve=False,
            initialize_schema=False,
        )


@st.cache_data(ttl=120)
def _multi_bike_performance(window_hours: int) -> dict:
    with db.session(read_only=True) as conn:
        return model_eval.multi_bike_performance_summary(
            conn,
            window_hours=window_hours,
            plan_sizes=(2, 3, 4, 5),
            initialize_schema=False,
        )


@st.cache_data(ttl=120)
def _per_horizon_performance(window_hours: int) -> dict:
    with db.session(read_only=True) as conn:
        return model_eval.per_horizon_performance_summary(
            conn, window_hours=window_hours, initialize_schema=False
        )


@st.cache_data(ttl=120)
def _empty_station_performance(window_hours: int) -> dict:
    with db.session(read_only=True) as conn:
        return model_eval.empty_station_performance_summary(
            conn, window_hours=window_hours, initialize_schema=False
        )


@st.cache_data(ttl=120)
def _count_performance(window_hours: int) -> dict:
    with db.session(read_only=True) as conn:
        return model_eval.count_performance_summary(
            conn, window_hours=window_hours, initialize_schema=False
        )


@st.cache_data(ttl=120)
def _threshold_k_performance(window_hours: int) -> dict:
    with db.session(read_only=True) as conn:
        return model_eval.threshold_k_performance_summary(
            conn,
            window_hours=window_hours,
            k_values=(1, 2, 3, 5),
            initialize_schema=False,
        )


@st.cache_data(ttl=120)
def _open_dock_performance(window_hours: int) -> dict:
    with db.session(read_only=True) as conn:
        return model_eval.open_dock_performance_summary(
            conn, window_hours=window_hours, initialize_schema=False
        )


@st.cache_data(ttl=120)
def _topk_recommendation_performance(window_hours: int) -> dict:
    with db.session(read_only=True) as conn:
        return model_eval.topk_recommendation_summary(
            conn,
            window_hours=window_hours,
            k_values=(1, 3, 5),
            initialize_schema=False,
        )


@st.cache_data(ttl=120)
def _survival_calibration(window_hours: int) -> dict:
    with db.session(read_only=True) as conn:
        return model_eval.survival_calibration_summary(
            conn, window_hours=window_hours, initialize_schema=False
        )


@st.cache_data(ttl=120)
def _model_metric_trend(model_key: str | None, days: int) -> pd.DataFrame:
    """Per-model rolling Brier/N over the last N days for the trend sparkline."""
    with db.session(read_only=True) as conn:
        if model_key:
            return conn.execute(
                """
                SELECT computed_at, brier_score, rank_loss, n
                FROM model_metrics
                WHERE group_key = 'model' AND group_value = ?
                  AND computed_at > now() - (? * INTERVAL '1 day')
                ORDER BY computed_at
                """,
                [model_key, days],
            ).df()
        return conn.execute(
            """
            SELECT computed_at, brier_score, rank_loss, n
            FROM model_metrics
            WHERE group_key = 'overall'
              AND computed_at > now() - (? * INTERVAL '1 day')
            ORDER BY computed_at
            """,
            [days],
        ).df()


@st.cache_data(ttl=120)
def _model_artifacts() -> list[dict]:
    with db.session(read_only=True) as conn:
        return model_registry.list_artifacts(conn)


@st.cache_data(ttl=60)
def _station_prediction_detail(
    station_id: str,
    user_lat: float | None,
    user_lon: float | None,
) -> tuple[pd.DataFrame, dict]:
    with db.session(read_only=True) as conn:
        candidate = conn.execute(
            """
            WITH latest AS (
              SELECT station_id, num_bikes_available, num_ebikes_available,
                     num_docks_available, last_reported, is_renting
              FROM (
                SELECT *, ROW_NUMBER() OVER (
                  PARTITION BY station_id ORDER BY last_reported DESC
                ) AS rn
                FROM station_status
                WHERE station_id = ?
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
              l.is_renting
            FROM stations s
            JOIN latest l USING (station_id)
            WHERE s.station_id = ?
            """,
            [station_id, station_id],
        ).df()
        if candidate.empty:
            return pd.DataFrame(), {}
        row = candidate.iloc[0]
        if user_lat is not None and user_lon is not None and pd.notna(row["lat"]) and pd.notna(row["lon"]):
            candidate["distance_km"] = recommendations.haversine_km(
                float(user_lat),
                float(user_lon),
                float(row["lat"]),
                float(row["lon"]),
            )
            search_radius_km = max(1.5, float(candidate["distance_km"].iloc[0]))
        else:
            candidate["distance_km"] = 0.0
            search_radius_km = 1.5
        scored, fitted = predictor.score_candidates(conn, candidate, search_radius_km=search_radius_km)
        model = {
            "model_version": fitted.model_version,
            "method": fitted.method,
            "trained_at": fitted.trained_at,
            "training_examples": fitted.n_examples,
            "training_positive": fitted.n_positive,
            "training_negative": fitted.n_negative,
            "models": fitted.summary(),
        }
        return scored, model


@st.cache_data(ttl=120)
def _station_prediction_metrics(station_id: str, window_hours: int) -> pd.DataFrame:
    with db.session(read_only=True) as conn:
        rows = conn.execute(
            """
            SELECT
              COALESCE(f.model_key, 'logistic') AS model_key,
              COALESCE(f.model_label, f.model_version) AS model_label,
              f.horizon_minutes,
              f.forecasted_at,
              f.p_has_ebike,
              o.observed_has_ebike,
              o.observed_ebikes
            FROM model_forecasts f
            JOIN model_outcomes o USING (forecast_id)
            WHERE f.station_id = ?
              AND f.forecasted_at >= now() - (? * INTERVAL '1 hour')
            """,
            [station_id, window_hours],
        ).df()
    if rows.empty:
        return pd.DataFrame()
    rows["p_has_ebike"] = rows["p_has_ebike"].astype(float).clip(0.001, 0.999)
    rows["observed"] = rows["observed_has_ebike"].astype(float)
    out = []
    for (model_key, model_label, horizon), group in rows.groupby(["model_key", "model_label", "horizon_minutes"]):
        p = group["p_has_ebike"]
        y = group["observed"]
        log_loss = -(y * p.apply(math.log) + (1 - y) * (1 - p).apply(math.log)).mean()
        out.append({
            "model_key": model_key,
            "model": model_label,
            "horizon_minutes": int(horizon),
            "n": int(len(group)),
            "brier_score": float(((p - y) ** 2).mean()),
            "log_loss": float(log_loss),
            "rank_loss": float(((p - y) ** 2).mean() + 0.05 * log_loss),
            "observed_rate": float(y.mean()),
            "mean_prediction": float(p.mean()),
        })
    return pd.DataFrame(out).sort_values(["horizon_minutes", "rank_loss"])


@st.cache_data(ttl=600)
def _trip_demand_profile(station_id: str, anchor_iso: str) -> pd.DataFrame:
    anchor = datetime.fromisoformat(anchor_iso)
    with db.session(read_only=True) as conn:
        return analytics.station_trip_demand_profile(conn, station_id, anchor=anchor)


@st.cache_data(ttl=300)
def _nearest(lat: float, lon: float, k: int) -> pd.DataFrame:
    with db.session(read_only=True) as conn:
        return analytics.nearest_stations(conn, lat, lon, k)


@st.cache_data(ttl=30)
def _latest_status(station_ids: tuple[str, ...]) -> pd.DataFrame:
    with db.session(read_only=True) as conn:
        return analytics.latest_status(conn, list(station_ids))


@st.cache_data(ttl=30)
def _stations_with_ebikes(lat: float, lon: float, radius_km: float) -> pd.DataFrame:
    with db.session(read_only=True) as conn:
        return analytics.stations_with_ebikes_nearby(
            conn, lat, lon, radius_km=radius_km, limit=40
        )


@st.cache_data(ttl=300)
def _free_bike_density(lat: float, lon: float, radius_km: float, hours: int) -> pd.DataFrame:
    with db.session(read_only=True) as conn:
        return analytics.free_bike_density(
            conn, lat, lon, radius_km=radius_km, hours=hours
        )


@st.cache_data(ttl=300)
def _free_bike_count_by_hour(hours: int) -> pd.DataFrame:
    with db.session(read_only=True) as conn:
        return analytics.free_bike_count_by_hour(conn, hours=hours)


@st.cache_data(ttl=60)
def _tile_state_for_area(lat: float, lon: float, radius_km: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    tile_ids = tile.tile_ids_in_disk(lat, lon, radius_km)
    if not tile_ids:
        return pd.DataFrame(), pd.DataFrame()
    try:
        live_bike_ids: set[str] | None = recommendations.fetch_live_free_bike_ids()
    except Exception:
        live_bike_ids = None
    with db.session(read_only=True) as conn:
        return tile_predictor.score_tiles(conn, tile_ids, live_bike_ids=live_bike_ids)


def _disabled_tile_state_for_area(lat: float, lon: float, radius_km: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    tile_ids = tile.tile_ids_in_disk(lat, lon, radius_km)
    if not tile_ids:
        return pd.DataFrame(), pd.DataFrame()
    with db.session(read_only=True) as conn:
        return disabled_predictor.score_tiles_disability(conn, tile_ids)


@st.cache_data(ttl=60)
def _free_bikes_in_box(
    lat_min: float, lat_max: float, lon_min: float, lon_max: float, hours: int
) -> pd.DataFrame:
    with db.session(read_only=True) as conn:
        return analytics.free_bikes_in_box(
            conn, lat_min, lat_max, lon_min, lon_max, hours=hours
        )


@st.cache_data(ttl=30)
def _free_bikes() -> pd.DataFrame:
    """Live free-floating ebike positions from Divvy GBFS.

    Divvy only floats ebikes (classics must be docked), and the feed has no
    battery range, so each row is just a bike_id, name, lat, lon. Excludes
    reserved/disabled bikes.
    """
    resp = requests.get(
        "https://gbfs.divvybikes.com/gbfs/en/free_bike_status.json",
        headers={"User-Agent": "divvy-observer/0.1 (personal research)"},
        timeout=10,
    )
    resp.raise_for_status()
    rows = []
    for b in resp.json().get("data", {}).get("bikes", []):
        if b.get("is_disabled") or b.get("is_reserved"):
            continue
        lat, lon = b.get("lat"), b.get("lon")
        if lat is None or lon is None:
            continue
        rows.append({
            "bike_id": b.get("bike_id"),
            "name": b.get("name"),
            "lat": float(lat),
            "lon": float(lon),
        })
    return pd.DataFrame(rows)


def _haversine_km(df: pd.DataFrame, lat: float, lon: float) -> pd.Series:
    """Vectorized haversine distance (km) from (lat, lon) to each row of df."""
    import numpy as np
    lat1 = np.radians(lat)
    lat2 = np.radians(df["lat"].to_numpy())
    dlat = lat2 - lat1
    dlon = np.radians(df["lon"].to_numpy() - lon)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return pd.Series(2 * 6371.0 * np.arcsin(np.sqrt(a)), index=df.index)


@st.cache_data(ttl=86400)
def _geocode(address: str) -> tuple[float, float, str] | None:
    """Geocode a free-form address to (lat, lon, display_name) via Nominatim.

    Returns None on no match. Raises requests.RequestException on network errors.
    Biases to Chicago when the user doesn't include a city.
    """
    query = address.strip()
    if not query:
        return None
    if "chicago" not in query.lower() and "il" not in query.lower():
        query = f"{query}, Chicago, IL"
    resp = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": query, "format": "json", "limit": 1, "addressdetails": 0},
        headers={"User-Agent": "divvy-observer/0.1 (personal research)"},
        timeout=10,
    )
    resp.raise_for_status()
    results = resp.json()
    if not results:
        return None
    r = results[0]
    return float(r["lat"]), float(r["lon"]), r.get("display_name", query)


@st.cache_data(ttl=120)
def _ebike_forecast(station_id: str, current_value: int, start, end) -> pd.DataFrame:
    with db.session(read_only=True) as conn:
        return analytics.ebike_forecast(
            conn, station_id, current_value=current_value, start=start, end=end
        )


def _format_station(row: pd.Series) -> str:
    cap = int(row.capacity) if pd.notna(row.capacity) else 0
    obs = int(row.n_obs) if pd.notna(row.n_obs) else 0
    return f"{row['name']}  ·  cap {cap}  ·  obs {obs:,}"


def _distance_label(km: float | None) -> str:
    if km is None or pd.isna(km):
        return "-"
    return f"{km * 1000:.0f} m" if km < 0.95 else f"{km:.2f} km"


def _prob_label(value: float | None, *, debug: bool = False) -> str:
    if value is None or pd.isna(value):
        return "—"
    return display_probability(float(value), debug=debug)


def _score_label(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value) * 100:.0f}/100"


def _minutes_label(minutes: float | None) -> str:
    if minutes is None or pd.isna(minutes):
        return "-"
    if minutes < 1:
        return "<1 min"
    return f"{minutes:.0f} min"


BASELINE_MODEL_KEY = "logistic"


def _skill_score(model_brier: float | None, baseline_brier: float | None) -> float | None:
    """Brier skill score vs baseline. >0 = better than baseline, <0 = worse."""
    if model_brier is None or baseline_brier is None:
        return None
    if pd.isna(model_brier) or pd.isna(baseline_brier):
        return None
    if baseline_brier <= 0:
        return None
    return 1.0 - float(model_brier) / float(baseline_brier)


def _skill_label(skill: float | None) -> str:
    if skill is None or pd.isna(skill):
        return "—"
    return f"{skill * 100:+.1f}%"


def _calibration_label(ece: float | None) -> str:
    if ece is None or pd.isna(ece):
        return "—"
    if ece <= 0.05:
        return "good"
    if ece <= 0.10:
        return "ok"
    return "off"


def _pipeline_health(
    perf: dict,
    collector_health: dict | None = None,
    system_status: dict | None = None,
) -> tuple[str, str, str]:
    """Return (status, headline, detail) for the model-pipeline health badge.

    status: 'green' | 'yellow' | 'red'
    """
    leaderboard = perf.get("model_leaderboard") or []
    total_resolved = sum(int(row.get("n") or 0) for row in leaderboard)
    last_outcome_age_min = None
    last_forecast_age_min = None
    if system_status:
        freshness = system_status.get("data_freshness") or {}
        latest_station = freshness.get("latest_station_reported")
        if latest_station:
            last_forecast_age_min = _age_minutes(pd.Timestamp(latest_station))
    if collector_health:
        last_forecast_age_min = (
            _age_minutes(collector_health.get("latest_station_reported"))
            if collector_health.get("latest_station_reported") is not None
            else last_forecast_age_min
        )

    if total_resolved == 0:
        return (
            "red",
            "Pipeline broken — no outcomes resolved in last 24h",
            "Forecasts may still be flowing, but the resolver hasn't attached observed outcomes. "
            "Run the resolve-outcomes job manually to drain the backlog.",
        )
    if total_resolved < 100:
        return (
            "yellow",
            f"Sparse signal — only {total_resolved} resolved outcomes in last 24h",
            "Metrics are noisy at this sample size. Wait for the box to bake or trigger a snapshot.",
        )
    detail_bits = [f"{total_resolved:,} resolved outcomes"]
    if last_forecast_age_min is not None and last_forecast_age_min < 60:
        detail_bits.append(f"last station tick {last_forecast_age_min:.0f}m ago")
    return (
        "green",
        f"Pipeline healthy · {total_resolved:,} resolved outcomes in last 24h",
        " · ".join(detail_bits),
    )


def _render_health_badge(status: str, headline: str, detail: str) -> None:
    color_map = {"green": "🟢", "yellow": "🟡", "red": "🔴"}
    icon = color_map.get(status, "⚪")
    if status == "green":
        st.success(f"{icon} **{headline}**\n\n{detail}")
    elif status == "yellow":
        st.warning(f"{icon} **{headline}**\n\n{detail}")
    else:
        st.error(f"{icon} **{headline}**\n\n{detail}")


def _render_leaderboard_block(
    leaderboard: list[dict],
    *,
    active_key: str | None,
    best_key: str | None,
    window_hours: int,
    key_prefix: str,
    scope_caption: str | None = None,
) -> None:
    """Render hero metrics + slim/full leaderboard for a leaderboard list.

    Used by both the global "All forecasts" tab and the per-k multi-bike tabs.
    No-op for empty leaderboards — caller should handle that case.
    """
    if not leaderboard:
        return

    leaderboard_df = pd.DataFrame(leaderboard)
    baseline_row = next(
        (row for row in leaderboard if str(row.get("model_key")) == BASELINE_MODEL_KEY),
        None,
    )
    baseline_brier = baseline_row.get("brier_score") if baseline_row else None
    leaderboard_df["skill_score"] = leaderboard_df["brier_score"].apply(
        lambda b: _skill_score(b, baseline_brier)
    )
    leaderboard_df["calibration_label"] = leaderboard_df.get(
        "ece", pd.Series([None] * len(leaderboard_df))
    ).apply(_calibration_label)

    active_row = leaderboard_df[leaderboard_df["model_key"] == active_key]
    best_row = leaderboard_df[leaderboard_df["model_key"] == best_key]
    # If the supplied best_key is not in this leaderboard slice (e.g. multi-bike
    # subset where the global best didn't show up), fall back to the slice's
    # top-ranked model so the hero still shows something useful.
    if best_row.empty and not leaderboard_df.empty:
        slice_best = leaderboard_df.sort_values(
            by="skill_score", ascending=False, na_position="last"
        ).iloc[0]
        slice_best_key = str(slice_best.get("model_key"))
        best_row = leaderboard_df[leaderboard_df["model_key"] == slice_best_key]
        best_key = slice_best_key
    active_label = (active_row.iloc[0]["model_label"] if not active_row.empty else (active_key or "—"))
    best_label = (best_row.iloc[0]["model_label"] if not best_row.empty else (best_key or "—"))
    active_skill = active_row.iloc[0]["skill_score"] if not active_row.empty else None
    best_skill = best_row.iloc[0]["skill_score"] if not best_row.empty else None
    active_n = int(active_row.iloc[0]["n"]) if not active_row.empty else 0

    cols = st.columns(3)
    cols[0].metric(
        "Active model (driving recommendations)",
        active_label,
        _skill_label(active_skill),
        help="Skill score vs the logistic baseline. Positive = better than baseline.",
    )
    cols[0].caption(f"{active_n:,} resolved · {window_hours}h window")

    cols[1].metric(
        "Best evaluated model",
        best_label,
        _skill_label(best_skill),
        help="Lowest rank loss in this slice.",
    )
    if best_key and not best_row.empty:
        cols[1].caption(f"{int(best_row.iloc[0]['n']):,} resolved")

    if active_key and best_key and active_key == best_key:
        cols[2].metric("Active = Best?", "✅ yes")
        cols[2].caption("Active driver is the leader.")
    elif active_key and best_key:
        delta = None
        if active_skill is not None and best_skill is not None:
            delta = (best_skill - active_skill) * 100
        cols[2].metric(
            "Active = Best?",
            "❌ no",
            f"+{delta:.1f}pp better available" if delta is not None else None,
            delta_color="inverse",
        )
        cols[2].caption(f"Best candidate: {best_label}")
    else:
        cols[2].metric("Active = Best?", "—")
        cols[2].caption("Awaiting more resolved outcomes.")

    if scope_caption:
        st.caption(scope_caption)

    show_all = st.toggle(
        "Show all metrics",
        value=False,
        key=f"{key_prefix}_show_all",
        help="Switch from the simplified view to the full per-model metric table.",
    )

    leaderboard_df = leaderboard_df.copy()
    leaderboard_df["model_display"] = leaderboard_df.apply(
        lambda r: (
            ("🟢 " if str(r.get("model_key")) == active_key else "")
            + ("🥇 " if str(r.get("model_key")) == best_key and str(r.get("model_key")) != active_key else "")
            + str(r.get("model_label") or r.get("model_key") or "?")
        ),
        axis=1,
    )
    leaderboard_df = leaderboard_df.sort_values(
        by="skill_score", ascending=False, na_position="last"
    )

    if not show_all:
        slim_cols = [
            "rank",
            "model_display",
            "n",
            "skill_score",
            "calibration_label",
            "recommended_hit_rate",
        ]
        slim = leaderboard_df[[c for c in slim_cols if c in leaderboard_df.columns]].copy()
        if "recommended_hit_rate" in slim.columns:
            slim["recommended_hit_rate"] = slim["recommended_hit_rate"].map(_prob_label)
        if "skill_score" in slim.columns:
            slim["skill_score"] = slim["skill_score"].map(_skill_label)
        st.dataframe(
            slim,
            hide_index=True,
            column_config={
                "rank": "Rank",
                "model_display": "Model",
                "n": st.column_config.NumberColumn("Resolved", format="%d"),
                "skill_score": st.column_config.Column(
                    "Skill vs baseline",
                    help="Brier skill: 1 - model/baseline. Positive = better than the logistic baseline.",
                ),
                "calibration_label": "Calibration",
                "recommended_hit_rate": "Top-pick hit rate",
            },
        )
        st.caption(
            f"Skill score uses **{BASELINE_MODEL_KEY}** as the baseline. "
            "Calibration: good (ECE ≤ 0.05), ok (≤ 0.10), off (> 0.10). "
            "Top-pick hit rate = how often the model's #1 recommendation actually had an eBike."
        )
    else:
        full_cols = [
            "rank",
            "model_display",
            "n",
            "skill_score",
            "rank_loss",
            "brier_score",
            "log_loss",
            "ece",
            "recommended_hit_rate",
            "distance_adjusted_regret",
            "decision_rank_loss",
            "count_log_loss",
            "crps",
            "capacity_violation_rate",
            "observed_rate",
            "mean_prediction",
        ]
        full = leaderboard_df[[c for c in full_cols if c in leaderboard_df.columns]].copy()
        for column in [
            "recommended_hit_rate",
            "capacity_violation_rate",
            "observed_rate",
            "mean_prediction",
        ]:
            if column in full.columns:
                full[column] = full[column].map(_prob_label)
        if "skill_score" in full.columns:
            full["skill_score"] = full["skill_score"].map(_skill_label)
        st.dataframe(
            full,
            hide_index=True,
            column_config={
                "rank": "Rank",
                "model_display": "Model",
                "n": "Resolved",
                "skill_score": st.column_config.Column("Skill vs baseline"),
                "rank_loss": st.column_config.NumberColumn("Rank loss", format="%.3f"),
                "brier_score": st.column_config.NumberColumn("Brier", format="%.3f"),
                "log_loss": st.column_config.NumberColumn("Log loss", format="%.3f"),
                "ece": st.column_config.NumberColumn("ECE", format="%.3f"),
                "recommended_hit_rate": "Top-pick hit",
                "distance_adjusted_regret": st.column_config.NumberColumn("Regret", format="%.3f"),
                "decision_rank_loss": st.column_config.NumberColumn("Decision loss", format="%.3f"),
                "count_log_loss": st.column_config.NumberColumn("Count NLL", format="%.3f"),
                "crps": st.column_config.NumberColumn("CRPS", format="%.3f"),
                "capacity_violation_rate": "Cap. violation",
                "observed_rate": "Observed hit",
                "mean_prediction": "Mean P",
            },
        )


def _render_active_model_charts(perf: dict, active_key: str | None, active_label: str) -> None:
    """Calibration scatter + 7-day rank-loss trend for the active model."""
    chart_cols = st.columns(2)

    calibration = perf.get("calibration") or []
    with chart_cols[0]:
        st.markdown(f"**Calibration — active model ({active_label})**")
        if calibration:
            cal_df = pd.DataFrame(calibration)
            ref = pd.DataFrame({"x": [0, 1], "y": [0, 1]})
            chart = alt.Chart(cal_df).mark_circle(size=120, opacity=0.8).encode(
                x=alt.X("mean_prediction:Q", title="Predicted probability", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("observed_hit_rate:Q", title="Observed hit rate", scale=alt.Scale(domain=[0, 1])),
                size=alt.Size("n:Q", title="N", scale=alt.Scale(range=[40, 400])),
                tooltip=[
                    alt.Tooltip("probability_band:N", title="Band"),
                    alt.Tooltip("n:Q", title="N"),
                    alt.Tooltip("mean_prediction:Q", format=".1%", title="Predicted"),
                    alt.Tooltip("observed_hit_rate:Q", format=".1%", title="Observed"),
                ],
            )
            line = alt.Chart(ref).mark_line(strokeDash=[4, 4], color="#888").encode(x="x:Q", y="y:Q")
            st.altair_chart((line + chart).properties(height=200), use_container_width=True)
            st.caption("Dots above the line = model under-predicts; below = over-predicts. Size = sample count.")
        else:
            st.info("Calibration data unavailable for this window.")

    with chart_cols[1]:
        st.markdown("**7-day rank-loss trend — active model**")
        if active_key:
            trend = _model_metric_trend(active_key, days=7)
        else:
            trend = pd.DataFrame()
        if not trend.empty and trend["rank_loss"].notna().any():
            trend_chart = alt.Chart(trend).mark_line(point=True).encode(
                x=alt.X("computed_at:T", title=None),
                y=alt.Y("rank_loss:Q", title="Rank loss", scale=alt.Scale(zero=False)),
                tooltip=[
                    alt.Tooltip("computed_at:T"),
                    alt.Tooltip("rank_loss:Q", format=".3f"),
                    alt.Tooltip("brier_score:Q", format=".3f"),
                    alt.Tooltip("n:Q"),
                ],
            )
            st.altair_chart(trend_chart.properties(height=200), use_container_width=True)
            st.caption("Lower is better. Watch for upward drift = model degrading vs recent days.")
        else:
            st.info("Not enough snapshots yet for a 7-day trend.")


MULTI_BIKE_PLAN_SIZES = (2, 3, 4, 5)


def _render_by_horizon_tab(window_hours: int, active_key: str | None, best_key: str | None) -> None:
    """How does each model degrade as the prediction horizon stretches?

    Shows a Brier-by-horizon line chart (color = model) plus a leaderboard
    aggregated across horizons. The hypothesis worth checking here is whether
    SOTA models close the gap on simple models when autocorrelation weakens
    (i.e. at 30/45/60/90 min) — if they do, this view will surface it.
    """
    try:
        payload = _per_horizon_performance(window_hours)
    except Exception as exc:
        st.error(f"per-horizon query failed: {exc}")
        return
    horizons = payload.get("horizons") or []
    by_horizon = payload.get("by_horizon") or {}
    if not horizons:
        st.info(f"No resolved forecasts in the last {window_hours}h.")
        return

    rows: list[dict] = []
    for h in horizons:
        for r in by_horizon.get(h, {}).get("model_leaderboard", []):
            rows.append({
                "horizon_minutes": int(h),
                "model_key": r.get("model_key"),
                "model_label": r.get("model_label"),
                "n": r.get("n"),
                "brier_score": r.get("brier_score"),
                "rank_loss": r.get("rank_loss"),
            })
    long_df = pd.DataFrame(rows)
    if long_df.empty:
        st.info(f"No resolved forecasts in the last {window_hours}h.")
        return

    st.markdown(
        f"**Brier by horizon — last {window_hours}h.** "
        f"Horizons currently emitted: {', '.join(f'{h}m' for h in horizons)}. "
        "Lower is better. A flat line = horizon-robust model; a steeply rising "
        "line = a model that loses skill quickly as the future gets fuzzier."
    )
    chart = (
        alt.Chart(long_df.dropna(subset=["brier_score"]))
        .mark_line(point=True)
        .encode(
            x=alt.X("horizon_minutes:Q", title="Horizon (minutes)", scale=alt.Scale(zero=False)),
            y=alt.Y("brier_score:Q", title="Brier score (lower is better)", scale=alt.Scale(zero=False)),
            color=alt.Color("model_label:N", title="Model"),
            tooltip=[
                alt.Tooltip("model_label:N", title="Model"),
                alt.Tooltip("horizon_minutes:Q", title="Horizon"),
                alt.Tooltip("brier_score:Q", format=".4f", title="Brier"),
                alt.Tooltip("n:Q", title="N"),
            ],
        )
        .properties(height=280)
    )
    st.altair_chart(chart, use_container_width=True)

    available_set = set(int(h) for h in horizons)
    long_horizons = [30, 45, 60, 90]
    missing_long = [h for h in long_horizons if h not in available_set]
    if missing_long:
        st.caption(
            f"Horizons {missing_long} have no resolved forecasts yet. "
            "After restarting the stack with the extended `predictor.HORIZONS`, "
            "the long-horizon points will start appearing within ~10 min for 30m, "
            "~45 min for 45m, ~70 min for 60m, and ~100 min for 90m forecasts."
        )

    avg = (
        long_df.groupby(["model_key", "model_label"], as_index=False)
        .agg(
            avg_brier=("brier_score", "mean"),
            avg_rank_loss=("rank_loss", "mean"),
            total_n=("n", "sum"),
        )
        .sort_values("avg_brier", na_position="last")
    )
    if avg.empty:
        return
    baseline = avg[avg["model_key"] == BASELINE_MODEL_KEY]
    baseline_brier = baseline.iloc[0]["avg_brier"] if not baseline.empty else None
    avg["skill_score"] = avg["avg_brier"].apply(lambda b: _skill_score(b, baseline_brier))
    avg["model_display"] = avg.apply(
        lambda r: (
            ("🟢 " if str(r.get("model_key")) == active_key else "")
            + ("🥇 " if str(r.get("model_key")) == best_key and str(r.get("model_key")) != active_key else "")
            + str(r.get("model_label") or r.get("model_key"))
        ),
        axis=1,
    )
    avg["skill_score"] = avg["skill_score"].map(_skill_label)
    st.markdown("**Average across horizons**")
    st.dataframe(
        avg[["model_display", "total_n", "avg_brier", "avg_rank_loss", "skill_score"]],
        hide_index=True,
        column_config={
            "model_display": "Model",
            "total_n": st.column_config.NumberColumn("Total resolved", format="%d"),
            "avg_brier": st.column_config.NumberColumn("Avg Brier", format="%.4f"),
            "avg_rank_loss": st.column_config.NumberColumn("Avg rank loss", format="%.4f"),
            "skill_score": st.column_config.Column(
                "Skill vs baseline",
                help="Mean Brier across horizons compared to the logistic baseline's mean.",
            ),
        },
    )


def _render_empty_station_tab(
    window_hours: int, active_key: str | None, best_key: str | None
) -> None:
    """Performance restricted to forecasts at currently-empty stations.

    At an empty station, P(has eBike at horizon) is essentially "will a bike
    arrive in the next N minutes?" — autocorrelation gives no signal. Models
    that explicitly represent arrival rates (empirical, ZINB-based,
    graph-flow) have a chance to differentiate themselves here.
    """
    try:
        payload = _empty_station_performance(window_hours)
    except Exception as exc:
        st.error(f"empty-station query failed: {exc}")
        return
    rows = payload.get("model_leaderboard") or []
    n_total = int(payload.get("n_total") or 0)
    appearance_rate = payload.get("appearance_rate")
    if not rows or n_total == 0:
        st.info(
            f"No resolved forecasts at currently-empty stations in the last {window_hours}h. "
            "This will populate as the collector + automation drain the queue."
        )
        return
    headline_bits = [f"{n_total:,} resolved forecasts at empty stations"]
    if appearance_rate is not None:
        headline_bits.append(f"appearance rate {appearance_rate * 100:.1f}%")
    st.markdown(
        f"**At currently-empty stations** ({' · '.join(headline_bits)} in last {window_hours}h). "
        "This isolates the *arrival* problem — current bike count = 0, so the question is "
        "whether the model can predict that a bike will appear. Autocorrelation-driven models "
        "lose their easy advantage; models that represent arrival dynamics have a chance to win."
    )
    _render_leaderboard_block(
        rows,
        active_key=active_key,
        best_key=best_key,
        window_hours=window_hours,
        key_prefix="model_perf_empty",
    )


def _render_count_tab(window_hours: int, active_key: str | None) -> None:
    """Count prediction leaderboard — NLL, CRPS, MAE on E[count].

    Models that don't emit a count PMF (RF/GB/logistic/empirical/stg_ncde)
    are still listed but with NLL/CRPS = '—'. They can still be ranked on
    MAE if they emit expected_ebikes.
    """
    try:
        payload = _count_performance(window_hours)
    except Exception as exc:
        st.error(f"count query failed: {exc}")
        return
    rows = payload.get("model_leaderboard") or []
    if not rows:
        st.info(f"No resolved forecasts in the last {window_hours}h.")
        return
    df = pd.DataFrame(rows)
    df["model_display"] = df.apply(
        lambda r: ("🟢 " if str(r.get("model_key")) == active_key else "")
        + str(r.get("model_label") or r.get("model_key")),
        axis=1,
    )
    st.markdown(
        f"**Count prediction — last {window_hours}h.** "
        "Headline metric is **count NLL** (lower = better) — the negative log-likelihood "
        "of the *integer* observed count under the model's PMF. CRPS scores the full "
        "distribution. MAE on E[count] gives a more legible point-prediction error."
    )
    avg_obs = float(df["mean_observed_ebikes"].mean()) if df["mean_observed_ebikes"].notna().any() else None
    if avg_obs is not None:
        st.caption(
            f"Mean observed eBike count across resolved forecasts: **{avg_obs:.2f}**. "
            "Models that don't emit a PMF (logistic / RF / gradient_boosting / empirical / "
            "stg_ncde_inventory) show — for NLL/CRPS but can still be scored on MAE if they "
            "emit expected_ebikes."
        )
    display_cols = [
        "rank",
        "model_display",
        "n",
        "n_with_pmf",
        "count_nll",
        "crps",
        "mae_expected",
        "rmse_expected",
    ]
    show = df[display_cols].copy()
    st.dataframe(
        show,
        hide_index=True,
        use_container_width=True,
        column_config={
            "rank": "Rank",
            "model_display": "Model",
            "n": st.column_config.NumberColumn("Resolved", format="%d"),
            "n_with_pmf": st.column_config.NumberColumn("With PMF", format="%d"),
            "count_nll": st.column_config.NumberColumn("Count NLL", format="%.3f"),
            "crps": st.column_config.NumberColumn("CRPS", format="%.3f"),
            "mae_expected": st.column_config.NumberColumn("MAE (E[count])", format="%.3f"),
            "rmse_expected": st.column_config.NumberColumn("RMSE", format="%.3f"),
        },
    )


def _render_threshold_k_tab(window_hours: int, active_key: str | None) -> None:
    """Brier score for P(observed_ebikes >= k) for k ∈ {1, 2, 3, 5}.

    Lets you see how predictive each model is at different decision thresholds
    a planner might care about (single bike, multi-bike, surge-fleet, etc.).
    """
    try:
        payload = _threshold_k_performance(window_hours)
    except Exception as exc:
        st.error(f"threshold-k query failed: {exc}")
        return
    by_k = payload.get("by_k") or {}
    k_values = payload.get("k_values") or []
    if not by_k:
        st.info(
            f"No PMF-emitting forecasts in last {window_hours}h. Threshold-K needs models "
            "that ship a count distribution (cc_nissm, tft_inventory, inventory_world, "
            "macflow_nissm_lite, dg_nissm)."
        )
        return

    # Long dataframe for chart
    long_rows: list[dict] = []
    for k in k_values:
        for r in by_k.get(k, {}).get("model_leaderboard", []):
            long_rows.append({
                "k": int(k),
                "model_key": r.get("model_key"),
                "model_label": r.get("model_label"),
                "brier": r.get("brier_score"),
                "n": r.get("n"),
                "observed_rate": r.get("observed_rate"),
            })
    long_df = pd.DataFrame(long_rows)
    if long_df.empty:
        st.info("No threshold-k data yet.")
        return

    st.markdown(
        f"**P(observed_ebikes ≥ k) Brier — last {window_hours}h.** "
        "Different decision thresholds for different planner needs: k=1 is the basic "
        '"any bike?" question, k=2..3 is a small group, k=5 is a fleet ask. '
        "Lower Brier = sharper, better-calibrated probability at that threshold."
    )
    chart = (
        alt.Chart(long_df.dropna(subset=["brier"]))
        .mark_line(point=True)
        .encode(
            x=alt.X("k:O", title="Threshold k"),
            y=alt.Y("brier:Q", title="Brier (lower is better)", scale=alt.Scale(zero=False)),
            color=alt.Color("model_label:N", title="Model"),
            tooltip=[
                alt.Tooltip("model_label:N", title="Model"),
                alt.Tooltip("k:O", title="k"),
                alt.Tooltip("brier:Q", format=".4f", title="Brier"),
                alt.Tooltip("observed_rate:Q", format=".3f", title="Base rate"),
                alt.Tooltip("n:Q", title="N"),
            ],
        )
        .properties(height=260)
    )
    st.altair_chart(chart, use_container_width=True)
    base_rates = (
        long_df.groupby("k")["observed_rate"].first().to_dict()
        if "observed_rate" in long_df.columns
        else {}
    )
    if base_rates:
        bits = " · ".join(f"k={k}: {rate * 100:.1f}%" for k, rate in sorted(base_rates.items()))
        st.caption(f"Empirical base rates: {bits}")


def _render_open_dock_tab(window_hours: int, active_key: str | None) -> None:
    """Per-model leaderboard for the parking-side dual of has-eBike."""
    try:
        payload = _open_dock_performance(window_hours)
    except Exception as exc:
        st.error(f"open-dock query failed: {exc}")
        return
    rows = payload.get("model_leaderboard") or []
    n_total = int(payload.get("n_total") or 0)
    open_rate = payload.get("open_rate")
    if not rows or n_total == 0:
        st.info(
            f"No models emitted p_capacity_violation in the last {window_hours}h. "
            "This metric is only populated by SOTA / inventory-world family models."
        )
        return
    headline_bits = [f"{n_total:,} resolved forecasts with p_capacity_violation"]
    if open_rate is not None:
        headline_bits.append(f"open-dock rate {open_rate * 100:.1f}%")
    st.markdown(
        f"**Open dock prediction — last {window_hours}h** ({' · '.join(headline_bits)}). "
        "Uses **(1 − p_capacity_violation)** as the predicted probability that a returning "
        "rider will find an open dock at horizon t. Target = (observed_docks > 0). This is "
        "a proxy for the eventual first-class p_has_open_dock target that's flagged as a "
        "TODO on the cdg_nmip side; the dual of the existing P(has eBike) prediction."
    )
    df = pd.DataFrame(rows)
    df["model_display"] = df.apply(
        lambda r: ("🟢 " if str(r.get("model_key")) == active_key else "")
        + str(r.get("model_label") or r.get("model_key")),
        axis=1,
    )
    df["mean_prediction"] = df["mean_prediction"].map(_prob_label)
    df["observed_rate"] = df["observed_rate"].map(_prob_label)
    st.dataframe(
        df[["rank", "model_display", "n", "brier_score", "log_loss", "mean_prediction", "observed_rate"]],
        hide_index=True,
        use_container_width=True,
        column_config={
            "rank": "Rank",
            "model_display": "Model",
            "n": st.column_config.NumberColumn("Resolved", format="%d"),
            "brier_score": st.column_config.NumberColumn("Brier", format="%.4f"),
            "log_loss": st.column_config.NumberColumn("Log loss", format="%.3f"),
            "mean_prediction": "Mean P(open)",
            "observed_rate": "Observed P(open)",
        },
    )
    st.caption(
        "When the open-dock rate is very high (most stations have docks most of the time), "
        "differences in Brier between models are small in absolute terms but still proxy for "
        "the harder cases (near-full stations) where the prediction matters most."
    )


def _render_topk_recommendation_tab(window_hours: int, active_key: str | None) -> None:
    """Top-K hit-rate over real recommendation requests."""
    try:
        payload = _topk_recommendation_performance(window_hours)
    except Exception as exc:
        st.error(f"top-k query failed: {exc}")
        return
    rows = payload.get("model_leaderboard") or []
    k_values = payload.get("k_values") or []
    if not rows:
        st.info(
            f"No resolvable recommendation requests in the last {window_hours}h. "
            "This benchmark needs `source LIKE 'api%'` requests (real recommend / "
            "multi_bike_plan calls) with resolved outcomes."
        )
        return
    n_req = max(int(r.get("n_requests") or 0) for r in rows)
    st.markdown(
        f"**Top-K recommendation accuracy — last {window_hours}h.** "
        "For each request, the *actually-best* station is the one that ended up with "
        "the most observed eBikes at the 10-minute horizon. A model gets credit if its "
        "top-K recommended_rank picks include that best station. This measures decision "
        "quality, not just probability quality."
    )
    if n_req < 10:
        st.warning(
            f"Thin sample — only {n_req} resolvable request{'s' if n_req != 1 else ''} "
            "in this window. Hit rates will jump around until traffic accumulates. "
            "Trigger requests to seed: `curl -X POST .../api/v1/recommendations` "
            "or `.../api/v1/multi_bike_plan`."
        )
    df = pd.DataFrame(rows)
    df["model_display"] = df.apply(
        lambda r: ("🟢 " if str(r.get("model_key")) == active_key else "")
        + str(r.get("model_label") or r.get("model_key")),
        axis=1,
    )
    cols = ["rank", "model_display", "n_requests"] + [f"top{k}_hit_rate" for k in k_values]
    rename = {f"top{k}_hit_rate": f"Top-{k}" for k in k_values}
    show = df[cols].copy()
    for k in k_values:
        show[f"top{k}_hit_rate"] = show[f"top{k}_hit_rate"].map(_prob_label)
    show = show.rename(columns=rename)
    show.rename(columns={"rank": "Rank", "model_display": "Model", "n_requests": "Requests"}, inplace=True)
    st.dataframe(show, hide_index=True, use_container_width=True)


def _render_survival_tab(window_hours: int, active_key: str | None) -> None:
    """Empty-station survival curves: predicted vs observed P(still empty at h)."""
    try:
        payload = _survival_calibration(window_hours)
    except Exception as exc:
        st.error(f"survival query failed: {exc}")
        return
    rows = payload.get("by_horizon_model") or []
    horizons = payload.get("horizons") or []
    if not rows or not horizons:
        st.info(f"No empty-station forecasts in the last {window_hours}h.")
        return
    st.markdown(
        f"**Empty-station survival — last {window_hours}h.** "
        "For forecasts where current_ebikes == 0, the *survival probability* is "
        "P(still empty at horizon h) = 1 − P(has eBike at h). Solid lines = each "
        "model's predicted average. Dashed black line = observed reality. "
        "Lines that hug the black line are well-calibrated arrival models."
    )
    df = pd.DataFrame(rows)

    # Build chart: predicted (one line per model) + observed (single dashed black line)
    predicted = df[["model_label", "horizon_minutes", "predicted_still_empty"]].rename(
        columns={"predicted_still_empty": "value"}
    )
    predicted["series"] = predicted["model_label"]
    observed_per_h = (
        df.groupby("horizon_minutes")["observed_still_empty"].mean().reset_index()
    )
    observed_per_h["series"] = "OBSERVED"
    observed_per_h = observed_per_h.rename(columns={"observed_still_empty": "value"})

    pred_chart = (
        alt.Chart(predicted)
        .mark_line(point=True)
        .encode(
            x=alt.X("horizon_minutes:Q", title="Horizon (min)"),
            y=alt.Y("value:Q", title="P(still empty)", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("series:N", title="Model"),
            tooltip=[
                alt.Tooltip("series:N", title="Model"),
                alt.Tooltip("horizon_minutes:Q", title="Horizon"),
                alt.Tooltip("value:Q", format=".3f"),
            ],
        )
    )
    obs_chart = (
        alt.Chart(observed_per_h)
        .mark_line(strokeDash=[6, 4], color="black", size=3)
        .encode(
            x=alt.X("horizon_minutes:Q"),
            y=alt.Y("value:Q"),
            tooltip=[
                alt.Tooltip("series:N", title=" "),
                alt.Tooltip("horizon_minutes:Q", title="Horizon"),
                alt.Tooltip("value:Q", format=".3f", title="Observed"),
            ],
        )
    )
    st.altair_chart((pred_chart + obs_chart).properties(height=300), use_container_width=True)

    st.markdown("**Per-horizon Brier (lower is better)**")
    pivot = df.pivot_table(
        index=["model_label"],
        columns="horizon_minutes",
        values="brier",
        aggfunc="mean",
    ).reset_index()
    pivot.columns = [str(c) if not isinstance(c, str) else c for c in pivot.columns]
    rename = {str(h): f"h={h}m" for h in horizons}
    pivot = pivot.rename(columns=rename)
    st.dataframe(pivot, hide_index=True, use_container_width=True)
    st.caption(
        "Models that systematically predict survival far below the observed line (e.g. "
        "predicting bikes will arrive when they don't) are over-confident on arrivals; "
        "those above the line are too pessimistic."
    )


def _model_performance_panel(perf: dict) -> None:
    """Health-first model performance section.

    Top-down: pipeline-health badge → tabs ("All forecasts" | k=2..5).
    The "All forecasts" tab keeps the global hero + slim leaderboard +
    calibration scatter + 7-day trend. Each k-tab evaluates models on the
    candidate forecasts emitted under multi-bike-plan requests of that size.
    """
    leaderboard = perf.get("model_leaderboard") or []
    try:
        collector = _collector_health()
    except Exception:
        collector = {}
    try:
        sys_status = _system_status_payload().get("status", {})
    except Exception:
        sys_status = {}
    status, headline, detail = _pipeline_health(perf, collector, sys_status)

    st.markdown("### Model performance")
    _render_health_badge(status, headline, detail)
    if not leaderboard:
        return

    active_key = (perf.get("active_model") or {}).get("model_key")
    best_key = (perf.get("best_current_model") or {}).get("model_key")
    window_hours = int(perf.get("window_hours") or 24)

    try:
        multi = _multi_bike_performance(window_hours)
    except Exception as exc:
        multi = {"by_plan_size": {}, "_error": str(exc)}

    tab_labels = (
        ["All forecasts"]
        + [f"k={k} plans" for k in MULTI_BIKE_PLAN_SIZES]
        + ["By horizon", "Empty stations"]
        + ["Count", "P(≥k)", "Open dock", "Top-K rec.", "Survival"]
    )
    tabs = st.tabs(tab_labels)

    with tabs[0]:
        _render_leaderboard_block(
            leaderboard,
            active_key=active_key,
            best_key=best_key,
            window_hours=window_hours,
            key_prefix="model_perf_all",
        )
        # Use the active model's row label for the calibration heading, falling
        # back to the active key if missing.
        active_label = next(
            (str(r.get("model_label") or r.get("model_key") or active_key)
             for r in leaderboard if str(r.get("model_key")) == active_key),
            active_key or "—",
        )
        _render_active_model_charts(perf, active_key, active_label)

    by_plan_size = (multi or {}).get("by_plan_size") or {}
    multi_tab_count = len(MULTI_BIKE_PLAN_SIZES)
    for tab, k in zip(tabs[1:1 + multi_tab_count], MULTI_BIKE_PLAN_SIZES):
        with tab:
            payload = by_plan_size.get(k) or {"n_requests": 0, "model_leaderboard": []}
            n_req = int(payload.get("n_requests") or 0)
            rows = payload.get("model_leaderboard") or []
            if not rows or n_req == 0:
                st.info(
                    f"No multi-bike plans of size **k={k}** in the last {window_hours}h. "
                    f"Trigger one to seed data:\n\n"
                    f"```bash\ncurl -X POST http://127.0.0.1:8000/api/v1/multi_bike_plan \\\n"
                    f"  -H 'Content-Type: application/json' \\\n"
                    f"  -d '{{\"lat\": 41.88, \"lon\": -87.63, \"k\": {k}}}'\n```"
                )
                continue
            scope_caption = (
                f"Evaluating **all candidate forecasts** logged for {n_req:,} multi-bike "
                f"plan{'s' if n_req != 1 else ''} of size k={k} in the last {window_hours}h. "
                "Each model is scored on the same candidate pool, so the comparison is apples-to-apples."
            )
            _render_leaderboard_block(
                rows,
                active_key=active_key,
                best_key=best_key,
                window_hours=window_hours,
                key_prefix=f"model_perf_k{k}",
                scope_caption=scope_caption,
            )

    by_horizon_tab = tabs[1 + multi_tab_count]
    empty_station_tab = tabs[2 + multi_tab_count]
    count_tab = tabs[3 + multi_tab_count]
    threshold_tab = tabs[4 + multi_tab_count]
    open_dock_tab = tabs[5 + multi_tab_count]
    topk_tab = tabs[6 + multi_tab_count]
    survival_tab = tabs[7 + multi_tab_count]
    with by_horizon_tab:
        _render_by_horizon_tab(window_hours, active_key, best_key)
    with empty_station_tab:
        _render_empty_station_tab(window_hours, active_key, best_key)
    with count_tab:
        _render_count_tab(window_hours, active_key)
    with threshold_tab:
        _render_threshold_k_tab(window_hours, active_key)
    with open_dock_tab:
        _render_open_dock_tab(window_hours, active_key)
    with topk_tab:
        _render_topk_recommendation_tab(window_hours, active_key)
    with survival_tab:
        _render_survival_tab(window_hours, active_key)


def _model_status_banner(perf: dict, model_payload: dict) -> None:
    active_key = model_payload.get("active_model_key") or (perf.get("active_model") or {}).get("model_key")
    active_label = (
        model_payload.get("active_model_display_label")
        or model_payload.get("active_model_label")
        or predictor.MODEL_SPECS.get(active_key or "", {}).get("label")
        or active_key
        or "-"
    )
    source = model_payload.get("active_model_source") or (perf.get("active_model") or {}).get("active_model_source") or "-"
    best = perf.get("best_current_model") or {}
    models_by_key = {
        str(item.get("model_key")): item
        for item in (model_payload.get("models") or [])
        if item.get("model_key") is not None
    }
    best_evaluated_key = model_payload.get("best_evaluated_model_key") or best.get("model_key")
    best_evaluated_label = (
        model_payload.get("best_evaluated_model_label")
        or best.get("model_label")
        or best.get("best_model_label")
        or predictor.MODEL_SPECS.get(best_evaluated_key or "", {}).get("label")
        or best_evaluated_key
        or "-"
    )
    best_usable_key = model_payload.get("best_usable_model_key")
    best_usable_label = (
        model_payload.get("best_usable_model_label")
        or predictor.MODEL_SPECS.get(best_usable_key or "", {}).get("label")
        or best_usable_key
        or "-"
    )
    best_trained_sota_key = model_payload.get("best_trained_sota_model_key")
    best_trained_sota_label = (
        predictor.MODEL_SPECS.get(best_trained_sota_key or "", {}).get("label")
        or best_trained_sota_key
        or "-"
    )
    active_warning = model_payload.get("active_model_warning") or model_payload.get("model_warning")
    best_evaluated_summary = models_by_key.get(str(best_evaluated_key), {}) if best_evaluated_key else {}
    best_evaluated_usable = bool(best_evaluated_summary.get("usable")) if best_evaluated_summary else None
    with st.container(border=True):
        cols = st.columns(4)
        cols[0].metric("Active prediction driver", active_label)
        cols[0].caption(
            f"key: {active_key or '-'} · source: {source} · "
            f"artifact: {model_payload.get('active_artifact_id') or 'none'} · "
            f"method: {model_payload.get('method') or '-'}"
        )
        metric_name = best.get("metric") or model_payload.get("selection_metric") or "decision_rank_loss"
        cols[1].metric("Best evaluated model", best_evaluated_label)
        cols[1].caption(
            f"{metric_name} · {int(best.get('n') or 0):,} resolved · "
            f"{perf.get('window_hours', model_payload.get('selection_window_hours', 24))}h"
        )
        cols[2].metric("Best usable model", best_usable_label)
        cols[2].caption(f"key: {best_usable_key or '-'}")
        cols[3].metric("Best trained SOTA", best_trained_sota_label)
        cols[3].caption(f"key: {best_trained_sota_key or '-'}")
        if active_warning:
            st.warning("Probabilities are provisional because no trained SOTA artifact is registered yet.")
        if best_evaluated_key and best_evaluated_usable is False:
            st.caption(f"Best evaluated historical model: {best_evaluated_key}. Current artifact unavailable.")


def _age_minutes(ts) -> float | None:
    if ts is None or pd.isna(ts):
        return None
    stamp = pd.Timestamp(ts)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    return (pd.Timestamp.now(tz="UTC") - stamp).total_seconds() / 60.0


def _heatmap(df: pd.DataFrame, value_col: str, title: str, fmt: str) -> alt.Chart:
    if df.empty:
        return alt.Chart(pd.DataFrame({"x": [], "y": [], "v": []})).mark_text(text="(no data)")
    plot_df = df.copy()
    plot_df["dow_label"] = plot_df["dow"].map(lambda d: DOW_LABELS[int(d)])
    scale = (
        alt.Scale(scheme="viridis", domain=[0, 1])
        if value_col.startswith("p_")
        else alt.Scale(scheme="viridis")
    )
    return (
        alt.Chart(plot_df)
        .mark_rect()
        .encode(
            x=alt.X("hour:O", title="Hour of day (local)"),
            y=alt.Y("dow_label:O", title="Day of week", sort=DOW_LABELS),
            color=alt.Color(f"{value_col}:Q", title=title, scale=scale),
            tooltip=[
                alt.Tooltip("dow_label:N", title="Day"),
                alt.Tooltip("hour:O", title="Hour"),
                alt.Tooltip(f"{value_col}:Q", title=title, format=fmt),
                alt.Tooltip("n:Q", title="Observations"),
            ],
        )
        .properties(height=220, title=title)
    )


def _range_chart(df: pd.DataFrame, capacity: float | None) -> alt.Chart:
    if df.empty:
        return alt.Chart(pd.DataFrame()).mark_text(text="(no data)")
    band = (
        alt.Chart(df)
        .mark_area(opacity=0.25)
        .encode(x="hour:O", y="p10_bikes:Q", y2="p90_bikes:Q")
    )
    iqr = (
        alt.Chart(df)
        .mark_area(opacity=0.45)
        .encode(x="hour:O", y="p25_bikes:Q", y2="p75_bikes:Q")
    )
    median = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(x="hour:O", y=alt.Y("p50_bikes:Q", title="Bikes (total)"))
    )
    ebike_median = (
        alt.Chart(df)
        .mark_line(point=True, strokeDash=[4, 3], color="#d62728")
        .encode(x="hour:O", y=alt.Y("p50_ebikes:Q", title="Ebikes (median)"))
    )
    layers = [band, iqr, median, ebike_median]
    if capacity:
        cap_df = pd.DataFrame({"y": [capacity]})
        cap_rule = alt.Chart(cap_df).mark_rule(strokeDash=[2, 2], color="#999").encode(y="y:Q")
        layers.append(cap_rule)
    return alt.layer(*layers).properties(height=320, title="Bike count by hour of day (p10/p25/p50/p75/p90)")


def _forecast_chart(fc: pd.DataFrame, current_value: int) -> alt.Chart:
    rows = [{"lag": 0, "value": float(current_value), "series": "Given current state", "n": None}]
    unc_anchor = float(fc["e_unconditional"].dropna().mean()) if fc["e_unconditional"].notna().any() else None
    if unc_anchor is not None:
        rows.append({"lag": 0, "value": unc_anchor, "series": "Long-run average", "n": None})

    for _, r in fc.iterrows():
        lag = int(r["lag_minutes"])
        if pd.notna(r["e_conditional"]):
            rows.append({"lag": lag, "value": float(r["e_conditional"]),
                         "series": "Given current state", "n": int(r["n_conditional"])})
        if pd.notna(r["e_unconditional"]):
            rows.append({"lag": lag, "value": float(r["e_unconditional"]),
                         "series": "Long-run average", "n": int(r["n_total"])})

    chart_df = pd.DataFrame(rows)
    line = (
        alt.Chart(chart_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("lag:O", title="Minutes ahead"),
            y=alt.Y("value:Q", title="Expected ebikes"),
            color=alt.Color(
                "series:N",
                title=None,
                scale=alt.Scale(
                    domain=["Given current state", "Long-run average"],
                    range=["#1f77b4", "#888888"],
                ),
            ),
            tooltip=[
                alt.Tooltip("series:N"),
                alt.Tooltip("lag:O", title="Lag (min)"),
                alt.Tooltip("value:Q", title="Expected", format=".2f"),
                alt.Tooltip("n:Q", title="Samples"),
            ],
        )
    )

    band_rows = [
        {"lag": int(r["lag_minutes"]), "p25": float(r["p25_conditional"]), "p75": float(r["p75_conditional"])}
        for _, r in fc.iterrows()
        if pd.notna(r["p25_conditional"]) and pd.notna(r["p75_conditional"])
    ]
    if band_rows:
        band_df = pd.DataFrame([{"lag": 0, "p25": float(current_value), "p75": float(current_value)}] + band_rows)
        band = (
            alt.Chart(band_df)
            .mark_area(opacity=0.18, color="#1f77b4")
            .encode(x="lag:O", y="p25:Q", y2="p75:Q")
        )
        return alt.layer(band, line).properties(height=320, title="Ebike forecast (decay toward long-run average)")
    return line.properties(height=320, title="Ebike forecast (decay toward long-run average)")


def _prediction_card(
    title: str,
    station: dict | None,
    *,
    primary: str = "score",
    probability_key: str = "p_has_ebike_10m",
    debug_probabilities: bool = False,
) -> None:
    st.markdown(f"**{title}**")
    if not station:
        st.metric("Station", "-")
        st.caption("No matching station in the search radius.")
        return
    value = (
        _score_label(station.get("walk_adjusted_score"))
        if primary == "score"
        else _prob_label(station.get(probability_key))
    )
    st.metric(station.get("name") or "Station", value)
    st.caption(
        f"{_prob_label(station.get('p_arrival'))} at arrival · "
        f"{_minutes_label(station.get('arrival_time_minutes'))} walk · "
        f"{_distance_label(station.get('distance_km'))} away · "
        f"{station.get('current_ebikes', 0)} ebikes now · "
        f"{station.get('active_model_key') or 'model'} · "
        f"{station.get('prediction_cache_status') or '-'}"
    )
    if debug_probabilities:
        raw_key = probability_key if primary == "probability" else "p_arrival"
        st.caption(f"raw probability = {_prob_label(station.get(raw_key), debug=True)}")


def _prediction_map(lat: float, lon: float, result: dict) -> None:
    alternatives = result.get("reliable_alternatives") or []
    practical = result.get("best_practical_station_5_10m") or result.get("best_station_5_10m")
    highest_probability = result.get("highest_probability_station_5_10m")
    free = result.get("closest_free_ebike")

    rows_by_id: dict[str, dict] = {}
    for item in alternatives:
        sid = item.get("station_id")
        if sid:
            rows_by_id[str(sid)] = dict(item)
    for item in [practical, highest_probability]:
        if item and item.get("station_id"):
            rows_by_id[str(item["station_id"])] = dict(item)

    station_rows = []
    practical_id = str(practical.get("station_id")) if practical and practical.get("station_id") else None
    probability_id = (
        str(highest_probability.get("station_id"))
        if highest_probability and highest_probability.get("station_id")
        else None
    )
    for item in rows_by_id.values():
        if item.get("lat") is None or item.get("lon") is None:
            continue
        score = float(item.get("walk_adjusted_score") or 0.0)
        p10 = float(item.get("p_has_ebike_10m") or 0.0)
        sid = str(item.get("station_id"))
        role = "Alternative"
        fill = [180, 180, 180, 180]
        radius_m = 45.0 + 75.0 * max(0.0, min(1.0, score))
        if score >= 0.80:
            fill = [238, 137, 0, 215]
        elif score >= 0.65:
            fill = [240, 200, 80, 205]
        if sid == probability_id:
            role = "Highest probability"
            fill = [245, 245, 245, 235]
            radius_m = max(radius_m, 100.0)
        if sid == practical_id:
            role = "Best practical"
            fill = [255, 204, 0, 245]
            radius_m = max(radius_m, 125.0)
        station_rows.append({
            "station_id": sid,
            "name": item.get("name") or "Station",
            "lat": float(item["lat"]),
            "lon": float(item["lon"]),
            "role": role,
            "walk_score_label": _score_label(score),
            "p10_label": _prob_label(p10),
            "distance_label": _distance_label(item.get("distance_km")),
            "current_ebikes": int(item.get("current_ebikes") or 0),
            "confidence": item.get("confidence") or "unknown",
            "radius_m": radius_m,
            "fill_r": fill[0],
            "fill_g": fill[1],
            "fill_b": fill[2],
            "fill_a": fill[3],
            "line_r": 35,
            "line_g": 35,
            "line_b": 35,
            "line_a": 255,
            "label": "Best" if sid == practical_id else ("Prob" if sid == probability_id else ""),
        })

    if not station_rows:
        return

    station_df = pd.DataFrame(station_rows)
    layers = [
        pdk.Layer(
            "ScatterplotLayer",
            id="prediction_stations",
            data=station_df,
            get_position="[lon, lat]",
            get_radius="radius_m",
            get_fill_color="[fill_r, fill_g, fill_b, fill_a]",
            get_line_color="[line_r, line_g, line_b, line_a]",
            line_width_min_pixels=2,
            pickable=True,
            auto_highlight=True,
            radius_min_pixels=7,
            radius_max_pixels=42,
            stroked=True,
            filled=True,
        ),
        pdk.Layer(
            "TextLayer",
            id="prediction_labels",
            data=station_df[station_df["label"] != ""],
            get_position="[lon, lat]",
            get_text="label",
            get_size=13,
            get_color=[20, 20, 20, 255],
            get_alignment_baseline="'center'",
            get_pixel_offset=[0, -28],
            pickable=False,
        ),
        pdk.Layer(
            "ScatterplotLayer",
            id="you_prediction",
            data=pd.DataFrame({"lat": [lat], "lon": [lon], "label": ["You"]}),
            get_position="[lon, lat]",
            get_radius=42,
            get_fill_color=[20, 20, 20, 235],
            get_line_color=[255, 255, 255, 255],
            line_width_min_pixels=2,
            radius_min_pixels=8,
            radius_max_pixels=14,
            pickable=False,
            stroked=True,
            filled=True,
        ),
    ]

    if free and free.get("lat") is not None and free.get("lon") is not None:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                id="free_prediction",
                data=pd.DataFrame([{
                    "lat": float(free["lat"]),
                    "lon": float(free["lon"]),
                    "name": f"Free bike #{free.get('name') or free.get('bike_id', '')}",
                "distance_label": _distance_label(free.get("distance_km")),
            }]),
                get_position="[lon, lat]",
                get_radius=38,
                get_fill_color=[255, 255, 255, 235],
                get_line_color=[20, 20, 20, 255],
                line_width_min_pixels=2,
                radius_min_pixels=6,
                radius_max_pixels=12,
                pickable=False,
                stroked=True,
                filled=True,
            )
        )

    tooltip = {
        "html": (
            "<b>{name}</b><br/>"
            "{role}<br/>"
            "Walk score: <b>{walk_score_label}</b><br/>"
            "P eBike in 10m: <b>{p10_label}</b><br/>"
            "{distance_label} away · {current_ebikes} now<br/>"
            "Confidence: {confidence}"
        ),
        "style": {"backgroundColor": "rgba(30,30,30,0.88)", "color": "white", "fontSize": "12px"},
    }
    deck = pdk.Deck(
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=15, pitch=0),
        layers=layers,
        tooltip=tooltip,
    )
    event = st.pydeck_chart(
        deck,
        use_container_width=True,
        selection_mode="single-object",
        on_select="rerun",
        key="prediction_map",
    )
    st.caption(
        "Gold = best practical pick. White marker = highest raw probability if different. "
        "Orange/amber/gray alternatives are ranked by walk score; larger circles are stronger scores. "
        "Black dot = you."
    )

    clicked = _picked_object(event, layer_id="prediction_stations")
    if clicked:
        st.session_state["selected_station"] = clicked.get("station_id")
        with st.container(border=True):
            st.markdown(f"**{clicked.get('name', 'Selected station')}**")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Walk score", clicked.get("walk_score_label", "-"))
            c2.metric("P eBike 10m", clicked.get("p10_label", "-"))
            c3.metric("Distance", clicked.get("distance_label", "-"))
            c4.metric("Ebikes now", clicked.get("current_ebikes", "-"))


def _plan_stop_card(index: int, stop: dict) -> None:
    badge = f"Stop {index}"
    stop_type = stop.get("stop_type")
    if stop_type == "free_bike":
        title = f"#{badge} · Free bike {stop.get('name') or stop.get('bike_id')}"
        primary_label = "P(still there)"
        primary_value = _prob_label(stop.get("p_has_ebike_at_horizon"))
    else:
        title = f"#{badge} · {stop.get('name') or 'Station'}"
        primary_label = "P(≥1 eBike)"
        primary_value = _prob_label(stop.get("p_has_ebike_at_horizon"))
    st.markdown(f"**{title}**")
    st.metric(primary_label, primary_value)
    walk_total = stop.get("walk_minutes_from_user", 0.0)
    walk_step = stop.get("walk_minutes_from_previous", walk_total)
    distance = stop.get("distance_from_user_km")
    expected = stop.get("expected_pickup")
    horizon = stop.get("horizon_minutes")
    parts: list[str] = []
    if walk_step is not None:
        parts.append(f"{_minutes_label(walk_step)} from last stop")
    if walk_total is not None:
        parts.append(f"{_minutes_label(walk_total)} from you")
    if distance is not None:
        parts.append(_distance_label(distance))
    if expected is not None:
        parts.append(f"E[bikes] ≈ {float(expected):.2f}")
    if horizon is not None:
        parts.append(f"horizon {int(horizon)}m")
    if stop_type == "station":
        current = stop.get("current_ebikes")
        if current is not None:
            parts.append(f"{int(current)} now")
        if stop.get("degraded"):
            parts.append("degraded PMF")
    elif stop_type == "free_bike" and stop.get("p_stays_source"):
        parts.append(f"source: {stop['p_stays_source']}")
    st.caption(" · ".join(parts))


def _plan_map(
    lat: float,
    lon: float,
    plan: dict,
    *,
    key: str = "plan_map",
    unused_free_bikes: list[dict] | None = None,
) -> None:
    stops = plan.get("stops") or []
    if not stops and not unused_free_bikes:
        return

    rows: list[dict] = []
    path_coords: list[list[float]] = [[lon, lat]]
    for idx, stop in enumerate(stops, start=1):
        s_lat = stop.get("lat")
        s_lon = stop.get("lon")
        if s_lat is None or s_lon is None:
            continue
        stop_type = stop.get("stop_type") or "station"
        if stop_type == "free_bike":
            fill = [60, 180, 255, 230]
            radius_m = 120.0
            count_str = "free-floating"
        else:
            fill = [255, 180, 40, 235]
            radius_m = 140.0
            current = stop.get("current_ebikes")
            count_str = f"{int(current)} eBikes now" if current is not None else "—"
        expected = stop.get("expected_pickup")
        expected_str = f"{float(expected):.1f}" if expected is not None else "—"
        walk_step = stop.get("walk_minutes_from_previous")
        walk_total = stop.get("walk_minutes_from_user")
        rows.append({
            "label": str(idx),
            "name": stop.get("name") or stop_type,
            "lat": float(s_lat),
            "lon": float(s_lon),
            "p_label": _prob_label(stop.get("p_has_ebike_at_horizon")),
            "walk_step_label": _minutes_label(walk_step if walk_step is not None else walk_total or 0.0),
            "walk_total_label": _minutes_label(walk_total if walk_total is not None else 0.0),
            "count_label": count_str,
            "expected_label": expected_str,
            "horizon_label": f"{int(stop.get('horizon_minutes') or 10)}m",
            "fill_r": fill[0],
            "fill_g": fill[1],
            "fill_b": fill[2],
            "fill_a": fill[3],
            "radius_m": radius_m,
            "stop_type": stop_type,
        })
        path_coords.append([float(s_lon), float(s_lat)])

    if not rows and not unused_free_bikes:
        return

    unused_rows: list[dict] = []
    for bike in unused_free_bikes or []:
        if bike.get("lat") is None or bike.get("lon") is None:
            continue
        unused_rows.append({
            "name": f"Free bike {bike.get('name') or bike.get('bike_id')}",
            "lat": float(bike["lat"]),
            "lon": float(bike["lon"]),
            "p_label": _prob_label(bike.get("p_stays")),
            "walk_total_label": _minutes_label(bike.get("walk_minutes_from_user") or 0.0),
            "count_label": "free-floating (not in plan)",
            "expected_label": f"{float(bike.get('p_stays') or 0.0):.2f}",
            "horizon_label": f"{int(bike.get('horizon_minutes') or 10)}m",
        })

    layers = []
    if len(path_coords) >= 2:
        layers.append(
            pdk.Layer(
                "PathLayer",
                id=f"{key}_path",
                data=pd.DataFrame([{"path": path_coords, "name": "Plan route"}]),
                get_path="path",
                get_color=[40, 110, 200, 210],
                get_width=4,
                width_min_pixels=3,
                pickable=False,
            )
        )
    if unused_rows:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                id=f"{key}_unused_free",
                data=pd.DataFrame(unused_rows),
                get_position="[lon, lat]",
                get_radius=55,
                get_fill_color=[120, 200, 255, 130],
                get_line_color=[20, 20, 20, 160],
                line_width_min_pixels=1,
                radius_min_pixels=5,
                radius_max_pixels=12,
                pickable=True,
                stroked=True,
                filled=True,
            )
        )
    if rows:
        layers.extend([
            pdk.Layer(
                "ScatterplotLayer",
                id=f"{key}_stops",
                data=pd.DataFrame(rows),
                get_position="[lon, lat]",
                get_radius="radius_m",
                get_fill_color="[fill_r, fill_g, fill_b, fill_a]",
                get_line_color=[20, 20, 20, 255],
                line_width_min_pixels=2,
                radius_min_pixels=10,
                radius_max_pixels=42,
                pickable=True,
                stroked=True,
                filled=True,
            ),
            pdk.Layer(
                "TextLayer",
                id=f"{key}_numbers",
                data=pd.DataFrame(rows),
                get_position="[lon, lat]",
                get_text="label",
                get_size=18,
                get_color=[20, 20, 20, 255],
                get_alignment_baseline="'center'",
                get_pixel_offset=[0, 0],
                pickable=False,
            ),
        ])
    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            id=f"{key}_you",
            data=pd.DataFrame({"lat": [lat], "lon": [lon]}),
            get_position="[lon, lat]",
            get_radius=42,
            get_fill_color=[20, 20, 20, 235],
            get_line_color=[255, 255, 255, 255],
            line_width_min_pixels=2,
            radius_min_pixels=8,
            radius_max_pixels=14,
            pickable=False,
            stroked=True,
            filled=True,
        )
    )

    tooltip = {
        "html": (
            "<b>{name}</b><br/>"
            "P(get bike here): <b>{p_label}</b><br/>"
            "Available: {count_label}<br/>"
            "E[count at +{horizon_label}]: {expected_label}<br/>"
            "Walk from previous: {walk_step_label}<br/>"
            "Walk from start: {walk_total_label}"
        ),
        "style": {"backgroundColor": "rgba(30,30,30,0.88)", "color": "white", "fontSize": "12px"},
    }
    deck = pdk.Deck(
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=14, pitch=0),
        layers=layers,
        tooltip=tooltip,
    )
    st.pydeck_chart(deck, use_container_width=True, key=key)


def _render_multi_bike_plan(
    lat: float,
    lon: float,
    k: int,
    near_radius_km: float,
    search_radius_km: float,
    plan_horizon_minutes: int,
    target_success_probability: float,
    query_label: str | None,
) -> None:
    try:
        result = _prediction_multi_bike_plan(
            lat,
            lon,
            int(k),
            near_radius_km,
            search_radius_km,
            plan_horizon_minutes,
            target_success_probability,
            query_label,
        )
    except (duckdb.Error, requests.RequestException) as exc:
        st.warning(f"Multi-bike plan unavailable: {exc}")
        return

    plan = result.get("plan") or {}
    alternatives = result.get("alternative_plans") or []
    candidate_count = result.get("candidate_count", 0)
    station_count = result.get("station_candidate_count", 0)
    free_count = result.get("free_bike_candidate_count", 0)
    feasible = bool(result.get("feasible"))
    unmet = int(result.get("unmet_demand") or 0)

    metrics_cols = st.columns(4)
    metrics_cols[0].metric(
        f"P(≥{int(k)} bikes)", _prob_label(plan.get("success_probability"))
    )
    metrics_cols[1].metric(
        "Total walking", _minutes_label(plan.get("total_walking_minutes"))
    )
    metrics_cols[2].metric("Stops in plan", str(int(plan.get("n_stops") or 0)))
    metrics_cols[3].metric("Expected bikes", f"{float(plan.get('expected_bikes') or 0.0):.2f}")

    badges = []
    badges.append(f"{station_count} station candidates")
    badges.append(f"{free_count} free-bike candidates")
    badges.append(f"target {int(target_success_probability * 100)}%")
    if not feasible:
        badges.append(f"NOT feasible · unmet ≈ {unmet}")
    st.caption(" · ".join(badges))

    if not plan.get("stops"):
        st.info("No reachable stops produced a plan. Try a wider search radius.")
        return

    unused_free = result.get("unused_free_bikes") or []
    nearby_unused_for_map = unused_free[:8]

    _plan_map(
        lat,
        lon,
        plan,
        key=f"plan_map_k{int(k)}",
        unused_free_bikes=nearby_unused_for_map,
    )

    st.markdown("**Plan**")
    stops = plan.get("stops") or []
    cols_per_row = 3
    for start in range(0, len(stops), cols_per_row):
        slice_ = stops[start : start + cols_per_row]
        cols = st.columns(len(slice_))
        for idx_offset, stop in enumerate(slice_):
            with cols[idx_offset]:
                _plan_stop_card(start + idx_offset + 1, stop)

    if unused_free:
        with st.expander(
            f"Free-floating eBikes nearby not in plan ({len(unused_free)})",
            expanded=True,
        ):
            st.caption(
                "Light-blue dots on the map. The planner kept these as backups "
                "because their P(stays) was lower than what's already in the plan, "
                "but they're still useful if a stop is empty when you arrive."
            )
            preview = unused_free[: min(6, len(unused_free))]
            cols_per_row = 3
            for start in range(0, len(preview), cols_per_row):
                slice_ = preview[start : start + cols_per_row]
                cols = st.columns(len(slice_))
                for idx, bike in enumerate(slice_):
                    with cols[idx]:
                        st.markdown(f"**Free bike {bike.get('name') or bike.get('bike_id')}**")
                        st.metric(
                            f"P(stays {int(bike.get('horizon_minutes') or 10)}m)",
                            _prob_label(bike.get("p_stays")),
                        )
                        st.caption(
                            f"{_distance_label(bike.get('distance_km'))} away · "
                            f"{_minutes_label(bike.get('walk_minutes_from_user') or 0.0)} walk · "
                            f"source: {bike.get('p_stays_source') or '—'}"
                        )

    if alternatives:
        with st.expander("Alternative plans", expanded=True):
            tabs = st.tabs([alt.get("strategy", f"Plan {i+1}") for i, alt in enumerate(alternatives)])
            for tab, alt in zip(tabs, alternatives):
                with tab:
                    alt_cols = st.columns(3)
                    alt_cols[0].metric("Success", _prob_label(alt.get("success_probability")))
                    alt_cols[1].metric("Walk", _minutes_label(alt.get("total_walking_minutes")))
                    alt_cols[2].metric("Stops", str(int(alt.get("n_stops") or 0)))
                    _plan_map(
                        lat,
                        lon,
                        alt,
                        key=f"plan_map_alt_{alt.get('strategy', 'x')}_k{int(k)}",
                    )

    st.caption(
        "Probability assumes independent stops; free-bike survival via tile_predictor. "
        f"Active model: {result.get('active_model_key') or '—'}. "
        f"Cache: {result.get('prediction_cache_status') or '—'}."
    )


def _prediction_service_section(lat: float, lon: float) -> None:
    st.markdown("### Prediction service")
    near_radius_km = 0.5
    search_radius_km = 1.5
    query_label = st.session_state.get("user_location_label") or f"{lat:.4f}, {lon:.4f}"
    k = int(
        st.number_input(
            "How many ebikes do you need?",
            min_value=1,
            max_value=10,
            value=1,
            step=1,
            key="prediction_service_k",
            help="When greater than 1, the page returns an ordered pickup plan instead of a single station.",
        )
    )
    if k > 1:
        _render_multi_bike_plan(
            lat=lat,
            lon=lon,
            k=k,
            near_radius_km=near_radius_km,
            search_radius_km=search_radius_km,
            plan_horizon_minutes=10,
            target_success_probability=0.85,
            query_label=query_label,
        )
        return
    try:
        result = _prediction_recommendation(lat, lon, near_radius_km, search_radius_km, query_label)
    except (duckdb.Error, requests.RequestException) as exc:
        st.warning(f"Prediction service unavailable: {exc}")
        return

    practical = result.get("best_practical_station_5_10m") or result.get("best_station_5_10m")
    highest_probability = result.get("highest_probability_station_5_10m")
    docked = result.get("closest_docked_station_with_ebike")
    free = result.get("closest_free_ebike")
    model = result.get("model", {})
    formula = result.get("ranking_formula") or {}
    perf = _prediction_performance(24)
    _model_status_banner(perf, model)
    debug_probabilities = st.checkbox("Show raw probabilities", value=False, key="prediction_debug_probabilities")

    top_cols = st.columns(2)
    with top_cols[0]:
        _prediction_card("Best practical docked eBike", practical, primary="score", debug_probabilities=debug_probabilities)
    with top_cols[1]:
        _prediction_card(
            "Highest-probability docked eBike",
            highest_probability,
            primary="probability",
            probability_key="p_arrival",
            debug_probabilities=debug_probabilities,
        )

    if formula:
        st.caption(
            "Walk score = "
            f"{display_probability(formula.get('probability_weight', 0))} probability + "
            f"{display_probability(formula.get('distance_weight', 0))} proximity + "
            f"{display_probability(formula.get('current_count_weight', 0))} current count. "
            "The probability-only option is shown separately."
        )

    if practical and practical.get("model_predictions"):
        pred_df = pd.DataFrame(practical["model_predictions"])
        pred_df["arrival"] = pred_df["p_arrival"]
        pred_chart_df = pred_df.melt(
            id_vars=["label"],
            value_vars=["p_arrival", "p_has_ebike_5m", "p_has_ebike_10m", "p_has_ebike_15m", "p_has_ebike_20m"],
            var_name="horizon",
            value_name="probability",
        )
        pred_chart_df["horizon"] = pred_chart_df["horizon"].map({
            "p_arrival": "arrival",
            "p_has_ebike_5m": "5m",
            "p_has_ebike_10m": "10m",
            "p_has_ebike_15m": "15m",
            "p_has_ebike_20m": "20m",
        })
        st.altair_chart(
            alt.Chart(pred_chart_df)
            .mark_bar()
            .encode(
                x=alt.X("label:N", title=None),
                y=alt.Y("probability:Q", title="P eBike", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("horizon:N"),
                xOffset="horizon:N",
                tooltip=[
                    alt.Tooltip("label:N", title="Model"),
                    alt.Tooltip("horizon:N", title="Horizon"),
                    alt.Tooltip("probability:Q", title="P eBike", format=".1%"),
                ],
            )
            .properties(height=180),
            use_container_width=True,
        )

    _prediction_map(lat, lon, result)

    cols = st.columns(3)
    with cols[0]:
        _prediction_card(
            "Closest docked eBike now",
            docked,
            primary="probability",
            probability_key="p_survives_10m",
            debug_probabilities=debug_probabilities,
        )
    with cols[1]:
        st.markdown("**Closest free-floating eBike**")
        if free:
            st.metric(f"#{free.get('name') or free.get('bike_id', '')}", _distance_label(free.get("distance_km")))
            st.caption("Live GBFS free-bike feed")
        else:
            st.metric("Free bike", "-")
            msg = result.get("free_bike_error") or "None inside the search radius."
            st.caption(msg)
    with cols[2]:
        st.markdown("**Model**")
        st.metric(model.get("method", "model"), f"{int(model.get('training_examples') or 0):,} examples")
        freshness = result.get("db_freshness_minutes")
        st.caption(
            f"Freshness {_minutes_label(freshness)} ago · "
            f"{result.get('candidate_count', 0)} stations scored"
        )

    alternatives = result.get("reliable_alternatives") or []
    if alternatives:
        with st.expander("Ranked reliable alternatives", expanded=False):
            table = pd.DataFrame(alternatives)
            alt_columns = [
                "name",
                "walk_adjusted_score",
                "distance_km",
                "distance_score",
                "current_ebikes",
                "expected_ebikes_10m",
                "p_has_ebike_5m",
                "p_has_ebike_10m",
                "p_dock_constrained_arrival_10m",
                "confidence",
                "sample_size_10m",
                "reason",
            ]
            display_table = table[[column for column in alt_columns if column in table.columns]].copy()
            for column in [
                "walk_adjusted_score",
                "distance_score",
                "p_has_ebike_5m",
                "p_has_ebike_10m",
                "p_dock_constrained_arrival_10m",
            ]:
                if column in display_table.columns:
                    display_table[column] = display_table[column].map(_prob_label)
            st.dataframe(
                display_table,
                hide_index=True,
                column_config={
                    "name": "Station",
                    "walk_adjusted_score": "Walk score",
                    "distance_km": st.column_config.NumberColumn("Distance km", format="%.2f"),
                    "distance_score": "Proximity",
                    "current_ebikes": "Now",
                    "expected_ebikes_10m": st.column_config.NumberColumn("Expected eBikes 10m", format="%.2f"),
                    "p_has_ebike_5m": "P eBike 5m",
                    "p_has_ebike_10m": "P eBike 10m",
                    "p_dock_constrained_arrival_10m": "Dock constrained",
                    "confidence": "Confidence",
                    "sample_size_10m": "Samples",
                    "reason": "Reason",
                },
            )

    _model_performance_panel(perf)

    artifacts = pd.DataFrame(_model_artifacts())
    if not artifacts.empty:
        with st.expander("Model detail panel", expanded=False):
            detail_cols = [
                "artifact_id",
                "model_key",
                "model_version",
                "trained_at",
                "train_start",
                "train_end",
                "valid_start",
                "valid_end",
                "feature_columns",
                "calibration_json",
                "is_primary_eligible",
                "is_active",
            ]
            panel = artifacts[[column for column in detail_cols if column in artifacts.columns]].copy()
            if "feature_columns" in panel:
                panel["feature_count"] = panel["feature_columns"].apply(lambda value: len(value) if isinstance(value, list) else 0)
                panel = panel.drop(columns=["feature_columns"])
            st.dataframe(panel, hide_index=True)

    place_models = pd.DataFrame(perf.get("by_place_model") or [])
    if not place_models.empty:
        with st.expander("Model leaderboard by searched place", expanded=False):
            st.caption("Places use the searched lat/lon rounded to about a block so nearby repeated searches accumulate.")
            st.dataframe(
                place_models[
                    ["query_label", "query_place_key", "model_label", "n", "rank_loss", "brier_score", "log_loss"]
                ],
                hide_index=True,
                column_config={
                    "query_label": "Searched place",
                    "query_place_key": "Place key",
                    "model_label": "Model",
                    "n": "Resolved forecasts",
                    "rank_loss": st.column_config.NumberColumn("Rank loss", format="%.3f"),
                    "brier_score": st.column_config.NumberColumn("Brier", format="%.3f"),
                    "log_loss": st.column_config.NumberColumn("Log loss", format="%.3f"),
                },
            )

    by_horizon = pd.DataFrame(perf.get("by_horizon") or [])
    if not by_horizon.empty:
        with st.expander("Rolling model performance", expanded=False):
            horizon_columns = [
                "horizon_minutes",
                "n",
                "brier_score",
                "log_loss",
                "count_log_loss",
                "crps",
                "observed_rate",
                "mean_prediction",
            ]
            by_horizon_display = by_horizon[[column for column in horizon_columns if column in by_horizon.columns]].copy()
            for column in ["observed_rate", "mean_prediction"]:
                if column in by_horizon_display.columns:
                    by_horizon_display[column] = by_horizon_display[column].map(_prob_label)
            st.dataframe(
                by_horizon_display,
                hide_index=True,
                column_config={
                    "horizon_minutes": "Horizon",
                    "n": "Resolved forecasts",
                    "brier_score": st.column_config.NumberColumn("Brier", format="%.3f"),
                    "log_loss": st.column_config.NumberColumn("Log loss", format="%.3f"),
                    "count_log_loss": st.column_config.NumberColumn("Count NLL", format="%.3f"),
                    "crps": st.column_config.NumberColumn("CRPS", format="%.3f"),
                    "observed_rate": "Observed hit rate",
                    "mean_prediction": "Mean prediction",
                },
            )


def _station_prediction_detail_section(station_id: str) -> None:
    user_loc = st.session_state.get("user_location")
    user_lat = float(user_loc[0]) if user_loc else None
    user_lon = float(user_loc[1]) if user_loc else None

    try:
        scored, model = _station_prediction_detail(station_id, user_lat, user_lon)
    except duckdb.Error as exc:
        st.warning(f"Prediction service unavailable for this station: {exc}")
        return
    if scored.empty:
        st.info("No live station status yet for the prediction model.")
        return

    row = scored.iloc[0]
    current = int(row.get("num_ebikes_available") or 0)
    transition_label = "Appears" if current == 0 else "Survives"
    transition_5 = row.get("p_appears_5m") if current == 0 else row.get("p_survives_5m")
    transition_10 = row.get("p_appears_10m") if current == 0 else row.get("p_survives_10m")

    st.subheader("Prediction service forecast")
    m = st.columns(6)
    m[0].metric("P eBike 5m", _prob_label(row.get("p_has_ebike_5m")))
    m[1].metric("P eBike 10m", _prob_label(row.get("p_has_ebike_10m")))
    m[2].metric("P zero 10m", _prob_label(row.get("p_zero_10m")))
    m[3].metric(f"{transition_label} 10m", _prob_label(transition_10))
    m[4].metric("Walk score", _score_label(row.get("walk_adjusted_score")))
    m[5].metric("Confidence", row.get("confidence") or "-")

    model_pred_rows = []
    for model_key in predictor.MODEL_KEYS:
        spec = predictor.MODEL_SPECS[model_key]
        model_pred_rows.append({
            "model_key": model_key,
            "model": spec["label"],
            "p_has_ebike_5m": row.get(f"p_has_ebike_5m_{model_key}"),
            "p_has_ebike_10m": row.get(f"p_has_ebike_10m_{model_key}"),
            "p_zero_10m": row.get(f"p_zero_10m_{model_key}"),
            f"p_{transition_label.lower()}_10m": (
                row.get(f"p_appears_10m_{model_key}") if current == 0
                else row.get(f"p_survives_10m_{model_key}")
            ),
            "learned_10m": row.get(f"p_learned_10m_{model_key}"),
            "empirical_10m": row.get(f"p_empirical_10m_{model_key}"),
            "expected_ebikes_10m": row.get(f"expected_ebikes_10m_{model_key}"),
            "p_dock_constrained_arrival_10m": row.get(f"p_dock_constrained_arrival_10m_{model_key}"),
            "walk_adjusted_score": row.get(f"walk_adjusted_score_{model_key}"),
            "sample_size_10m": row.get(f"sample_size_10m_{model_key}"),
        })
    model_pred_df = pd.DataFrame(model_pred_rows)
    chart_rows = model_pred_df.melt(
        id_vars=["model"],
        value_vars=["p_has_ebike_5m", "p_has_ebike_10m"],
        var_name="horizon",
        value_name="probability",
    )
    chart_rows["horizon"] = chart_rows["horizon"].map({
        "p_has_ebike_5m": "5m",
        "p_has_ebike_10m": "10m",
    })
    model_chart = (
        alt.Chart(chart_rows)
        .mark_bar()
        .encode(
            x=alt.X("model:N", title=None),
            y=alt.Y("probability:Q", title="P eBike", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "horizon:N",
                title="Horizon",
                scale=alt.Scale(domain=["5m", "10m"], range=["#f2c94c", "#ee8900"]),
            ),
            xOffset="horizon:N",
            tooltip=[
                alt.Tooltip("model:N", title="Model"),
                alt.Tooltip("horizon:N", title="Horizon"),
                alt.Tooltip("probability:Q", title="P eBike", format=".1%"),
            ],
        )
        .properties(height=220)
    )
    st.altair_chart(model_chart, use_container_width=True)

    model_pred_display = model_pred_df.copy()
    for column in [
        "p_has_ebike_5m",
        "p_has_ebike_10m",
        "p_zero_10m",
        f"p_{transition_label.lower()}_10m",
        "learned_10m",
        "empirical_10m",
        "p_dock_constrained_arrival_10m",
        "walk_adjusted_score",
    ]:
        if column in model_pred_display.columns:
            model_pred_display[column] = model_pred_display[column].map(_prob_label)
    st.dataframe(
        model_pred_display,
        hide_index=True,
        column_config={
            "model_key": None,
            "model": "Model",
            "p_has_ebike_5m": "P eBike 5m",
            "p_has_ebike_10m": "P eBike 10m",
            "p_zero_10m": "P zero 10m",
            f"p_{transition_label.lower()}_10m": f"P {transition_label} 10m",
            "learned_10m": "Learned 10m",
            "empirical_10m": "Empirical 10m",
            "expected_ebikes_10m": st.column_config.NumberColumn("Expected eBikes 10m", format="%.2f"),
            "p_dock_constrained_arrival_10m": "Dock constrained",
            "walk_adjusted_score": "Walk score",
            "sample_size_10m": "Samples",
        },
    )

    prob_rows = [
        {"horizon": "5m", "measure": "Has eBike", "probability": row.get("p_has_ebike_5m"), "color": "#f2c94c"},
        {"horizon": "10m", "measure": "Has eBike", "probability": row.get("p_has_ebike_10m"), "color": "#f2c94c"},
        {"horizon": "5m", "measure": "Zero eBikes", "probability": row.get("p_zero_5m"), "color": "#4d4d4d"},
        {"horizon": "10m", "measure": "Zero eBikes", "probability": row.get("p_zero_10m"), "color": "#4d4d4d"},
        {"horizon": "5m", "measure": transition_label, "probability": transition_5, "color": "#ee8900"},
        {"horizon": "10m", "measure": transition_label, "probability": transition_10, "color": "#ee8900"},
    ]
    prob_df = pd.DataFrame(prob_rows).dropna(subset=["probability"])
    if not prob_df.empty:
        chart = (
            alt.Chart(prob_df)
            .mark_bar()
            .encode(
                x=alt.X("horizon:N", title="Horizon"),
                y=alt.Y("probability:Q", title="Probability", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color(
                    "measure:N",
                    title=None,
                    scale=alt.Scale(
                        domain=["Has eBike", "Zero eBikes", transition_label],
                        range=["#f2c94c", "#4d4d4d", "#ee8900"],
                    ),
                ),
                column=alt.Column("measure:N", title=None),
                tooltip=[
                    alt.Tooltip("measure:N", title="Measure"),
                    alt.Tooltip("horizon:N", title="Horizon"),
                    alt.Tooltip("probability:Q", title="Probability", format=".1%"),
                ],
            )
            .properties(height=180)
        )
        st.altair_chart(chart, use_container_width=True)

    decomposition = pd.DataFrame([
        {
            "horizon": "5m",
            "final_probability": row.get("p_has_ebike_5m"),
            "learned_model": row.get("p_learned_5m"),
            "empirical_baseline": row.get("p_empirical_5m"),
            "sample_size": row.get("sample_size_5m"),
        },
        {
            "horizon": "10m",
            "final_probability": row.get("p_has_ebike_10m"),
            "learned_model": row.get("p_learned_10m"),
            "empirical_baseline": row.get("p_empirical_10m"),
            "sample_size": row.get("sample_size_10m"),
        },
    ])
    decomposition_display = decomposition.copy()
    for column in ["final_probability", "learned_model", "empirical_baseline"]:
        if column in decomposition_display.columns:
            decomposition_display[column] = decomposition_display[column].map(_prob_label)
    st.dataframe(
        decomposition_display,
        hide_index=True,
        column_config={
            "horizon": "Horizon",
            "final_probability": "Final P eBike",
            "learned_model": "Learned model",
            "empirical_baseline": "Empirical baseline",
            "sample_size": "Same-hour samples",
        },
    )

    features = pd.DataFrame([
        {"feature": "Current eBikes", "value": str(current)},
        {"feature": "Current count bucket", "value": ["0", "1", "2", "3+"][predictor.current_bucket(current)]},
        {"feature": "Trend 5m", "value": f"{float(row.get('trend_5m') or 0):+.1f} eBikes"},
        {"feature": "Trend 10m", "value": f"{float(row.get('trend_10m') or 0):+.1f} eBikes"},
        {"feature": "Trend 15m", "value": f"{float(row.get('trend_15m') or 0):+.1f} eBikes"},
        {"feature": "Churn rate", "value": f"{float(row.get('churn_rate') or 0):.2f}"},
        {"feature": "Station same-hour availability", "value": _prob_label(row.get("station_same_hour_rate"))},
        {"feature": "Nearby same-hour fallback", "value": _prob_label(row.get("nearby_same_hour_rate"))},
        {"feature": "Neighbor same-hour availability", "value": _prob_label(row.get("station_neighbor_same_hour_rate"))},
        {"feature": "Expected trip arrivals in 10m", "value": f"{float(row.get('trip_arrivals_same_hour_10m') or 0):.2f}"},
        {"feature": "Expected trip departures in 10m", "value": f"{float(row.get('trip_departures_same_hour_10m') or 0):.2f}"},
        {"feature": "Expected net trip arrivals in 10m", "value": f"{float(row.get('trip_net_arrivals_same_hour_10m') or 0):+.2f}"},
        {"feature": "Recent trip net arrivals 30m", "value": f"{float(row.get('trip_recent_net_arrivals_30m') or 0):+.0f}"},
        {"feature": "Inbound route volume same hour", "value": f"{float(row.get('route_inbound_trips_same_hour') or 0):.0f}"},
        {"feature": "Inbound route median duration", "value": _minutes_label(row.get("route_inbound_median_duration_minutes"))},
        {"feature": "Inbound trips likely due by horizon", "value": f"{float(row.get('route_inbound_due_horizon') or 0):.2f}"},
        {"feature": "E-bike share of inbound trips", "value": _prob_label(row.get("route_inbound_ebike_share_same_hour"))},
        {"feature": "Apparent temperature", "value": f"{float(row.get('weather_apparent_temperature') or 0):.1f} C"},
        {"feature": "Precipitation", "value": f"{float(row.get('weather_precipitation') or 0):.2f} mm"},
        {"feature": "Wind gust", "value": f"{float(row.get('weather_wind_gusts_10m') or 0):.1f} km/h"},
        {"feature": "Bad weather flag", "value": "yes" if row.get("weather_bad_conditions") else "no"},
        {"feature": "Commute hour", "value": "yes" if row.get("is_commute_hour") else "no"},
        {"feature": "Federal holiday", "value": "yes" if row.get("is_federal_holiday") else "no"},
        {"feature": "Data freshness", "value": _minutes_label(row.get("data_age_minutes"))},
        {"feature": "Distance from selected location", "value": _distance_label(row.get("distance_km"))},
        {"feature": "Proximity component", "value": _prob_label(row.get("distance_score"))},
        {"feature": "Current-count component", "value": _prob_label(row.get("current_count_score"))},
    ])
    with st.expander("Model inputs and scoring components", expanded=False):
        st.dataframe(
            features,
            hide_index=True,
            column_config={
                "feature": "Feature",
                "value": "Value",
            },
        )
        model_cols = st.columns(4)
        model_cols[0].metric("Model method", model.get("method", "-"))
        model_cols[1].metric("Training examples", f"{int(model.get('training_examples') or 0):,}")
        model_cols[2].metric("Positive labels", f"{int(model.get('training_positive') or 0):,}")
        model_cols[3].metric("Negative labels", f"{int(model.get('training_negative') or 0):,}")

    station_metrics = _station_prediction_metrics(station_id, 168)
    if not station_metrics.empty:
        with st.expander("Resolved forecast accuracy for this station", expanded=False):
            station_metrics_display = station_metrics.copy()
            for column in ["observed_rate", "mean_prediction"]:
                if column in station_metrics_display.columns:
                    station_metrics_display[column] = station_metrics_display[column].map(_prob_label)
            st.dataframe(
                station_metrics_display,
                hide_index=True,
                column_config={
                    "model_key": None,
                    "model": "Model",
                    "horizon_minutes": "Horizon",
                    "n": "Resolved forecasts",
                    "brier_score": st.column_config.NumberColumn("Brier", format="%.3f"),
                    "log_loss": st.column_config.NumberColumn("Log loss", format="%.3f"),
                    "rank_loss": st.column_config.NumberColumn("Rank loss", format="%.3f"),
                    "observed_rate": "Observed hit rate",
                    "mean_prediction": "Mean prediction",
                },
            )


def _station_trip_demand_section(station_id: str) -> None:
    anchor = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    anchor = anchor.replace(minute=(anchor.minute // 10) * 10)
    profile = _trip_demand_profile(station_id, anchor.isoformat())

    st.subheader("Historical trip demand near this time")
    if profile.empty:
        st.info(
            "No historical trip-flow rows for this station yet. Run "
            "`uv run python -m divvy.tripdata sync --months 6` to load trip-CSV history, "
            "or `uv run python -m divvy.inferred_flows backfill` to infer flows from live station-status deltas."
        )
        return

    plot_rows = []
    series = [
        ("weighted_departures_10m", "Departures", "Weighted local profile"),
        ("weighted_arrivals_10m", "Arrivals", "Weighted local profile"),
        ("same_weekday_departures_10m", "Departures", "Same weekday only"),
        ("same_weekday_arrivals_10m", "Arrivals", "Same weekday only"),
    ]
    for _, row in profile.iterrows():
        for column, direction, cohort in series:
            plot_rows.append({
                "display_ts": row["display_ts"],
                "time_label": row["time_label"],
                "direction": direction,
                "cohort": cohort,
                "expected_trips": row[column],
                "offset_minutes": row["offset_minutes"],
                "sample_days": row["sample_days"],
                "same_weekday_days": row["same_weekday_days"],
                "same_day_type_days": row["same_day_type_days"],
            })
    chart_df = pd.DataFrame(plot_rows)
    base = (
        alt.Chart(chart_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("display_ts:T", title="Local time"),
            y=alt.Y("expected_trips:Q", title="Expected trips per 10 min"),
            color=alt.Color(
                "direction:N",
                title=None,
                scale=alt.Scale(domain=["Departures", "Arrivals"], range=["#e69f00", "#2b2b2b"]),
            ),
            strokeDash=alt.StrokeDash(
                "cohort:N",
                title=None,
                scale=alt.Scale(domain=["Weighted local profile", "Same weekday only"], range=[[1, 0], [5, 3]]),
            ),
            tooltip=[
                alt.Tooltip("time_label:N", title="Local time"),
                alt.Tooltip("direction:N", title="Flow"),
                alt.Tooltip("cohort:N", title="Cohort"),
                alt.Tooltip("expected_trips:Q", title="Trips / 10m", format=".2f"),
                alt.Tooltip("sample_days:Q", title="Sample days"),
                alt.Tooltip("same_weekday_days:Q", title="Same weekday days"),
                alt.Tooltip("same_day_type_days:Q", title="Same day-type days"),
            ],
        )
        .properties(height=260)
    )
    now_rule = (
        alt.Chart(profile[profile["offset_minutes"] == 0])
        .mark_rule(color="#666666", strokeDash=[2, 2])
        .encode(x="display_ts:T")
    )
    st.altair_chart(base + now_rule, use_container_width=True)

    first = profile.iloc[0]
    csv_days = int(profile["csv_days"].max()) if "csv_days" in profile.columns else 0
    inferred_days = int(profile["inferred_days"].max()) if "inferred_days" in profile.columns else 0
    if csv_days and inferred_days:
        source_mix = f"{csv_days} from trip CSVs, {inferred_days} inferred from live station counts"
    elif csv_days:
        source_mix = f"{csv_days} from trip CSVs"
    elif inferred_days:
        source_mix = f"{inferred_days} inferred from live station counts (no trip CSVs loaded for this range)"
    else:
        source_mix = ""
    st.caption(
        "Solid lines are the weighted profile used for local demand context; dashed lines show same-weekday-only history. "
        f"Current anchor: {first['anchor_time_label']} ({first['anchor_day_type']}). "
        f"Sample: {int(first['sample_days'])} days"
        + (f" ({source_mix})" if source_mix else "")
        + f", {int(first['same_weekday_days'])} same-weekday days. "
        "Weights emphasize trips within roughly one week of today's day-of-year, then same weekday, then matching weekday/weekend. "
        "Inferred buckets are reconstructed from 60s station_status deltas (suspected operator rebalances filtered out); "
        "CSV buckets come from Divvy's monthly trip archives."
    )

    with st.expander("Demand profile table", expanded=False):
        st.dataframe(
            profile[
                [
                    "time_label",
                    "weighted_departures_10m",
                    "weighted_arrivals_10m",
                    "same_weekday_departures_10m",
                    "same_weekday_arrivals_10m",
                    "same_day_type_departures_10m",
                    "same_day_type_arrivals_10m",
                    "weighted_ebike_departures_10m",
                    "weighted_ebike_arrivals_10m",
                    "sample_days",
                    "same_weekday_days",
                    "effective_weight_days",
                ]
            ],
            hide_index=True,
            column_config={
                "time_label": "Local time",
                "weighted_departures_10m": st.column_config.NumberColumn("Departures / 10m", format="%.2f"),
                "weighted_arrivals_10m": st.column_config.NumberColumn("Arrivals / 10m", format="%.2f"),
                "same_weekday_departures_10m": st.column_config.NumberColumn("Same weekday dep.", format="%.2f"),
                "same_weekday_arrivals_10m": st.column_config.NumberColumn("Same weekday arr.", format="%.2f"),
                "same_day_type_departures_10m": st.column_config.NumberColumn("Same type dep.", format="%.2f"),
                "same_day_type_arrivals_10m": st.column_config.NumberColumn("Same type arr.", format="%.2f"),
                "weighted_ebike_departures_10m": st.column_config.NumberColumn("eBike dep.", format="%.2f"),
                "weighted_ebike_arrivals_10m": st.column_config.NumberColumn("eBike arr.", format="%.2f"),
                "sample_days": "Days",
                "same_weekday_days": "Same weekday days",
                "effective_weight_days": st.column_config.NumberColumn("Effective weighted days", format="%.1f"),
            },
        )


def _time_series_chart(df: pd.DataFrame, capacity: float | None) -> alt.Chart:
    if df.empty:
        return alt.Chart(pd.DataFrame()).mark_text(text="(no data)")
    long_df = df.melt(
        id_vars=["ts"],
        value_vars=["num_classic_bikes", "num_ebikes_available", "num_docks_available"],
        var_name="series",
        value_name="count",
    )
    series_order = ["num_classic_bikes", "num_ebikes_available", "num_docks_available"]
    series_titles = {
        "num_classic_bikes": "Classic bikes",
        "num_ebikes_available": "E-bikes",
        "num_docks_available": "Empty docks",
    }
    long_df["series"] = long_df["series"].map(series_titles)
    chart = (
        alt.Chart(long_df)
        .mark_line(interpolate="step-after")
        .encode(
            x=alt.X("ts:T", title="Time"),
            y=alt.Y("count:Q", title="Count"),
            color=alt.Color("series:N", title=None, sort=[series_titles[s] for s in series_order]),
            tooltip=[alt.Tooltip("ts:T"), alt.Tooltip("series:N"), alt.Tooltip("count:Q")],
        )
        .properties(height=320, title="Raw time series")
        .interactive(bind_y=False)
    )
    if capacity:
        rule = alt.Chart(pd.DataFrame({"y": [capacity]})).mark_rule(strokeDash=[2, 2], color="#999").encode(y="y:Q")
        return alt.layer(chart, rule)
    return chart


def _station_map(lat: float, lon: float, radius_km: float = 2.0) -> None:
    """Pydeck map: user + nearby stations that currently have ≥1 ebike.

    Stations are rendered as amber circles whose radius scales with ebike
    count, so 'where can I grab a bike' is a glance. Hover/tap a station
    for its name + counts.
    """
    df = _stations_with_ebikes(lat, lon, radius_km)
    if df.empty:
        st.info(f"No stations with available ebikes within {radius_km:.1f} km.")
        return

    plot_df = df.copy()
    plot_df["radius_m"] = (plot_df["ebikes_available"].astype(float) * 12.0 + 35.0).clip(35, 180)
    plot_df["distance_label"] = plot_df["distance_km"].apply(
        lambda d: f"{d*1000:.0f} m" if d < 0.95 else f"{d:.2f} km"
    )
    plot_df["capacity_str"] = plot_df["capacity"].fillna(0).astype(int).astype(str)

    station_layer = pdk.Layer(
        "ScatterplotLayer",
        id="stations",
        data=plot_df,
        get_position="[lon, lat]",
        get_radius="radius_m",
        get_fill_color=[238, 137, 0, 205],
        get_line_color=[35, 35, 35, 255],
        line_width_min_pixels=2,
        pickable=True,
        auto_highlight=True,
        radius_min_pixels=4,
        radius_max_pixels=40,
    )

    you_layer = pdk.Layer(
        "ScatterplotLayer",
        id="you",
        data=pd.DataFrame({"lat": [lat], "lon": [lon], "label": ["You"]}),
        get_position="[lon, lat]",
        get_radius=45,
        get_fill_color=[20, 20, 20, 235],
        get_line_color=[255, 255, 255, 255],
        line_width_min_pixels=2,
        radius_min_pixels=8,
        radius_max_pixels=14,
        stroked=True,
        filled=True,
        pickable=False,
    )

    tooltip = {
        "html": (
            "<b>{name}</b><br/>"
            "🚴‍♂️ <b>{ebikes_available}</b> ebikes · {classic_bikes} regular<br/>"
            "{distance_label} away · cap {capacity_str}<br/>"
            "<i>click to drill in</i>"
        ),
        "style": {"backgroundColor": "rgba(30,30,30,0.85)", "color": "white", "fontSize": "12px"},
    }

    deck = pdk.Deck(
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=14, pitch=0),
        layers=[station_layer, you_layer],
        tooltip=tooltip,
    )
    event = st.pydeck_chart(
        deck,
        use_container_width=True,
        selection_mode="single-object",
        on_select="rerun",
        key="station_map",
    )
    st.caption(
        f"Black = you · amber = station (size proportional to ebikes available) · "
        f"{len(plot_df)} station{'s' if len(plot_df) != 1 else ''} with ebikes within {radius_km:.1f} km. "
        "Tap a marker to drill in."
    )

    clicked = _picked_object(event, layer_id="stations")
    if clicked:
        _render_station_click_callout(clicked)


def _picked_object(event, layer_id: str) -> dict | None:
    """Extract the single clicked object for `layer_id` from a pydeck event.

    Streamlit's pydeck event shape varies slightly across versions, so we
    accept both attribute and dict access.
    """
    if event is None:
        return None
    selection = getattr(event, "selection", None) or (event.get("selection") if isinstance(event, dict) else None)
    if not selection:
        return None
    objs = getattr(selection, "objects", None) or (selection.get("objects") if isinstance(selection, dict) else None)
    if not objs:
        return None
    items = objs.get(layer_id) if isinstance(objs, dict) else None
    if not items:
        return None
    return items[0]


def _render_tile_drilldown(picked: dict, state_df: pd.DataFrame, score_df: pd.DataFrame) -> None:
    """For a clicked H3 tile, show its current bikes, stations, and per-horizon forecasts."""
    tile_id = picked.get("tile_id")
    if not tile_id:
        st.caption("Couldn't resolve tile from click event.")
        return
    tile_row = state_df[state_df["tile_id"] == tile_id]
    if tile_row.empty:
        st.caption("Tile not found in current view.")
        return
    row = tile_row.iloc[0]
    tile_scores = score_df[score_df["tile_id"] == tile_id].sort_values("horizon_minutes")

    with st.container(border=True):
        st.markdown(f"**Tile {tile_id}**")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Free now", int(row["current_free_ebikes"]))
        c2.metric("Reserved now", int(row["current_reserved_free_ebikes"]))
        c3.metric("Docked now", int(row["current_docked_ebikes"]))
        c4.metric("Stations", int(row["n_stations_in_tile"]))
        c5.metric(
            "Fresh res. (30m)",
            int(row["reservation_events_30m"]),
            delta=int(row["reservation_events_5m"]) or None,
            delta_color="off" if not int(row["reservation_events_5m"]) else "normal",
            help="Bikes newly reserved in this tile (false → true transitions). Delta shows the last 5 minutes.",
        )

        if not tile_scores.empty:
            chart_data = pd.DataFrame({
                "horizon (min)": tile_scores["horizon_minutes"].astype(int),
                "Free expected": tile_scores["free_expected_count"].astype(float),
                "Docked expected": tile_scores["dock_expected_count"].astype(float),
            })
            melted = chart_data.melt(
                id_vars="horizon (min)", var_name="source", value_name="expected ebikes"
            )
            chart = (
                alt.Chart(melted)
                .mark_bar()
                .encode(
                    x=alt.X("horizon (min):O"),
                    y=alt.Y("expected ebikes:Q", stack="zero"),
                    color="source:N",
                    tooltip=["horizon (min):O", "source:N", alt.Tooltip("expected ebikes:Q", format=".2f")],
                )
                .properties(height=180)
            )
            st.altair_chart(chart, use_container_width=True)

            prob_table = pd.DataFrame({
                "horizon (min)": tile_scores["horizon_minutes"].astype(int),
                "P(free has bike)": tile_scores["free_p_has_bike"].astype(float),
                "P(any dock has bike)": tile_scores["dock_p_any_has_ebike"].astype(float),
                "P(any ebike anywhere)": tile_scores["combined_p_any_ebike"].astype(float),
            })
            st.dataframe(prob_table, hide_index=True, use_container_width=True)

        bikes = list(row["bikes"]) if row["bikes"] is not None else []
        if bikes:
            st.markdown("**Free ebikes here**")
            bike_table = pd.DataFrame([
                {
                    "bike_id": b["bike_id"],
                    "dwell (min)": round(b["dwell_seconds_so_far"] / 60.0, 1),
                    "P(stays 5m)": b["p_stays"].get(5),
                    "P(stays 10m)": b["p_stays"].get(10),
                    "P(stays 15m)": b["p_stays"].get(15),
                    "P(stays 20m)": b["p_stays"].get(20),
                }
                for b in bikes
            ])
            st.dataframe(bike_table, hide_index=True, use_container_width=True)

        reserved = list(row.get("reserved_bikes") or [])
        if reserved:
            st.markdown("**Reserved ebikes here** (rider has held but not started)")
            res_table = pd.DataFrame([
                {
                    "bike_id": b["bike_id"],
                    "in tile (min)": round(b["dwell_seconds_so_far"] / 60.0, 1),
                }
                for b in reserved
            ])
            st.dataframe(res_table, hide_index=True, use_container_width=True)
            st.caption(
                "Note: 'in tile' is time since the bike last entered this hex — usually close to time-since-reservation "
                "but not exact (a bike can sit free for hours and then get reserved)."
            )

        stations = list(row["stations"]) if row["stations"] is not None else []
        if stations:
            st.markdown("**Stations in this tile**")
            station_table = pd.DataFrame([
                {
                    "name": s.get("name"),
                    "current ebikes": s.get("num_ebikes_available"),
                    "docks available": s.get("num_docks_available"),
                    "capacity": s.get("capacity"),
                }
                for s in stations
            ])
            st.dataframe(station_table, hide_index=True, use_container_width=True)


def _render_hex_click_drilldown(obj: dict, hours: int) -> None:
    """Show the bike events that fall inside the clicked hex.

    HexagonLayer's click event carries `points` (the rows aggregated into
    the hex). We derive the centroid from those points rather than trusting
    `position` (which pydeck/deck.gl reports inconsistently for aggregated
    layers — it can be a screen coord, an aggregated value, or absent).
    """
    points = obj.get("points") or []
    n_events = obj.get("elevationValue") or len(points)

    # Pull lat/lon from the source rows that landed in this hex.
    lats, lons = [], []
    for p in points:
        src = p.get("source") if isinstance(p, dict) else None
        if isinstance(src, dict):
            if "lat" in src and "lon" in src:
                lats.append(float(src["lat"]))
                lons.append(float(src["lon"]))
        # Some deck versions put coords directly on the point
        if isinstance(p, dict) and "lat" in p and "lon" in p and not isinstance(p.get("source"), dict):
            lats.append(float(p["lat"]))
            lons.append(float(p["lon"]))

    if not lats:
        st.caption("Couldn't resolve hex location from click event.")
        return

    lat = sum(lats) / len(lats)
    lon = sum(lons) / len(lons)

    # Hex radius is 80m; pad a bit to cover the corners.
    radius_m = 100.0
    lat_delta = radius_m / 111_000.0
    lon_delta = radius_m / 85_000.0
    bikes = _free_bikes_in_box(
        lat - lat_delta, lat + lat_delta, lon - lon_delta, lon + lon_delta, hours
    )

    with st.container(border=True):
        st.markdown(
            f"**Hex at ({lat:.4f}, {lon:.4f})** — "
            f"{int(n_events):,} position events over the last "
            f"{hours // 24}d"
        )
        if bikes.empty:
            st.caption("No detailed events found for this cell.")
            return

        c1, c2, c3 = st.columns(3)
        c1.metric("Position events", f"{len(bikes):,}")
        c2.metric("Distinct bikes", f"{bikes['bike_id'].nunique():,}")
        last_seen = bikes["fetched_at"].max()
        c3.metric(
            "Most recent",
            last_seen.strftime("%m-%d %H:%M") if pd.notna(last_seen) else "—",
        )
        st.dataframe(
            bikes[["name", "fetched_at", "lat", "lon", "bike_id"]],
            hide_index=True,
            height=240,
            column_config={
                "name": "Bike #",
                "fetched_at": st.column_config.DatetimeColumn("Seen at (UTC)"),
                "lat": st.column_config.NumberColumn("Lat", format="%.5f"),
                "lon": st.column_config.NumberColumn("Lon", format="%.5f"),
                "bike_id": "bike_id",
            },
        )


def _render_station_click_callout(obj: dict) -> None:
    """Show a quick summary for the clicked station and stash it for the
    Station detail section below.
    """
    sid = obj.get("station_id")
    name = obj.get("name") or "Selected station"
    if sid:
        st.session_state["selected_station"] = sid

    eb = obj.get("ebikes_available")
    cl = obj.get("classic_bikes")
    dist = obj.get("distance_label", "")
    cap = obj.get("capacity_str", "")

    with st.container(border=True):
        st.markdown(f"**📍 {name}**  ·  *click registered — full detail loaded below ↓*")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Ebikes", int(eb) if eb is not None else "—")
        c2.metric("Regular", int(cl) if cl is not None else "—")
        c3.metric("Distance", dist or "—")
        c4.metric("Capacity", cap or "—")


def _collector_health_section() -> None:
    health = _collector_health()
    if health.get("error"):
        st.warning(f"Collector health unavailable: {health['error']}")
        return

    latest_tick = health.get("latest_tick_at")
    tick_age = _age_minutes(latest_tick)
    replica_age = _age_minutes(health.get("read_replica_updated_at"))
    status = "fresh" if tick_age is not None and tick_age <= 3 else "stale"
    with st.container(border=True):
        st.markdown("**Data ingestion**")
        cols = st.columns(5)
        cols[0].metric("Collector", status, help="Fresh means a successful collector tick in the last 3 minutes.")
        cols[1].metric("Latest tick", _minutes_label(tick_age))
        cols[2].metric("Ticks 10m", int(health.get("ticks_10m") or 0))
        cols[3].metric("Station rows 10m", f"{int(health.get('station_rows_10m') or 0):,}")
        cols[4].metric("Free-bike events 10m", f"{int(health.get('free_events_10m') or 0):,}")

        detail_cols = st.columns(5)
        detail_cols[0].metric("Station payload", f"{int(health.get('station_payload_count') or 0):,}")
        detail_cols[1].metric("Free-bike payload", f"{int(health.get('free_bike_payload_count') or 0):,}")
        detail_cols[2].metric("Replica age", _minutes_label(replica_age))
        detail_cols[3].metric("Forecast queue", f"{int(health.get('pending_queue_files') or 0)} pending")
        detail_cols[4].metric("Resolved 10m", f"{int(health.get('outcomes_10m') or 0):,}")

        latest_fetch = health.get("latest_station_fetch") or health.get("latest_station_reported")
        free_fetch = health.get("latest_free_bike_fetch")
        if latest_fetch or free_fetch:
            st.caption(
                f"Latest station fetch: {_minutes_label(_age_minutes(latest_fetch))} ago · "
                f"latest free-bike event: {_minutes_label(_age_minutes(free_fetch))} ago · "
                f"failed queue files: {int(health.get('failed_queue_files') or 0)}"
            )


def _system_status_section() -> None:
    try:
        payload = _system_status_payload()
    except Exception as exc:
        st.warning(f"System status unavailable: {exc}")
        return
    status = payload.get("status") or {}
    selection = payload.get("selection") or {}
    cache = payload.get("cache") or status.get("prediction_cache") or {}
    queue = status.get("queue") or {}
    with st.container(border=True):
        st.markdown("**Prediction system**")
        cols = st.columns(5)
        cols[0].metric("Active driver", selection.get("active_model_key") or "-")
        cols[0].caption(selection.get("active_model_source") or "-")
        cols[1].metric("Best 7d SOTA", selection.get("best_sota_model_key") or "-")
        cols[1].caption(selection.get("reason") or "-")
        cols[2].metric("Active equals best", str(selection.get("active_equals_best")))
        cols[2].caption(selection.get("selection_metric") or "decision_rank_loss")
        cols[3].metric("Cache", cache.get("status") or "-")
        cols[3].caption(f"{_minutes_label(cache.get('age_minutes'))} · {int(cache.get('stations_cached') or 0):,} stations")
        cols[4].metric("Queue", f"{int(queue.get('pending_files') or 0)} pending")
        cols[4].caption(f"{int(queue.get('failed_files') or 0)} failed")

        jobs = pd.DataFrame(status.get("jobs") or [])
        if not jobs.empty:
            with st.expander("Background jobs", expanded=False):
                keep = ["job_name", "status", "started_at", "finished_at", "duration_seconds", "message", "error"]
                st.dataframe(jobs[[c for c in keep if c in jobs.columns]], hide_index=True)
        locks = pd.DataFrame(status.get("locks") or [])
        if not locks.empty:
            stale = locks[locks.get("stale", False) == True] if "stale" in locks else pd.DataFrame()  # noqa: E712
            if not stale.empty:
                st.warning(f"{len(stale)} stale background locks are recorded.")


def _free_bike_density_section(lat: float, lon: float, radius_km: float = 2.0) -> None:
    """Tiled view: free-ebike persistence + dock predictions per H3 hex.

    Each H3 res-9 hex is colored by ``1 - p_any_ebike_10m`` (red = likely no
    ebike anywhere in this hex 10 min from now). Hex elevation shows the
    combined current count (free + docked). Free ebikes appear as dots colored
    by their per-bike survival probability; stations show as larger markers.
    """
    state_df, score_df = _tile_state_for_area(lat, lon, radius_km)
    if state_df.empty:
        st.info(
            "No tile data yet for this area. Let the poller run for a few hours; "
            "this view shows current ebike availability and where it's likely to persist."
        )
        return

    horizon_focus = 10
    score_10 = score_df[score_df["horizon_minutes"] == horizon_focus]
    color_by_tile = {
        str(r["tile_id"]): float(r["combined_p_any_ebike"])
        for _, r in score_10.iterrows()
    }

    hex_rows = []
    bike_rows = []
    reserved_bike_rows = []
    station_rows = []
    for _, row in state_df.iterrows():
        tid = row["tile_id"]
        p_any = color_by_tile.get(tid, 0.0)
        # 1 - p means "P(no ebike anywhere in this hex)" — bright red when high.
        bad = max(0.0, min(1.0, 1.0 - p_any))
        hex_rows.append({
            "tile_id": tid,
            "current_free_ebikes": int(row["current_free_ebikes"]),
            "current_reserved_free_ebikes": int(row["current_reserved_free_ebikes"]),
            "current_docked_ebikes": int(row["current_docked_ebikes"]),
            "current_total_ebikes": int(row["current_free_ebikes"] + row["current_docked_ebikes"]),
            "n_stations": int(row["n_stations_in_tile"]),
            "churn_5m": int(row["churn_rate_5m"]),
            "reservation_events_5m": int(row["reservation_events_5m"]),
            "reservation_events_30m": int(row["reservation_events_30m"]),
            "p_any_ebike_10m": float(p_any),
            "p_no_ebike_10m": bad,
            "color_r": int(round(40 + 215 * bad)),
            "color_g": int(round(180 * (1.0 - bad))),
            "color_b": int(round(60 * (1.0 - bad))),
        })
        for bike in row["bikes"]:
            p_stays_10 = float(bike["p_stays"].get(horizon_focus, 0.0))
            dwell_min = bike["dwell_seconds_so_far"] / 60.0
            bike_rows.append({
                "bike_id": bike["bike_id"],
                "lat": bike["lat"],
                "lon": bike["lon"],
                "dwell_minutes": dwell_min,
                "p_stays_10m": p_stays_10,
                "color_r": int(round(220 * (1.0 - p_stays_10))),
                "color_g": int(round(200 * p_stays_10)),
                "color_b": 60,
            })
        for bike in row.get("reserved_bikes", []) or []:
            reserved_bike_rows.append({
                "bike_id": bike["bike_id"],
                "lat": bike["lat"],
                "lon": bike["lon"],
                "dwell_minutes": bike["dwell_seconds_so_far"] / 60.0,
            })
        for st_meta in row["stations"]:
            station_rows.append({
                "station_id": st_meta["station_id"],
                "name": st_meta["name"],
                "lat": st_meta["lat"],
                "lon": st_meta["lon"],
                "num_ebikes_available": int(st_meta["num_ebikes_available"]),
            })

    hex_df = pd.DataFrame(hex_rows)
    bikes_df = pd.DataFrame(bike_rows)
    reserved_bikes_df = pd.DataFrame(reserved_bike_rows)
    stations_df = pd.DataFrame(station_rows)

    tile_layer = pdk.Layer(
        "H3HexagonLayer",
        id="tile_hexes",
        data=hex_df,
        get_hexagon="tile_id",
        get_fill_color="[color_r, color_g, color_b, 160]",
        get_elevation="current_total_ebikes",
        elevation_scale=40,
        extruded=True,
        coverage=0.92,
        pickable=True,
        auto_highlight=True,
    )

    layers = [tile_layer]
    if not bikes_df.empty:
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            id="free_bikes",
            data=bikes_df,
            get_position="[lon, lat]",
            get_radius=18,
            get_fill_color="[color_r, color_g, color_b, 220]",
            radius_min_pixels=4,
            pickable=True,
            auto_highlight=True,
        ))
    if not reserved_bikes_df.empty:
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            id="reserved_bikes",
            data=reserved_bikes_df,
            get_position="[lon, lat]",
            get_radius=18,
            get_fill_color=[180, 100, 230, 230],  # purple = reserved
            radius_min_pixels=4,
            pickable=True,
            auto_highlight=True,
        ))
    if not stations_df.empty:
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            id="stations",
            data=stations_df,
            get_position="[lon, lat]",
            get_radius=35,
            get_fill_color=[31, 119, 180, 220],
            radius_min_pixels=6,
            pickable=True,
            auto_highlight=True,
        ))
    layers.append(pdk.Layer(
        "ScatterplotLayer",
        id="you_hex",
        data=pd.DataFrame({"lat": [lat], "lon": [lon]}),
        get_position="[lon, lat]",
        get_radius=40,
        get_fill_color=[0, 0, 0, 230],
        radius_min_pixels=6,
    ))

    deck = pdk.Deck(
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=14, pitch=40),
        layers=layers,
        tooltip={
            "html": (
                "<b>tile</b> {tile_id}<br/>"
                "free now: {current_free_ebikes} · reserved now: {current_reserved_free_ebikes} · docked now: {current_docked_ebikes}<br/>"
                "stations in tile: {n_stations} · 5-min churn events: {churn_5m}<br/>"
                "fresh reservations 5m / 30m: {reservation_events_5m} / {reservation_events_30m}<br/>"
                "P(any ebike 10 min): {p_any_ebike_10m}"
            ),
            "style": {"font-size": "12px"},
        },
    )
    event = st.pydeck_chart(
        deck,
        use_container_width=True,
        selection_mode="single-object",
        on_select="rerun",
        key="tile_map",
    )

    n_free = int(hex_df["current_free_ebikes"].sum())
    n_reserved = int(hex_df["current_reserved_free_ebikes"].sum())
    n_docked = int(hex_df["current_docked_ebikes"].sum())
    st.caption(
        f"H3 res-9 hexes (~174 m edge). Color: red = likely no ebike anywhere in this hex 10 min from now, "
        f"green = likely yes. Elevation = current ebike count (free + docked). "
        f"Dots: free ebikes colored by per-bike survival probability. Purple dots: currently reserved. "
        f"Now: {n_free} free, {n_reserved} reserved, {n_docked} docked across {len(hex_df)} hexes. "
        f"Divvy GBFS doesn't expose docked reservation counts, so reservation stats cover free-floating only."
    )

    # Reservation hotspot leaderboard — most fresh reservations in the last 30 min.
    hotspots = hex_df.sort_values("reservation_events_30m", ascending=False).head(5)
    hotspots = hotspots[hotspots["reservation_events_30m"] > 0]
    if not hotspots.empty:
        st.markdown("**Hot reservation hotspots (last 30 min)**")
        st.dataframe(
            hotspots[[
                "tile_id",
                "current_free_ebikes",
                "current_reserved_free_ebikes",
                "reservation_events_5m",
                "reservation_events_30m",
            ]].rename(columns={
                "current_free_ebikes": "free now",
                "current_reserved_free_ebikes": "reserved now",
                "reservation_events_5m": "new res (5m)",
                "reservation_events_30m": "new res (30m)",
            }),
            hide_index=True,
            use_container_width=True,
        )

    picked_tile = _picked_object(event, layer_id="tile_hexes")
    if picked_tile:
        _render_tile_drilldown(picked_tile, state_df, score_df)

    by_hour = _free_bike_count_by_hour(168)
    if not by_hour.empty:
        st.markdown("**Citywide free-bike count by hour of day (last 7d avg)**")
        chart = (
            alt.Chart(by_hour)
            .mark_bar()
            .encode(
                x=alt.X("hour_of_day:O", title="Hour (local)"),
                y=alt.Y("avg_bikes:Q", title="Avg free-floating ebikes"),
                tooltip=[
                    alt.Tooltip("hour_of_day:O", title="Hour"),
                    alt.Tooltip("avg_bikes:Q", title="Avg", format=".0f"),
                    alt.Tooltip("min_bikes:Q", title="Min"),
                    alt.Tooltip("max_bikes:Q", title="Max"),
                    alt.Tooltip("n_hours_observed:Q", title="Samples"),
                ],
            )
            .properties(height=200)
        )
        st.altair_chart(chart, use_container_width=True)


def _maintenance_map_section(lat: float, lon: float, radius_km: float = 2.0) -> None:
    """Tiled maintenance view: disabled bikes per H3 hex with repair-time forecasts.

    Each H3 res-9 hex is colored by the total current disabled count
    (free + docked) and sized by accumulated bike-hours of disability over the
    last 24h. Disabled free bikes appear as orange dots; the top-disability
    hexes are called out as an "ops hotspot" leaderboard.
    """
    state_df, score_df = _disabled_tile_state_for_area(lat, lon, radius_km)
    if state_df.empty:
        st.info(
            "No maintenance tile data yet for this area. Let the poller run for "
            "a few hours; this view shows where bikes are currently flagged as "
            "needing repair and where operator attention will likely be required."
        )
        return

    # P(any repair) at 6h horizon for tooltip; keep 24h as the slower band.
    score_6h = score_df[score_df["horizon_hours"] == 6.0]
    p6_by_tile = {
        str(r["tile_id"]): float(r["p_any_repair"]) for _, r in score_6h.iterrows()
    }
    score_24h = score_df[score_df["horizon_hours"] == 24.0]
    p24_by_tile = {
        str(r["tile_id"]): float(r["p_any_repair"]) for _, r in score_24h.iterrows()
    }

    max_total_disabled = max(
        int(r["current_disabled_free_ebikes"] + r["current_disabled_docked_bikes"])
        for _, r in state_df.iterrows()
    )
    max_total_disabled = max(1, max_total_disabled)
    max_bike_hours = max(float(r["bike_hours_disabled_24h"]) for _, r in state_df.iterrows())
    max_bike_hours = max(1.0, max_bike_hours)

    hex_rows = []
    disabled_bike_rows = []
    disabled_station_rows = []
    for _, row in state_df.iterrows():
        tid = row["tile_id"]
        total_disabled = int(row["current_disabled_free_ebikes"] + row["current_disabled_docked_bikes"])
        # Color: pale grey at 0 → deep orange at the area's max.
        intensity = min(1.0, total_disabled / max_total_disabled)
        hex_rows.append({
            "tile_id": tid,
            "current_disabled_free_ebikes": int(row["current_disabled_free_ebikes"]),
            "current_disabled_docked_bikes": int(row["current_disabled_docked_bikes"]),
            "current_disabled_docks": int(row["current_disabled_docks"]),
            "current_total_disabled": total_disabled,
            "bike_hours_disabled_24h": float(row["bike_hours_disabled_24h"]),
            "disability_events_24h": int(row["disability_events_24h"]),
            "repair_events_24h": int(row["repair_events_24h"]),
            "median_free_dwell_minutes": float(row["median_free_dwell_minutes"]),
            "p_any_repair_6h": p6_by_tile.get(tid, 0.0),
            "p_any_repair_24h": p24_by_tile.get(tid, 0.0),
            "color_r": int(round(220 * intensity + 35 * (1 - intensity))),
            "color_g": int(round(130 * intensity + 35 * (1 - intensity))),
            "color_b": int(round(40 * intensity + 35 * (1 - intensity))),
        })
        for bike in row["disabled_free_bikes"]:
            disabled_bike_rows.append({
                "bike_id": bike["bike_id"],
                "lat": bike["lat"],
                "lon": bike["lon"],
                "dwell_minutes": bike["dwell_seconds_so_far"] / 60.0,
            })
        for st_meta in row["disabled_stations"]:
            if int(st_meta.get("num_bikes_disabled") or 0) > 0 or int(st_meta.get("num_docks_disabled") or 0) > 0:
                disabled_station_rows.append({
                    "station_id": st_meta["station_id"],
                    "name": st_meta["name"],
                    "lat": st_meta["lat"],
                    "lon": st_meta["lon"],
                    "num_bikes_disabled": int(st_meta["num_bikes_disabled"]),
                    "num_docks_disabled": int(st_meta["num_docks_disabled"]),
                })

    hex_df = pd.DataFrame(hex_rows)
    disabled_bikes_df = pd.DataFrame(disabled_bike_rows)
    disabled_stations_df = pd.DataFrame(disabled_station_rows)

    tile_layer = pdk.Layer(
        "H3HexagonLayer",
        id="disabled_tile_hexes",
        data=hex_df,
        get_hexagon="tile_id",
        get_fill_color="[color_r, color_g, color_b, 160]",
        # Scale elevation by bike-hours of disability over the last 24h.
        get_elevation="bike_hours_disabled_24h",
        elevation_scale=20,
        extruded=True,
        coverage=0.92,
        pickable=True,
        auto_highlight=True,
    )

    layers = [tile_layer]
    if not disabled_bikes_df.empty:
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            id="disabled_free_bikes",
            data=disabled_bikes_df,
            get_position="[lon, lat]",
            get_radius=20,
            get_fill_color=[230, 110, 20, 230],  # orange = disabled free bike
            radius_min_pixels=5,
            pickable=True,
            auto_highlight=True,
        ))
    if not disabled_stations_df.empty:
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            id="disabled_stations",
            data=disabled_stations_df,
            get_position="[lon, lat]",
            get_radius=35,
            get_fill_color=[180, 60, 30, 220],  # darker red = station with disabled bikes
            radius_min_pixels=6,
            pickable=True,
            auto_highlight=True,
        ))
    layers.append(pdk.Layer(
        "ScatterplotLayer",
        id="you_maintenance_hex",
        data=pd.DataFrame({"lat": [lat], "lon": [lon]}),
        get_position="[lon, lat]",
        get_radius=40,
        get_fill_color=[0, 0, 0, 230],
        radius_min_pixels=6,
    ))

    deck = pdk.Deck(
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=14, pitch=40),
        layers=layers,
        tooltip={
            "html": (
                "<b>tile</b> {tile_id}<br/>"
                "disabled now: {current_total_disabled} "
                "({current_disabled_free_ebikes} free + {current_disabled_docked_bikes} docked)<br/>"
                "disabled docks: {current_disabled_docks}<br/>"
                "bike-hours disabled (24h): {bike_hours_disabled_24h}<br/>"
                "disability / repair events (24h): {disability_events_24h} / {repair_events_24h}<br/>"
                "P(any repair within 6h): {p_any_repair_6h}<br/>"
                "P(any repair within 24h): {p_any_repair_24h}"
            ),
            "style": {"font-size": "12px"},
        },
    )
    st.pydeck_chart(deck, use_container_width=True, key="disabled_tile_map")

    n_free_disabled = int(hex_df["current_disabled_free_ebikes"].sum())
    n_docked_disabled = int(hex_df["current_disabled_docked_bikes"].sum())
    n_disabled_docks = int(hex_df["current_disabled_docks"].sum())
    total_bike_hours = float(hex_df["bike_hours_disabled_24h"].sum())
    st.caption(
        f"H3 res-9 hexes (~174 m edge). Color: deeper orange = more disabled bikes right now. "
        f"Elevation = bike-hours of disability accumulated over the last 24h. "
        f"Orange dots: free-floating bikes currently flagged disabled. "
        f"Red dots: stations with disabled bikes or docks. "
        f"Now: {n_free_disabled} free + {n_docked_disabled} docked disabled, "
        f"{n_disabled_docks} disabled docks, "
        f"{total_bike_hours:.0f} bike-hours of disability accumulated in 24h across "
        f"{len(hex_df)} hexes. Divvy doesn't expose individual bike_ids for docked disability, "
        f"so dwell times cover free-floating only."
    )

    # Ops hotspot leaderboard — the highest-disability hexes by bike-hours.
    hotspots = hex_df.sort_values("bike_hours_disabled_24h", ascending=False).head(5)
    hotspots = hotspots[hotspots["bike_hours_disabled_24h"] > 0]
    if not hotspots.empty:
        st.markdown("**Ops hotspots — most accumulated disability (last 24h)**")
        st.dataframe(
            hotspots[[
                "tile_id",
                "current_total_disabled",
                "current_disabled_free_ebikes",
                "current_disabled_docked_bikes",
                "bike_hours_disabled_24h",
                "disability_events_24h",
                "repair_events_24h",
                "p_any_repair_6h",
            ]].rename(columns={
                "current_total_disabled": "disabled now",
                "current_disabled_free_ebikes": "free disabled",
                "current_disabled_docked_bikes": "docked disabled",
                "bike_hours_disabled_24h": "bike·h 24h",
                "disability_events_24h": "broke 24h",
                "repair_events_24h": "fixed 24h",
                "p_any_repair_6h": "P(repair 6h)",
            }),
            hide_index=True,
            use_container_width=True,
        )


def _home_section() -> None:
    """Landing section: 3 closest stations to the user's browser location.

    Each card shows current ebikes (and number of regular bikes), plus the
    average daily ebike range from the last 7 days of observations.
    """
    st.subheader("Nearest stations")
    st.caption("Click the 📍 button to share your location and see the 3 closest stations.")

    loc = streamlit_geolocation()

    # streamlit_geolocation returns {'latitude': None, 'longitude': None, ...}
    # before the user has clicked the button, then real values after.
    if loc and loc.get("latitude") is not None and loc.get("longitude") is not None:
        st.session_state["user_location"] = (float(loc["latitude"]), float(loc["longitude"]))
        st.session_state["user_location_source"] = "browser"
        st.session_state["user_location_accuracy_m"] = loc.get("accuracy")
        st.session_state.pop("user_location_label", None)

    # Address fallback — useful when permission denied, or you want to plan from
    # somewhere other than where you're sitting.
    with st.expander("Or enter an address", expanded=False):
        with st.form("address_form", clear_on_submit=False):
            addr = st.text_input(
                "Address or place",
                placeholder="e.g. 1060 W Addison St  ·  Millennium Park  ·  Wicker Park",
            )
            submitted = st.form_submit_button("Find")
        if submitted and addr.strip():
            try:
                result = _geocode(addr)
            except requests.RequestException as exc:
                st.error(f"Geocoding failed: {exc}")
            else:
                if result is None:
                    st.error(f"No match for {addr!r}. Try adding a neighborhood or ZIP.")
                else:
                    lat_, lon_, label = result
                    st.session_state["user_location"] = (lat_, lon_)
                    st.session_state["user_location_source"] = "address"
                    st.session_state["user_location_label"] = label
                    st.session_state.pop("user_location_accuracy_m", None)
                    st.rerun()

    user_loc = st.session_state.get("user_location")
    if user_loc is None:
        st.info(
            "Waiting for a location. On mobile, browser GPS requires an HTTPS page "
            "or localhost; plain http://<computer-ip>:8501 usually cannot request it."
        )
        return

    lat, lon = user_loc
    label = st.session_state.get("user_location_label")
    if label:
        st.caption(f"📍 {label}  ·  ({lat:.4f}, {lon:.4f})")
    else:
        accuracy = st.session_state.get("user_location_accuracy_m")
        if isinstance(accuracy, (int, float)):
            st.caption(f"📍 {lat:.4f}, {lon:.4f}  ·  accuracy ~{accuracy:.0f} m")
        else:
            st.caption(f"📍 {lat:.4f}, {lon:.4f}")

    _prediction_service_section(lat, lon)

    # === At-a-glance map: stations with ebikes ===
    st.markdown("### Where can I grab a bike right now?")
    _station_map(lat, lon, radius_km=2.0)

    nearest = _nearest(lat, lon, 3)
    if nearest.empty:
        st.warning("No stations with location data in the DB yet — has the poller run?")
        return

    sids = tuple(nearest["station_id"].tolist())
    latest = _latest_status(sids).set_index("station_id") if sids else pd.DataFrame()

    end_dt = datetime.now(timezone.utc).replace(tzinfo=None)
    start_dt = end_dt - timedelta(days=7)

    cols = st.columns(len(nearest))
    for col, (_, row) in zip(cols, nearest.iterrows()):
        sid = row["station_id"]
        with col:
            st.markdown(f"**{row['name']}**")
            cap_str = f"cap {int(row['capacity'])}" if pd.notna(row["capacity"]) else "cap —"
            st.caption(f"{row['distance_km']:.2f} km away · {cap_str}")

            if sid in latest.index:
                r = latest.loc[sid]
                eb = int(r["num_ebikes_available"] or 0)
                cl = int(r["num_classic_bikes"] or 0)
                st.metric("Ebikes now", eb, help=f"{cl} regular bikes also available")
                st.caption(f"{eb} ebikes · {cl} regular")
            else:
                st.metric("Ebikes now", "—", help="No live status yet for this station.")

            sm = _ebike_summary(sid, start_dt, end_dt)
            if sm and (sm.get("n_days") or 0) > 0:
                st.metric(
                    "Avg daily ebike range (7d)",
                    f"{float(sm['avg_daily_range']):.1f}",
                    help="Mean of (daily max − daily min) ebike count over the last 7 days.",
                )
            else:
                st.metric("Avg daily ebike range (7d)", "—", help="Need ≥2 obs/day; let the poller run longer.")

            if st.button("Drill into details →", key=f"drill_{sid}"):
                st.session_state["selected_station"] = sid
                st.rerun()

    # === Free-floating ebikes ===
    st.markdown("### Free-floating ebikes nearby")
    try:
        bikes = _free_bikes()
    except requests.RequestException as exc:
        st.warning(f"Couldn't fetch free-floating bikes: {exc}")
        return

    if bikes.empty:
        st.info("No free-floating ebikes currently reported.")
        return

    bikes = bikes.assign(distance_km=_haversine_km(bikes, lat, lon))
    within_1km = bikes[bikes["distance_km"] <= 1.0].sort_values("distance_km")
    closest = bikes.sort_values("distance_km").head(5)

    if within_1km.empty:
        nearest_one = closest.iloc[0]
        st.info(
            f"No free-floating ebikes within 1 km. "
            f"Closest is **{nearest_one['name']}** at {nearest_one['distance_km']:.2f} km."
        )
        show = closest
        st.caption(f"5 closest of {len(bikes):,} citywide:")
    else:
        show = within_1km.head(5)
        extra = max(0, len(within_1km) - 5)
        more = f"  (+{extra} more within 1 km)" if extra else ""
        st.caption(
            f"**{len(within_1km)}** free-floating ebike{'s' if len(within_1km) != 1 else ''} "
            f"within 1 km · showing {len(show)} closest{more}"
        )

    bike_cols = st.columns(min(5, len(show)))
    for col, (_, b) in zip(bike_cols, show.iterrows()):
        with col:
            dist_m = b["distance_km"] * 1000
            dist_str = f"{dist_m:.0f} m" if dist_m < 950 else f"{b['distance_km']:.2f} km"
            st.metric(f"#{b['name']}", dist_str, help=f"bike_id {b['bike_id'][:10]}…")

    with st.expander("Show on map"):
        user_pt = pd.DataFrame({
            "lat": [lat], "lon": [lon],
            "color": ["#1f77b4"], "size": [25],
        })
        bike_pts = show[["lat", "lon"]].assign(color="#d62728", size=10)
        map_df = pd.concat([user_pt, bike_pts], ignore_index=True)
        st.map(map_df, color="color", size="size", zoom=15)
        st.caption("Blue = you · red = ebikes")

    # === Where do free bikes typically cluster? ===
    with st.expander("Where do free bikes typically cluster? (7-day density)"):
        _free_bike_density_section(lat, lon, radius_km=2.0)

    # === Maintenance map ===
    with st.expander("Maintenance map — where are disabled bikes accumulating?"):
        _maintenance_map_section(lat, lon, radius_km=2.0)


def main() -> None:
    st.set_page_config(page_title="Divvy Observer", layout="wide")
    st.title("Divvy Observer")
    read_path = config.READ_DB_PATH if config.ENABLE_READ_REPLICA and config.READ_DB_PATH.exists() else config.DB_PATH
    st.caption(f"Reading {read_path}")
    _collector_health_section()
    _system_status_section()

    _home_section()
    st.divider()
    st.subheader("Station detail")

    try:
        stations_df = _stations()
    except duckdb.IOException as exc:
        st.error(
            "Could not open the DuckDB file. The poller may be writing — try again in a moment.\n\n"
            f"{exc}"
        )
        return
    if stations_df.empty:
        st.warning("No stations yet. Make sure the poller has run at least once.")
        return

    with st.sidebar:
        st.header("Station")
        search = st.text_input("Search by name", "")
        filtered = (
            stations_df[stations_df["name"].str.contains(search, case=False, na=False)]
            if search
            else stations_df
        )
        st.caption(f"{len(filtered)} of {len(stations_df)} stations match")
        if filtered.empty:
            st.warning("No matches.")
            return

        labels = {row["station_id"]: _format_station(row) for _, row in filtered.iterrows()}
        options = filtered["station_id"].tolist()
        default_sid = st.session_state.get("selected_station")
        default_index = options.index(default_sid) if default_sid in options else 0
        sid = st.selectbox(
            "Pick a station",
            options=options,
            index=default_index,
            format_func=lambda s: labels[s],
        )
        st.session_state["selected_station"] = sid

        st.header("Window")
        end_default = datetime.now(timezone.utc).date() + timedelta(days=1)
        start_default = end_default - timedelta(days=30)
        date_range = st.date_input(
            "Date range (UTC)",
            value=(start_default, end_default),
            help="Filter by station's reported timestamp (UTC).",
        )
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_dt = datetime.combine(date_range[0], datetime.min.time())
            end_dt = datetime.combine(date_range[1], datetime.min.time())
        else:
            start_dt, end_dt = start_default, end_default
            start_dt = datetime.combine(start_dt, datetime.min.time())
            end_dt = datetime.combine(end_dt, datetime.min.time())

        st.header("Filter")
        require_renting = st.checkbox(
            "Only count rows where is_renting=true",
            value=True,
            help="When unchecked, includes time the station was offline / not renting.",
        )

    meta = _meta(sid)
    cap = meta.get("capacity")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Capacity", int(cap) if cap else "—")
    c2.metric("Observations", f"{int(meta.get('n_obs') or 0):,}")
    first_obs = meta.get("first_obs")
    last_obs = meta.get("last_obs")
    c3.metric("First obs", first_obs.strftime("%Y-%m-%d") if isinstance(first_obs, datetime) else "—")
    c4.metric("Last obs", last_obs.strftime("%Y-%m-%d %H:%M") if isinstance(last_obs, datetime) else "—")

    st.caption(
        f"{meta.get('name', '')}  ·  short_name {meta.get('short_name')}  ·  "
        f"({meta.get('lat'):.4f}, {meta.get('lon'):.4f})"
        if meta
        else ""
    )

    tab_ebikes, tab_avail, tab_range, tab_churn, tab_ts = st.tabs(
        ["Ebikes", "Availability", "Range", "Churn", "Time series"]
    )

    with tab_ebikes:
        sm = _ebike_summary(sid, start_dt, end_dt)
        if not sm:
            st.info("No ebike data in window yet — let the poller run longer.")
        else:
            current = int(sm.get("current_eb") or 0)
            m = st.columns(5)
            m[0].metric("Current ebikes", current)
            m[1].metric("Min", int(sm.get("min_eb") or 0))
            m[2].metric("Median", f"{float(sm.get('median_eb') or 0):.1f}")
            m[3].metric("Mean", f"{float(sm.get('mean_eb') or 0):.2f}")
            m[4].metric("Max", int(sm.get("max_eb") or 0))

            n_days = int(sm.get("n_days") or 0)
            d = st.columns(3)
            d[0].metric(
                "Avg daily range",
                f"{float(sm.get('avg_daily_range') or 0):.1f}" if n_days else "—",
                help="Mean of (daily max − daily min) ebike count, across days with ≥2 observations.",
            )
            d[1].metric(
                "Min daily range",
                f"{int(sm.get('min_daily_range') or 0)}" if n_days else "—",
            )
            d[2].metric(
                "Max daily range",
                f"{int(sm.get('max_daily_range') or 0)}" if n_days else "—",
            )

            as_of = sm.get("as_of")
            as_of_str = as_of.strftime("%Y-%m-%d %H:%M UTC") if isinstance(as_of, datetime) else "—"
            st.caption(
                f"As of {as_of_str}  ·  {int(sm.get('n_obs') or 0):,} observations  ·  "
                f"{n_days} day{'s' if n_days != 1 else ''} with data"
            )

            _station_prediction_detail_section(sid)
            _station_trip_demand_section(sid)

            st.subheader(f"Forecast: expected ebikes in the next 20 minutes (current = {current})")
            fc = _ebike_forecast(sid, current, start_dt, end_dt)
            if fc.empty or int(fc["n_total"].fillna(0).sum()) == 0:
                st.info(
                    "Not enough historical pairs yet at lags 5/10/15/20 min. "
                    "Let the poller collect a few hours first."
                )
            else:
                st.altair_chart(_forecast_chart(fc, current), use_container_width=True)
                st.caption(
                    "**Blue line** = E[ebikes(t+k) | ebikes(t) = current]: average future "
                    "ebike count across all historical times this station had exactly this "
                    "current count. **Grey line** = unconditional E[ebikes(t+k)]: the long-run "
                    "average over the window. The blue line decays toward grey as k grows — "
                    "today's state matters less the further out you look. Shaded band = "
                    "25–75th percentile of the conditional distribution."
                )
                with st.expander("Forecast table (raw)"):
                    st.dataframe(fc, hide_index=True)

    with tab_avail:
        df = _availability(sid, start_dt, end_dt, require_renting)
        if df.empty:
            st.info("No data in window.")
        else:
            cols = st.columns(3)
            with cols[0]:
                st.altair_chart(
                    _heatmap(df, "p_empty", "P(empty: 0 bikes)", ".1%"),
                    use_container_width=True,
                )
            with cols[1]:
                st.altair_chart(
                    _heatmap(df, "p_full", "P(full: 0 docks)", ".1%"),
                    use_container_width=True,
                )
            with cols[2]:
                st.altair_chart(
                    _heatmap(df, "p_ebike_available", "P(≥1 ebike)", ".1%"),
                    use_container_width=True,
                )
            with st.expander("Mean bike / dock counts by DOW × hour"):
                st.dataframe(df, hide_index=True)

    with tab_range:
        df = _range(sid, start_dt, end_dt)
        if df.empty:
            st.info("No data in window.")
        else:
            st.altair_chart(_range_chart(df, cap), use_container_width=True)
            st.caption(
                "Solid line = median bikes. Dashed red = median ebikes. "
                "Inner band = 25–75th percentile. Outer band = 10–90th. Grey rule = capacity."
            )
            with st.expander("Raw quantiles"):
                st.dataframe(df, hide_index=True)

    with tab_churn:
        df = _churn(sid, start_dt, end_dt)
        if df.empty:
            st.info("No data in window.")
        else:
            st.altair_chart(
                _heatmap(
                    df.assign(n=df["n_days"]).rename(columns={"rides_per_hour_est": "rides_per_hour_est"}),
                    "rides_per_hour_est",
                    "Est. churn (|Δbikes| per hour, per-day avg)",
                    ".1f",
                ),
                use_container_width=True,
            )
            st.caption(
                "Churn = sum of |Δ num_bikes_available| between consecutive station updates, "
                "divided by the number of distinct local days. Approximates rides/hr — undercounts if "
                "checkouts and returns happen between two reports."
            )
            with st.expander("Raw churn table"):
                st.dataframe(df, hide_index=True)

    with tab_ts:
        span_days = max(1, (end_dt - start_dt).days)
        if span_days > 14:
            resample = 15
        elif span_days > 4:
            resample = 5
        else:
            resample = None
        df = _ts(sid, start_dt, end_dt, resample)
        if df.empty:
            st.info("No data in window.")
        else:
            st.altair_chart(_time_series_chart(df, cap), use_container_width=True)
            st.caption(
                f"Window: {start_dt:%Y-%m-%d} → {end_dt:%Y-%m-%d}  ·  "
                f"resampled to {resample}-minute buckets" if resample else "Raw timestamps."
            )


if __name__ == "__main__":
    main()
