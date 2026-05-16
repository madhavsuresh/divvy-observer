"""Streamlit dashboard for divvy-observer.

Two audiences are served by separate tabs:
  - Riders: "Find a Bike", "Station detail" — frequency-framed displays.
  - Maintainers: "Performance", "Calibration", "Decisions", "System".

See ``CALIBRATION_VIZ_DESIGN.md`` for the design rationale, and ``viz.py``
+ ``dashboard_metrics.py`` for the underlying chart and math primitives.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import altair as alt
import duckdb
import numpy as np
import pandas as pd
import pydeck as pdk
import requests
import streamlit as st
from streamlit_geolocation import streamlit_geolocation

from divvy import (
    analytics,
    config,
    dashboard_metrics,
    db,
    live_cache,
    model_eval,
    model_selection,
    predictor,
    recommendations,
    service_state,
    viz,
)
from divvy.display import display_probability


CHICAGO_LAT, CHICAGO_LON = 41.8781, -87.6298


# =============================================================================
# Cached data fetchers
# =============================================================================


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
        replica_mtime = datetime.fromtimestamp(
            config.READ_DB_PATH.stat().st_mtime, tz=timezone.utc
        ).replace(tzinfo=None)

    payload = {
        "pending_queue_files": pending_files,
        "failed_queue_files": failed_files,
        "read_replica_updated_at": replica_mtime,
    }
    try:
        with db.session(read_only=True) as conn:
            latest_tick = conn.execute(
                """
                SELECT ticked_at, station_payload_count, free_bike_payload_count,
                       station_rows_inserted, free_bike_events_inserted,
                       forecast_queue_files_processed, forecast_rows_logged,
                       forecast_queue_files_failed, outcomes_resolved,
                       metrics_rows_snapshotted
                FROM collector_ticks
                ORDER BY ticked_at DESC LIMIT 1
                """
            ).fetchone()
            recent = conn.execute(
                """
                SELECT COUNT(*) AS ticks_10m,
                       COALESCE(SUM(station_rows_inserted), 0) AS station_rows_10m,
                       COALESCE(SUM(forecast_rows_logged), 0) AS forecasts_10m,
                       COALESCE(SUM(outcomes_resolved), 0) AS outcomes_10m
                FROM collector_ticks
                WHERE ticked_at >= now() - INTERVAL '10 minutes'
                """
            ).fetchone()
            freshness = conn.execute(
                """
                SELECT (SELECT MAX(last_reported) FROM station_status) AS latest_station_reported,
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
            "station_rows_inserted": int(latest_tick[3] or 0),
            "forecast_rows_logged": int(latest_tick[6] or 0),
            "forecast_queue_files_failed": int(latest_tick[7] or 0),
            "outcomes_resolved": int(latest_tick[8] or 0),
            "metrics_rows_snapshotted": int(latest_tick[9] or 0),
        })
    if recent:
        payload.update({
            "ticks_10m": int(recent[0] or 0),
            "station_rows_10m": int(recent[1] or 0),
            "forecasts_10m": int(recent[2] or 0),
            "outcomes_10m": int(recent[3] or 0),
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
    """Active-model + cache health snapshot. Tested by tests/test_dashboard_payload.py."""
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
    lat: float, lon: float, near_radius_km: float, search_radius_km: float, query_label: str | None,
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
def _performance_summary(window_hours: int) -> dict:
    with db.session(read_only=True) as conn:
        return model_eval.performance_summary(
            conn, window_hours=window_hours, resolve=False, initialize_schema=False
        )


@st.cache_data(ttl=120)
def _per_horizon_performance(window_hours: int) -> dict:
    with db.session(read_only=True) as conn:
        return model_eval.per_horizon_performance_summary(
            conn, window_hours=window_hours, initialize_schema=False
        )


@st.cache_data(ttl=120)
def _topk_recommendation(window_hours: int) -> dict:
    with db.session(read_only=True) as conn:
        return model_eval.topk_recommendation_summary(
            conn, window_hours=window_hours, k_values=(1, 3, 5), initialize_schema=False
        )


@st.cache_data(ttl=120)
def _multi_bike_performance(window_hours: int) -> dict:
    with db.session(read_only=True) as conn:
        return model_eval.multi_bike_performance_summary(
            conn, window_hours=window_hours, plan_sizes=(2, 3, 4, 5), initialize_schema=False
        )


@st.cache_data(ttl=120)
def _resolved_forecasts(window_hours: int) -> pd.DataFrame:
    """Joined `(forecast, outcome)` frame the new charts consume directly."""
    with db.session(read_only=True) as conn:
        return dashboard_metrics.resolved_forecasts(conn, window_hours=window_hours)


@st.cache_data(ttl=120)
def _metric_trend(days: int) -> pd.DataFrame:
    with db.session(read_only=True) as conn:
        return dashboard_metrics.metric_trend(
            conn,
            metric_columns=("brier_score", "log_loss", "rank_loss", "ece"),
            days=days,
            by_model=True,
        )


@st.cache_data(ttl=300)
def _nearest(lat: float, lon: float, k: int) -> pd.DataFrame:
    with db.session(read_only=True) as conn:
        return analytics.nearest_stations(conn, lat, lon, k)


@st.cache_data(ttl=86400)
def _geocode(address: str) -> tuple[float, float, str] | None:
    """Free-form address → (lat, lon, display_name) via Nominatim."""
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


@st.cache_data(ttl=30)
def _free_bikes() -> pd.DataFrame:
    """Live floating-eBike positions from Divvy GBFS."""
    try:
        resp = requests.get(
            "https://gbfs.divvybikes.com/gbfs/en/free_bike_status.json",
            headers={"User-Agent": "divvy-observer/0.1 (personal research)"},
            timeout=10,
        )
        resp.raise_for_status()
    except requests.RequestException:
        return pd.DataFrame()
    rows = []
    for b in resp.json().get("data", {}).get("bikes", []):
        if b.get("is_disabled") or b.get("is_reserved"):
            continue
        lat, lon = b.get("lat"), b.get("lon")
        if lat is None or lon is None:
            continue
        rows.append({"bike_id": b.get("bike_id"), "lat": float(lat), "lon": float(lon)})
    return pd.DataFrame(rows)


# =============================================================================
# Formatting helpers
# =============================================================================


def _prob_label(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "—"
    return display_probability(float(value))


def _km_label(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "—"
    v = float(value)
    return f"{v * 1000:.0f} m" if v < 0.95 else f"{v:.2f} km"


def _minutes_label(minutes: float | None) -> str:
    if minutes is None or pd.isna(minutes):
        return "—"
    m = float(minutes)
    return "<1 min" if m < 1 else f"{m:.0f} min"


def _age_minutes(ts) -> float | None:
    if ts is None or (isinstance(ts, float) and math.isnan(ts)):
        return None
    if isinstance(ts, str):
        try:
            ts = pd.Timestamp(ts)
        except (ValueError, TypeError):
            return None
    if isinstance(ts, pd.Timestamp):
        ts = ts.to_pydatetime()
    if not isinstance(ts, datetime):
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    delta = datetime.now(timezone.utc) - ts
    return delta.total_seconds() / 60.0


# =============================================================================
# Tab 1 — Find a Bike (rider product surface)
# =============================================================================


def _resolve_location(default_lat: float, default_lon: float) -> tuple[float, float, str]:
    """Sidebar widget that returns (lat, lon, label) from one of three sources.

    Priority: live geolocation > geocoded address > manual lat/lon.
    """
    st.sidebar.subheader("Where are you?")
    location_method = st.sidebar.radio(
        "Location source",
        options=["Address", "Browser geolocation", "Manual lat/lon"],
        index=0,
        key="location_method",
    )
    if location_method == "Address":
        address = st.sidebar.text_input(
            "Address or place",
            value=st.session_state.get("find_address", ""),
            placeholder="e.g. 600 W Chicago Ave",
            key="find_address",
        )
        if address:
            try:
                hit = _geocode(address)
            except requests.RequestException as exc:
                st.sidebar.error(f"Geocoding failed: {exc}")
                hit = None
            if hit:
                lat, lon, display = hit
                return lat, lon, display
            st.sidebar.warning("No match — falling back to default.")
        return default_lat, default_lon, "Chicago (default)"

    if location_method == "Browser geolocation":
        coords = streamlit_geolocation()
        if coords and coords.get("latitude") is not None:
            lat = float(coords["latitude"])
            lon = float(coords["longitude"])
            return lat, lon, "Browser location"
        st.sidebar.info("Click the 📍 button to grant location.")
        return default_lat, default_lon, "Chicago (default)"

    cols = st.sidebar.columns(2)
    lat = cols[0].number_input("Lat", value=default_lat, format="%.4f", key="find_lat")
    lon = cols[1].number_input("Lon", value=default_lon, format="%.4f", key="find_lon")
    return float(lat), float(lon), f"({lat:.4f}, {lon:.4f})"


def _station_dock_probability(station: dict, *, horizon_label: str = "10m") -> float | None:
    """Pull P(has open dock at horizon) from the API station payload.

    Tries the dock-constrained arrival probability first (primary signal),
    then falls back to the capacity-violation probability. Both are stored
    as "no dock" probabilities, so the rider quantity is `1 − value`.
    """
    raw = station.get(f"p_dock_constrained_arrival_{horizon_label}")
    if raw is None:
        raw = station.get(f"p_capacity_violation_{horizon_label}")
    if raw is None:
        return None
    try:
        return max(0.0, min(1.0, 1.0 - float(raw)))
    except (TypeError, ValueError):
        return None


def _station_thumbnail(
    station: dict,
    *,
    horizon_label: str = "10m",
    show_bike: bool = True,
    show_dock: bool = False,
) -> None:
    """Compact rider-facing card for one station: dot grid(s) + name + distance.

    Shows the P(has bike) dot grid by default. When ``show_dock`` is on,
    a second grid colored orange shows P(has open dock) for the return
    trip — useful when the rider is bringing a bike back and needs an
    open dock at the destination.
    """
    p_bike = station.get(f"p_has_ebike_{horizon_label}") or station.get("p_arrival")
    p_dock = _station_dock_probability(station, horizon_label=horizon_label) if show_dock else None
    name = station.get("name") or "Unknown station"
    distance = station.get("distance_km")
    arrival = station.get("arrival_time_minutes")
    current = station.get("current_ebikes") or 0
    current_docks = station.get("current_docks")

    if show_bike and show_dock:
        cols = st.columns(2)
        with cols[0]:
            positions = dashboard_metrics.dot_grid_positions(p_bike, n=100, cols=10)
            st.altair_chart(
                viz.dot_grid_chart(
                    positions, probability=p_bike, title=f"🚲 {name}",
                    filled_color=viz.BIKE_COLOR, dot_size=70,
                ),
                use_container_width=False,
            )
        with cols[1]:
            positions = dashboard_metrics.dot_grid_positions(p_dock, n=100, cols=10)
            st.altair_chart(
                viz.dot_grid_chart(
                    positions, probability=p_dock, title=f"🅿️ open dock",
                    filled_color=viz.DOCK_COLOR, dot_size=70,
                ),
                use_container_width=False,
            )
    elif show_dock:
        positions = dashboard_metrics.dot_grid_positions(p_dock, n=100, cols=10)
        st.altair_chart(
            viz.dot_grid_chart(
                positions, probability=p_dock, title=name,
                filled_color=viz.DOCK_COLOR, dot_size=90,
            ),
            use_container_width=False,
        )
    else:
        positions = dashboard_metrics.dot_grid_positions(p_bike, n=100, cols=10)
        st.altair_chart(
            viz.dot_grid_chart(
                positions, probability=p_bike, title=name,
                filled_color=viz.BIKE_COLOR, dot_size=90,
            ),
            use_container_width=False,
        )

    caption_parts = [
        f"📍 {_km_label(distance)}",
        f"🚶 {_minutes_label(arrival)}",
        f"🚲 {int(current)} now",
    ]
    if show_dock and current_docks is not None:
        try:
            caption_parts.append(f"🅿️ {int(current_docks)} docks")
        except (TypeError, ValueError):
            pass
    st.caption(" · ".join(caption_parts))


def _render_find_a_bike() -> None:
    st.subheader("🚲 Find a bike")
    st.caption(
        "Each dot = 1 in 100 trips. Fill count shows your chance of finding a bike "
        "when you arrive. Choose the station with the densest dots that's also a "
        "comfortable walk. Toggle 🅿️ to also see open-dock probabilities — "
        "useful when you're bringing a bike *back*."
    )

    lat, lon, location_label = _resolve_location(CHICAGO_LAT, CHICAGO_LON)
    st.sidebar.caption(f"📍 {location_label}")

    with st.sidebar.expander("Search radius", expanded=False):
        near_radius_km = st.slider(
            "Near (preferred) radius (km)", 0.1, 1.0, 0.5, 0.05, key="find_near_radius"
        )
        search_radius_km = st.slider(
            "Search radius (km)", 0.5, 3.0, 1.5, 0.1, key="find_search_radius"
        )

    mode = st.sidebar.radio(
        "I'm looking for…",
        options=["🚲 a bike", "🅿️ an open dock", "Both"],
        index=0,
        key="find_mode",
        help="Pickup, return, or both probabilities side-by-side.",
    )
    show_bike = mode in ("🚲 a bike", "Both")
    show_dock = mode in ("🅿️ an open dock", "Both")

    try:
        payload = _prediction_recommendation(
            lat, lon, near_radius_km, search_radius_km, query_label="streamlit"
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Recommendation request failed: {exc}")
        return

    if payload.get("candidate_count", 0) == 0:
        st.warning(payload.get("message", "No stations found in that radius."))
        return

    best = payload.get("best_practical_station_5_10m") or {}
    alternatives = payload.get("reliable_alternatives") or []

    # ----- Hero card -----
    st.markdown("### Recommended right now")
    hero_cols = st.columns([1.4, 1])
    with hero_cols[0]:
        _station_thumbnail(best, horizon_label="10m", show_bike=show_bike, show_dock=show_dock)
    with hero_cols[1]:
        st.markdown(f"#### {best.get('name', '—')}")
        lines = [
            f"**Walk:** {_km_label(best.get('distance_km'))} "
            f"(~{_minutes_label(best.get('arrival_time_minutes'))})",
            f"**Bikes right now:** {int(best.get('current_ebikes') or 0)}",
        ]
        if show_bike:
            lines.append(f"**P(bike when you arrive):** {_prob_label(best.get('p_arrival'))}")
        if show_dock:
            p_dock = _station_dock_probability(best, horizon_label="10m")
            current_docks = best.get("current_docks")
            if current_docks is not None:
                lines.append(f"**Docks right now:** {int(current_docks)}")
            lines.append(f"**P(open dock when you arrive):** {_prob_label(p_dock)}")
        st.markdown("  \n".join(lines))
        lcb = best.get("reliable_probability_lcb")
        if lcb is not None and show_bike:
            st.caption(
                f"Conservative bike estimate (95% LCB after calibration): "
                f"{_prob_label(lcb)}. Use this as your worst-case planning bet."
            )
        st.caption(f"Active model: `{payload.get('active_model_key') or '—'}`")

    # ----- Comparison cohort -----
    if alternatives:
        st.markdown("### Other nearby options")
        cohort = alternatives[:4]
        cols = st.columns(len(cohort))
        for col, station in zip(cols, cohort):
            with col:
                _station_thumbnail(
                    station, horizon_label="10m",
                    show_bike=show_bike, show_dock=show_dock,
                )

    # ----- Map -----
    st.markdown("### Map")
    map_rows = []
    if best:
        map_rows.append(best)
    map_rows.extend(alternatives)
    if map_rows:
        map_df = pd.DataFrame(map_rows)
        map_df = map_df[map_df["lat"].notna() & map_df["lon"].notna()].copy()
        map_df["p_has_ebike_10m"] = pd.to_numeric(map_df.get("p_has_ebike_10m"), errors="coerce").fillna(0.0)
        map_df["color_r"] = (220 * (1 - map_df["p_has_ebike_10m"])).astype(int)
        map_df["color_g"] = (200 * map_df["p_has_ebike_10m"]).astype(int)
        map_df["color_b"] = 80
        map_df["radius"] = 60 + 80 * map_df["p_has_ebike_10m"]
        layers = [
            pdk.Layer(
                "ScatterplotLayer",
                data=map_df,
                get_position=["lon", "lat"],
                get_fill_color=["color_r", "color_g", "color_b", 180],
                get_radius="radius",
                pickable=True,
            ),
            pdk.Layer(
                "ScatterplotLayer",
                data=pd.DataFrame([{"lat": lat, "lon": lon}]),
                get_position=["lon", "lat"],
                get_fill_color=[60, 100, 220, 220],
                get_radius=70,
            ),
        ]
        deck = pdk.Deck(
            layers=layers,
            initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=14, pitch=0),
            map_style="light",
            tooltip={"text": "{name}\nP(bike in 10m): {p_has_ebike_10m}"},
        )
        st.pydeck_chart(deck)

    # ----- Multi-bike plan -----
    st.divider()
    st.markdown("### Need more than one bike?")
    with st.expander("Multi-bike plan", expanded=False):
        plan_cols = st.columns([1, 1, 1])
        k = plan_cols[0].number_input("How many bikes?", min_value=2, max_value=8, value=2, step=1, key="plan_k")
        horizon = plan_cols[1].select_slider(
            "Plan horizon (min)", options=[5, 10, 15, 20], value=10, key="plan_horizon"
        )
        target = plan_cols[2].slider(
            "Target success", min_value=0.5, max_value=0.99, value=0.9, step=0.01, key="plan_target",
        )
        try:
            plan = _prediction_multi_bike_plan(
                lat, lon, int(k), near_radius_km, search_radius_km,
                plan_horizon_minutes=int(horizon),
                target_success_probability=float(target),
                query_label="streamlit_multi_bike",
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Plan request failed: {exc}")
            plan = None
        if plan:
            primary = plan.get("primary_plan") or {}
            success = primary.get("success_probability")
            stops = primary.get("stops") or []
            if not stops:
                st.info(plan.get("message", "No viable plan yet — try a larger radius."))
            else:
                inner = st.columns([1, 1.5])
                with inner[0]:
                    positions = dashboard_metrics.dot_grid_positions(success, n=100, cols=10)
                    st.altair_chart(
                        viz.dot_grid_chart(
                            positions,
                            probability=success,
                            title=f"Plan: {len(stops)} stop{'s' if len(stops) > 1 else ''}",
                            filled_color="#2ca02c",
                        ),
                        use_container_width=False,
                    )
                with inner[1]:
                    stop_df = pd.DataFrame(stops)
                    cols_to_show = [c for c in (
                        "order", "name", "distance_km", "arrival_time_minutes",
                        "p_arrival", "current_ebikes",
                    ) if c in stop_df.columns]
                    if cols_to_show:
                        st.dataframe(
                            stop_df[cols_to_show].rename(
                                columns={
                                    "order": "Order",
                                    "name": "Station",
                                    "distance_km": "Distance (km)",
                                    "arrival_time_minutes": "Walk (min)",
                                    "p_arrival": "P(arrival)",
                                    "current_ebikes": "Bikes now",
                                }
                            ),
                            hide_index=True,
                            use_container_width=True,
                        )


# =============================================================================
# Tab 2 — Station detail (per-station deep dive)
# =============================================================================


def _target_from_payload(payload: dict, station_id: str) -> dict | None:
    for candidate in [payload.get("best_practical_station_5_10m"), *(payload.get("reliable_alternatives") or [])]:
        if candidate and candidate.get("station_id") == station_id:
            return candidate
    return None


def _build_station_horizon_curve(payload: dict, station_id: str) -> pd.DataFrame:
    """Extract per-horizon P(has bike) for the active model + comparison models."""
    rows = []
    target_station = _target_from_payload(payload, station_id)
    if target_station is None:
        return pd.DataFrame()
    for horizon in predictor.HORIZONS:
        p = target_station.get(f"p_has_ebike_{horizon}m")
        if p is None:
            continue
        rows.append({
            "horizon_minutes": horizon,
            "p_has_ebike": float(p),
            "model_label": "active",
        })
    # Per-model variants
    for model in (target_station.get("model_predictions") or []):
        for horizon in predictor.HORIZONS:
            p = model.get(f"p_has_ebike_{horizon}m")
            if p is None:
                continue
            rows.append({
                "horizon_minutes": horizon,
                "p_has_ebike": float(p),
                "model_label": model.get("label") or model.get("model_key") or "model",
            })
    return pd.DataFrame(rows)


def _build_station_dock_curve(payload: dict, station_id: str) -> pd.DataFrame:
    """Extract per-horizon P(has open dock) for the active model.

    Pulled from the per-model predictions list — the top-level payload only
    carries the 10m dock probability. Falls back to ``p_capacity_violation``
    when ``p_dock_constrained_arrival`` is missing.
    """
    target_station = _target_from_payload(payload, station_id)
    if target_station is None:
        return pd.DataFrame()
    rows = []
    for model in (target_station.get("model_predictions") or []):
        label = model.get("label") or model.get("model_key") or "model"
        for horizon in predictor.HORIZONS:
            raw = model.get(f"p_dock_constrained_arrival_{horizon}m")
            if raw is None:
                raw = model.get(f"p_capacity_violation_{horizon}m")
            if raw is None:
                continue
            try:
                p_open = max(0.0, min(1.0, 1.0 - float(raw)))
            except (TypeError, ValueError):
                continue
            rows.append({
                "horizon_minutes": horizon,
                "p_has_open_dock": p_open,
                "model_label": label,
            })
    return pd.DataFrame(rows)


def _render_station_detail() -> None:
    st.subheader("📍 Station detail")
    try:
        stations_df = _stations()
    except duckdb.IOException as exc:
        st.error(f"Could not open DuckDB: {exc}")
        return
    if stations_df.empty:
        st.warning("No stations indexed yet.")
        return

    search = st.text_input("Search stations", value="", placeholder="name or short code")
    filtered = (
        stations_df[stations_df["name"].str.contains(search, case=False, na=False)]
        if search else stations_df
    )
    if filtered.empty:
        st.info("No stations match.")
        return

    options = filtered["station_id"].tolist()
    labels = {row["station_id"]: f"{row['name']} (cap {int(row.capacity) if pd.notna(row.capacity) else 0})"
              for _, row in filtered.iterrows()}
    default_sid = st.session_state.get("selected_station")
    default_idx = options.index(default_sid) if default_sid in options else 0
    sid = st.selectbox(
        "Station", options=options, index=default_idx, format_func=lambda s: labels.get(s, s),
        key="station_detail_selector",
    )
    st.session_state["selected_station"] = sid

    meta = _meta(sid)
    cap = meta.get("capacity")
    summary = _ebike_summary(sid, datetime.utcnow() - timedelta(days=30), datetime.utcnow())

    # Header metrics
    cols = st.columns(5)
    cols[0].metric("Capacity", int(cap) if cap else "—")
    cols[1].metric("Bikes now", int(summary.get("current_eb") or 0))
    cols[2].metric("Median (30d)", f"{float(summary.get('median_eb') or 0):.1f}")
    cols[3].metric("Daily range", f"{float(summary.get('avg_daily_range') or 0):.1f}")
    last_obs = meta.get("last_obs")
    cols[4].metric("Last seen",
                   last_obs.strftime("%H:%M") if isinstance(last_obs, datetime) else "—")

    # ----- Per-horizon forecast curve -----
    st.markdown("#### Forecast — probabilities by horizon")
    try:
        recommendation = _prediction_recommendation(
            float(meta.get("lat") or CHICAGO_LAT),
            float(meta.get("lon") or CHICAGO_LON),
            0.5, 1.5, query_label=f"station:{sid}",
        )
        horizon_df = _build_station_horizon_curve(recommendation, sid)
        dock_df = _build_station_dock_curve(recommendation, sid)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Couldn't fetch live forecast: {exc}")
        horizon_df = pd.DataFrame()
        dock_df = pd.DataFrame()

    if horizon_df.empty and dock_df.empty:
        st.info("No live forecast available — the predictor returned no rows for this station.")
    else:
        toggle_cols = st.columns([1, 1, 1])
        show_bike_curve = toggle_cols[0].toggle(
            "🚲 Show P(has bike)", value=True, key="station_detail_show_bike"
        )
        show_dock_curve = toggle_cols[1].toggle(
            "🅿️ Show P(has open dock)", value=True, key="station_detail_show_dock"
        )
        compare = toggle_cols[2].toggle(
            "Compare all models", value=False, key="station_detail_compare",
            help="Show every model's curve instead of just the active model.",
        )

        if not horizon_df.empty and show_bike_curve:
            bike_df = horizon_df.copy()
            if not compare:
                bike_df = bike_df[bike_df["model_label"] == "active"]
            bike_df["target"] = "P(has bike)"
            bike_df = bike_df.rename(columns={"p_has_ebike": "probability"})
        else:
            bike_df = pd.DataFrame()

        if not dock_df.empty and show_dock_curve:
            # No 'active' aggregation for dock (only per-model), so just dedupe per-horizon mean.
            if not compare and not dock_df.empty:
                # Average across models for a single curve when not in compare mode.
                summary = (
                    dock_df.groupby("horizon_minutes")["p_has_open_dock"].mean().reset_index()
                )
                summary["model_label"] = "active"
                dock_summary = summary
            else:
                dock_summary = dock_df.rename(columns={"p_has_open_dock": "probability"}).copy()
                dock_summary["target"] = "P(has open dock)"
            if "probability" not in dock_summary.columns:
                dock_summary = dock_summary.rename(columns={"p_has_open_dock": "probability"})
            dock_summary["target"] = "P(has open dock)"
        else:
            dock_summary = pd.DataFrame()

        combined = pd.concat([bike_df, dock_summary], ignore_index=True) if not (bike_df.empty and dock_summary.empty) else pd.DataFrame()
        if combined.empty:
            st.info("Both target toggles are off — turn one on to see the curve.")
        else:
            combined = combined.dropna(subset=["probability"])
            color_scale = alt.Scale(
                domain=["P(has bike)", "P(has open dock)"],
                range=[viz.BIKE_COLOR, viz.DOCK_COLOR],
            )
            stroke_scale = alt.Scale(scheme=viz.MODEL_PALETTE)
            line_encoding = {
                "x": alt.X("horizon_minutes:Q", title="Minutes from now"),
                "y": alt.Y("probability:Q", title="Probability", scale=alt.Scale(domain=[0, 1])),
                "color": alt.Color("target:N", scale=color_scale, legend=alt.Legend(title="Target")),
                "tooltip": [
                    alt.Tooltip("target:N", title="Target"),
                    alt.Tooltip("model_label:N", title="Model"),
                    alt.Tooltip("horizon_minutes:Q", title="Horizon"),
                    alt.Tooltip("probability:Q", format=".2f", title="P"),
                ],
            }
            if compare and combined["model_label"].nunique() > 1:
                line_encoding["strokeDash"] = alt.StrokeDash("model_label:N", legend=alt.Legend(title="Model"))
            chart = alt.Chart(combined).mark_line(point=True, strokeWidth=2).encode(**line_encoding)
            st.altair_chart(
                chart.properties(title="Bike + dock probability over horizon", height=240, width="container"),
                use_container_width=True,
            )

    # ----- Historical availability heatmap -----
    st.markdown("#### Historical P(has bike) by hour × day-of-week")
    hist_df = _availability(sid, datetime.utcnow() - timedelta(days=30), datetime.utcnow(), True)
    if hist_df.empty:
        st.info("Not enough history yet.")
    else:
        chart = alt.Chart(hist_df).mark_rect().encode(
            x=alt.X("local_hour:O", title="Hour"),
            y=alt.Y("dow_label:N", title="Day", sort=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]),
            color=alt.Color(
                "p_ebike_available:Q",
                scale=alt.Scale(scheme="blues", domain=[0, 1]),
                title="P(≥1 ebike)",
            ),
            tooltip=[
                alt.Tooltip("dow_label:N", title="Day"),
                alt.Tooltip("local_hour:Q", title="Hour"),
                alt.Tooltip("p_ebike_available:Q", format=".1%"),
            ],
        ).properties(height=240, width="container")
        st.altair_chart(chart, use_container_width=True)

    # ----- Recent live availability time series -----
    st.markdown("#### Recent availability")
    ts_df = _ts(sid, datetime.utcnow() - timedelta(days=2), datetime.utcnow(), None)
    if not ts_df.empty:
        ts_chart = alt.Chart(ts_df).mark_line().encode(
            x=alt.X("local_time:T", title="When (local)"),
            y=alt.Y("num_bikes_available:Q", title="Bikes available"),
            tooltip=[
                alt.Tooltip("local_time:T"),
                alt.Tooltip("num_bikes_available:Q"),
                alt.Tooltip("num_ebikes_available:Q"),
            ],
        ).properties(height=200, width="container")
        st.altair_chart(ts_chart, use_container_width=True)


# =============================================================================
# Tab 3 — Model Performance (maintainer leaderboard)
# =============================================================================


def _format_leaderboard_for_display(perf: dict) -> pd.DataFrame:
    leaderboard = perf.get("model_leaderboard") or []
    if not leaderboard:
        return pd.DataFrame()
    enriched = dashboard_metrics.attach_skill_scores(leaderboard, baseline_key="empirical", metric="brier_score")
    df = viz.leaderboard_frame(enriched)
    rename_map = {
        "rank": "#",
        "model_label": "Model",
        "model_key": "Key",
        "n": "N",
        "brier_score": "Brier ↓",
        "log_loss": "LogLoss ↓",
        "ece": "ECE ↓",
        "rank_loss": "RankLoss ↓",
        "decision_rank_loss": "Decision ↓",
        "skill_score": "Skill ↑",
        "observed_rate": "Observed",
        "mean_prediction": "Mean pred",
        "recommended_hit_rate": "TopHit ↑",
        "distance_adjusted_regret": "Regret ↓",
        "count_log_loss": "CountNLL ↓",
        "crps": "CRPS ↓",
        "mean_expected_ebikes": "E[bikes]",
        "mean_observed_ebikes": "Obs[bikes]",
    }
    return df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})


def _render_performance(window_hours: int) -> None:
    st.subheader(f"🏆 Model performance ({window_hours}h window)")

    try:
        perf = _performance_summary(window_hours)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not compute performance: {exc}")
        return

    overall = perf.get("overall") or {}
    active = perf.get("active_model") or {}
    n = int(overall.get("n") or 0)

    if n == 0:
        st.info("No resolved outcomes in this window yet. Let the collector keep running.")
        return

    # ----- KPI strip -----
    kpi = st.columns(5)
    kpi[0].metric("Resolved", f"{n:,}")
    kpi[1].metric("Brier", f"{overall.get('brier_score'):.4f}" if overall.get("brier_score") is not None else "—")
    kpi[2].metric("ECE", f"{overall.get('ece'):.3f}" if overall.get("ece") is not None else "—")
    kpi[3].metric("LogLoss", f"{overall.get('log_loss'):.4f}" if overall.get("log_loss") is not None else "—")
    kpi[4].metric("Active model", active.get("model_label") or active.get("model_key") or "—")

    active_equals_best = perf.get("active_equals_best")
    best = perf.get("best_current_model") or {}
    if active_equals_best is False and best:
        st.warning(
            f"📈 A non-active model is leading: **{best.get('model_label')}** has the "
            f"lowest decision rank loss. Active model is **{active.get('model_label')}**."
        )
    elif active_equals_best is True:
        st.success("✅ Active model is also the best-scoring model in this window.")

    # ----- Leaderboard -----
    st.markdown("#### Leaderboard")
    leaderboard_df = _format_leaderboard_for_display(perf)
    if leaderboard_df.empty:
        st.info("No per-model rows.")
    else:
        st.dataframe(leaderboard_df, hide_index=True, use_container_width=True)
        st.caption(
            "**Skill** = `1 − Brier_model / Brier_empirical_baseline`; >0 means better than baseline. "
            "**Decision** is the composite `decision_rank_loss` the active-model promoter uses."
        )

    # ----- Brier decomposition -----
    st.markdown("#### Brier decomposition (Murphy 1973)")
    resolved = _resolved_forecasts(window_hours)
    if not resolved.empty:
        decomp = dashboard_metrics.brier_decomposition_by_model(resolved)
        if not decomp.empty:
            st.altair_chart(
                viz.brier_decomposition_chart(decomp),
                use_container_width=True,
            )
            st.caption(
                "`Brier = Reliability − Resolution + Uncertainty`. Lower **reliability** (red) is better — "
                "it's calibration error. Higher **resolution** (green) is better — it's discrimination above "
                "the marginal rate. **Uncertainty** (grey) is fixed by the data and identical across models."
            )
        else:
            st.info("Not enough samples for decomposition.")
    else:
        st.info("No resolved forecasts in window.")

    # ----- Per-horizon performance -----
    st.markdown("#### Skill by horizon")
    horizons = perf.get("by_horizon_model") or []
    if horizons:
        h_df = pd.DataFrame(horizons)
        if "model_label" in h_df.columns and "brier_score" in h_df.columns:
            chart = alt.Chart(h_df.dropna(subset=["brier_score"])).mark_line(point=True).encode(
                x=alt.X("horizon_minutes:O", title="Horizon (min)"),
                y=alt.Y("brier_score:Q", title="Brier ↓"),
                color=alt.Color("model_label:N", scale=alt.Scale(scheme=viz.MODEL_PALETTE), legend=alt.Legend(title="Model")),
                tooltip=[
                    alt.Tooltip("model_label:N", title="Model"),
                    alt.Tooltip("horizon_minutes:Q", title="Horizon"),
                    alt.Tooltip("brier_score:Q", format=".4f"),
                    alt.Tooltip("n:Q", title="N", format=","),
                ],
            ).properties(height=240, width="container")
            st.altair_chart(chart, use_container_width=True)
            st.caption("Skill typically degrades with horizon. A flat line = the model isn't using temporal context.")

    # ----- Rolling trend -----
    st.markdown("#### Rolling trend")
    days = st.slider("Trend window (days)", min_value=1, max_value=30, value=7, key="perf_trend_days")
    trend = _metric_trend(days)
    if trend.empty:
        st.info("No metric snapshots yet. The poller writes these every tick.")
    else:
        metric_choice = st.selectbox(
            "Metric", options=["brier_score", "log_loss", "rank_loss", "ece"], index=0, key="perf_trend_metric"
        )
        st.altair_chart(
            viz.metric_trend_chart(trend, value_col=metric_choice, title=f"{metric_choice} per model"),
            use_container_width=True,
        )


# =============================================================================
# Tab 4 — Calibration (maintainer deep dive)
# =============================================================================


_TARGET_SPECS = {
    "P(has bike)": {
        "prob_col": "p_has_ebike",
        "outcome_col": "observed_has_ebike",
        "label": "P(has bike)",
    },
    "P(has open dock)": {
        "prob_col": "p_has_open_dock",
        "outcome_col": "observed_has_open_dock",
        "label": "P(has open dock)",
    },
}


def _render_calibration(window_hours: int) -> None:
    st.subheader(f"📐 Calibration ({window_hours}h window)")
    resolved = _resolved_forecasts(window_hours)
    if resolved.empty:
        st.info("No resolved outcomes yet.")
        return

    # Always attach the dock-target columns so the selector works.
    resolved = dashboard_metrics.derive_dock_target(resolved)

    # ----- Target + model + horizon filters -----
    filt_cols = st.columns([1.4, 2, 2, 1])
    target_label = filt_cols[0].radio(
        "Target",
        options=list(_TARGET_SPECS.keys()),
        index=0,
        key="calib_target",
        help="Choose which probability to calibrate against. Bike = pickup, dock = return trip.",
    )
    target = _TARGET_SPECS[target_label]
    prob_col = target["prob_col"]
    outcome_col = target["outcome_col"]
    if prob_col not in resolved.columns or resolved[prob_col].dropna().empty:
        st.info(f"No samples for {target_label} yet — this signal may not be populated for any active model.")
        return

    models = sorted({m for m in resolved["model_key"].dropna().unique()})
    horizons = sorted({int(h) for h in resolved["horizon_minutes"].dropna().unique()})
    selected_models = filt_cols[1].multiselect(
        "Models",
        options=models,
        default=models[:5] if len(models) > 5 else models,
        key="calib_models",
    )
    selected_horizons = filt_cols[2].multiselect(
        "Horizons (min)",
        options=horizons,
        default=[h for h in (10, 30) if h in horizons] or horizons[:2],
        key="calib_horizons",
    )
    facet_by_horizon = filt_cols[3].checkbox("Facet by horizon", value=False, key="calib_facet")

    filtered = resolved[
        resolved["model_key"].isin(selected_models)
        & resolved["horizon_minutes"].isin(selected_horizons)
    ].copy()
    filtered = filtered.dropna(subset=[prob_col, outcome_col])
    if filtered.empty:
        st.info("Filter selected zero rows for this target. Pick at least one model and horizon.")
        return

    target_caption = f"Target: **{target_label}** · {len(filtered):,} resolved samples in scope"
    st.caption(target_caption)

    # ----- Reliability diagram -----
    st.markdown("#### Reliability diagram")
    st.caption(
        f"Predicted {target_label} on x, observed positive rate on y, with Wilson 95% intervals. "
        "Dots on the diagonal = calibrated. Below the diagonal = overconfident; above = underconfident."
    )
    rel = dashboard_metrics.reliability_curve(
        filtered, prob_col=prob_col, outcome_col=outcome_col,
    )
    if rel.empty:
        st.info("No bins have ≥5 samples — try a longer window.")
    else:
        st.altair_chart(
            viz.reliability_diagram_chart(
                rel, facet_col="horizon_minutes" if facet_by_horizon else None
            ),
            use_container_width=True,
        )

    # ----- Score distribution (discrimination) -----
    st.markdown("#### Discrimination — predicted probability split by outcome")
    st.caption(
        f"Two overlaid histograms: predicted {target_label} when the outcome was positive (blue) vs. "
        "negative (red). Well-separated = the model is discriminating; overlapping = the model can't "
        "tell good cases from bad."
    )
    dist = dashboard_metrics.score_distribution(
        filtered, prob_col=prob_col, outcome_col=outcome_col,
    )
    if dist.empty:
        st.info("Not enough samples.")
    else:
        facet_arg = "model_label" if len(selected_models) > 1 else None
        st.altair_chart(
            viz.score_distribution_chart(dist, facet_col=facet_arg),
            use_container_width=True,
        )

    # ----- Sharpness ↔ ECE scatter -----
    st.markdown("#### Sharpness vs calibration")
    st.caption(
        "Each point is a `(model, horizon, hour-band)` bucket. **Bottom-left** is the target: sharp "
        "AND calibrated. **Top-left** is dangerous: confidently wrong. **Right** is calibrated-but-blunt."
    )
    bucketed = dashboard_metrics.time_of_week_features(filtered).copy()
    bucketed["hour_band"] = pd.cut(
        bucketed["local_hour"],
        bins=[-1, 6, 10, 16, 20, 24],
        labels=["overnight", "morning", "midday", "evening", "late"],
    )
    sharp = dashboard_metrics.sharpness_ece_scatter(
        bucketed,
        prob_col=prob_col,
        outcome_col=outcome_col,
        bucket_cols=("hour_band",),
        min_per_bucket=30,
    )
    if sharp.empty:
        st.info("Not enough samples per bucket — try a longer window.")
    else:
        st.altair_chart(viz.sharpness_ece_chart(sharp), use_container_width=True)

    # ----- Coverage heatmap -----
    st.markdown("#### Where is the model broken? (hour × day-of-week)")
    st.caption(
        "Each cell is one (day, hour) bucket. The colour shows the predicted-minus-observed gap. "
        "Solid red = systematically overconfident; solid blue = systematically underconfident; "
        "white = well-calibrated."
    )
    heatmap_df = dashboard_metrics.coverage_heatmap_data(
        filtered, prob_col=prob_col, outcome_col=outcome_col, min_per_cell=15,
    )
    if heatmap_df.empty:
        st.info("Not enough per-cell samples yet — extend the window or pick fewer model filters.")
    else:
        metric_choice = st.radio(
            "Cell colour", options=["calibration_gap", "ece", "observed_rate"],
            index=0, horizontal=True, key="calib_heatmap_metric",
        )
        st.altair_chart(
            viz.coverage_heatmap_chart(heatmap_df, metric=metric_choice),
            use_container_width=True,
        )

    # ----- Count PMF PIT (where models emit PMFs) -----
    st.markdown("#### Count PMF — randomized PIT")
    st.caption(
        "For the count predictions (0, 1, 2, 3, 4, ≥5 bikes), the randomized PIT histogram diagnoses "
        "**shape** of miscalibration. Flat = calibrated; **U** = underdispersed (PMF too tight); **∩** = "
        "overdispersed (PMF too wide); **left-skew** = predicting too few bikes; **right-skew** = too many."
    )
    pit_df = dashboard_metrics.count_pit_histogram(filtered, n_bins=10)
    if pit_df.empty:
        st.info("No model in the filter set ships a count PMF.")
    else:
        st.altair_chart(viz.count_pit_histogram_chart(pit_df), use_container_width=True)

    # ----- Worst station-hour drill-in -----
    st.markdown("#### Worst-offender station-hours")
    worst = dashboard_metrics.worst_station_hours(
        filtered, prob_col=prob_col, outcome_col=outcome_col, top=25,
    )
    if worst.empty:
        st.info("Not enough per-station samples.")
    else:
        st.dataframe(
            worst.rename(columns={
                "model_key": "Model",
                "station_id": "Station ID",
                "station_name": "Station",
                "local_hour": "Hour",
                "n": "N",
                "brier": "Brier",
                "mean_prediction": "Mean pred",
                "observed_rate": "Observed",
                "calibration_gap": "Gap",
            }),
            hide_index=True,
            use_container_width=True,
        )


# =============================================================================
# Tab 5 — Decisions (recommendation-quality view)
# =============================================================================


def _render_decisions(window_hours: int) -> None:
    st.subheader(f"🎯 Decision impact ({window_hours}h window)")
    st.caption(
        "These metrics score the **recommendations** the API drives, not just the probabilities. "
        "A well-calibrated probability is necessary but not sufficient — the wrapping decision logic "
        "(walk-distance penalty, LCB clipping, top-k ordering) has its own knobs."
    )

    perf = _performance_summary(window_hours)
    decisions = perf.get("decision_metrics") or {}
    rec = perf.get("recommended_precision") or {}

    cols = st.columns(3)
    if rec:
        cols[0].metric("Recommended hit rate", f"{rec.get('hit_rate', 0):.1%}")
        cols[0].caption(f"{rec.get('n', 0):,} recommended forecasts")
    else:
        cols[0].metric("Recommended hit rate", "—")
    if decisions.get("active_recommendation_hit_rate") is not None:
        cols[1].metric("Active top pick", f"{decisions['active_recommendation_hit_rate']:.1%}")
    else:
        cols[1].metric("Active top pick", "—")
    if decisions.get("model_specific_top_hit_rate") is not None:
        cols[2].metric("Best-model top pick", f"{decisions['model_specific_top_hit_rate']:.1%}")
    else:
        cols[2].metric("Best-model top pick", "—")

    # ----- Top-k recommendation hit rate -----
    st.markdown("#### Top-k hit rate by model")
    st.caption(
        "Of the requests where the API ranked candidate stations, what fraction had the actually-best "
        "station in its top-1, top-3, or top-5 list?"
    )
    topk = _topk_recommendation(window_hours)
    topk_df = pd.DataFrame(topk.get("model_leaderboard") or [])
    if topk_df.empty:
        st.info("No multi-candidate requests resolved yet.")
    else:
        st.altair_chart(viz.topk_hitrate_chart(topk_df), use_container_width=True)

    # ----- Regret distribution -----
    st.markdown("#### Distance-adjusted regret")
    st.caption(
        "Regret = oracle-best utility − actually-chosen utility. The chosen station had a "
        "walk-distance penalty applied; the oracle is whoever maximized utility in hindsight."
    )
    # Pull recommendation_outcomes for regret detail.
    try:
        with db.session(read_only=True) as conn:
            regret_df = conn.execute(
                """
                SELECT model_key, distance_adjusted_regret, distance_km, observed_has_ebike
                FROM recommendation_outcomes
                WHERE created_at > now() - (? * INTERVAL '1 hour')
                  AND distance_adjusted_regret IS NOT NULL
                """,
                [window_hours],
            ).df()
    except duckdb.Error:
        regret_df = pd.DataFrame()
    if regret_df.empty:
        st.info("No regret samples yet — the resolver populates this table from outcomes.")
    else:
        regret_df["model_label"] = regret_df["model_key"]
        st.altair_chart(viz.regret_distribution_chart(regret_df), use_container_width=True)

    # ----- Multi-bike plan performance -----
    st.markdown("#### Multi-bike plan performance")
    st.caption(
        "Per-plan-size leaderboards. Multi-bike plans are scored across **all** candidate forecasts "
        "in the request (not just the picked stops — that pool is too small for stable Brier estimates)."
    )
    multi = _multi_bike_performance(window_hours)
    by_size = multi.get("by_plan_size") or {}
    if not any(v.get("model_leaderboard") for v in by_size.values()):
        st.info("No multi-bike requests in window.")
        return
    plan_sizes = sorted(by_size.keys())
    tabs = st.tabs([f"k={k} bikes" for k in plan_sizes])
    for tab, k in zip(tabs, plan_sizes):
        with tab:
            slice_data = by_size[k]
            board = slice_data.get("model_leaderboard") or []
            n_req = slice_data.get("n_requests", 0)
            st.caption(f"{n_req:,} requests")
            if not board:
                st.info("No requests yet at this plan size.")
            else:
                df = viz.leaderboard_frame(board)
                st.dataframe(df, hide_index=True, use_container_width=True)


# =============================================================================
# Tab 6 — System health (operational view)
# =============================================================================


def _render_system_health() -> None:
    st.subheader("⚙️ System health")
    collector = _collector_health()
    status = _system_status_payload()

    # Top-line freshness
    cols = st.columns(4)
    latest_tick = collector.get("latest_tick_at")
    cols[0].metric(
        "Last collector tick",
        latest_tick.strftime("%H:%M:%S") if isinstance(latest_tick, datetime) else "—",
        f"{_age_minutes(latest_tick):.1f} min ago" if _age_minutes(latest_tick) is not None else None,
    )
    cols[1].metric("Pending queue", f"{collector.get('pending_queue_files', 0):,}")
    cols[2].metric("Failed queue", f"{collector.get('failed_queue_files', 0):,}")
    replica_age = _age_minutes(collector.get("read_replica_updated_at"))
    cols[3].metric(
        "Read replica",
        f"{replica_age:.1f} min" if replica_age is not None else "—",
        help="How stale the read replica is (refreshes every 60s by default).",
    )

    if collector.get("error"):
        st.error(f"Collector health query failed: {collector['error']}")

    # 10-min activity
    st.markdown("#### Last 10 minutes")
    activity = st.columns(4)
    activity[0].metric("Ticks", f"{collector.get('ticks_10m', 0)}")
    activity[1].metric("Station rows", f"{collector.get('station_rows_10m', 0):,}")
    activity[2].metric("Forecasts logged", f"{collector.get('forecasts_10m', 0):,}")
    activity[3].metric("Outcomes resolved", f"{collector.get('outcomes_10m', 0):,}")

    # Model selection state
    st.markdown("#### Active model state")
    selection = status.get("selection") or {}
    if not selection:
        st.info("No model_selection_state rows yet — promoter hasn't run.")
    else:
        sel_cols = st.columns(3)
        sel_cols[0].metric("Active model", selection.get("active_model_key") or "—")
        sel_cols[1].metric("Best SOTA", selection.get("best_sota_model_key") or "—")
        sel_cols[2].metric("Reason", str(selection.get("reason") or "—"))
        with st.expander("Selection state raw"):
            st.json(selection)

    # Live prediction cache
    cache = status.get("cache") or {}
    if cache:
        st.markdown("#### Live prediction cache")
        cache_cols = st.columns(3)
        cache_cols[0].metric("Stations cached", f"{cache.get('stations_with_predictions', 0):,}")
        cache_cols[1].metric("Models cached", f"{cache.get('distinct_models', 0)}")
        oldest = cache.get("oldest_as_of_minutes")
        if oldest is not None:
            cache_cols[2].metric("Oldest entry", f"{oldest:.1f} min")

    # Recent service heartbeats
    st.markdown("#### Service heartbeats")
    try:
        with db.session(read_only=True) as conn:
            hb = conn.execute(
                "SELECT service_name, heartbeat_at, pid FROM service_heartbeats ORDER BY heartbeat_at DESC"
            ).df()
    except duckdb.Error:
        hb = pd.DataFrame()
    if hb.empty:
        st.info("No heartbeats yet.")
    else:
        hb["age_min"] = hb["heartbeat_at"].apply(_age_minutes)
        st.dataframe(hb, hide_index=True, use_container_width=True)

    # Recent background jobs
    st.markdown("#### Recent background jobs")
    try:
        with db.session(read_only=True) as conn:
            jobs = conn.execute(
                """
                SELECT job_name, status, started_at, duration_seconds, message
                FROM background_job_runs
                ORDER BY started_at DESC LIMIT 25
                """
            ).df()
    except duckdb.Error:
        jobs = pd.DataFrame()
    if jobs.empty:
        st.info("No background job runs recorded yet.")
    else:
        st.dataframe(jobs, hide_index=True, use_container_width=True)

    # Free bike density (light operational sanity)
    st.markdown("#### Live free-floating eBikes")
    free = _free_bikes()
    if free.empty:
        st.caption("Couldn't fetch the live GBFS feed (or no floaters right now).")
    else:
        st.caption(f"{len(free):,} floating eBikes in the live feed")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    st.set_page_config(page_title="Divvy Observer", layout="wide")
    st.title("🚲 Divvy Observer")

    read_path = (
        config.READ_DB_PATH if config.ENABLE_READ_REPLICA and config.READ_DB_PATH.exists() else config.DB_PATH
    )
    st.caption(f"Reading {read_path}")

    # Global window selector
    with st.sidebar:
        st.header("Window")
        window_hours = st.select_slider(
            "Performance window",
            options=[6, 12, 24, 48, 72, 168],
            value=24,
            format_func=lambda h: f"{h}h" if h < 168 else "1w",
            help="How far back the maintainer tabs aggregate resolved outcomes.",
            key="global_window_hours",
        )

    tab_find, tab_station, tab_perf, tab_calib, tab_decisions, tab_health = st.tabs([
        "🚲 Find a bike",
        "📍 Station detail",
        "🏆 Performance",
        "📐 Calibration",
        "🎯 Decisions",
        "⚙️ System",
    ])

    with tab_find:
        _render_find_a_bike()

    with tab_station:
        _render_station_detail()

    with tab_perf:
        _render_performance(int(window_hours))

    with tab_calib:
        _render_calibration(int(window_hours))

    with tab_decisions:
        _render_decisions(int(window_hours))

    with tab_health:
        _render_system_health()


if __name__ == "__main__":
    main()
