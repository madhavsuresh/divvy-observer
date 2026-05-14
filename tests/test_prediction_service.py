from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import duckdb
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from divvy import db, forecast_queue, inventory_dp, model_eval, predictor, recommendations, tripdata, weather
from divvy.api import app


def _seed_db(path) -> None:
    predictor._MODEL_CACHE.clear()
    conn = duckdb.connect(str(path))
    db.init_schema(conn)
    base = datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0) - timedelta(minutes=35)
    stations = [
        ("s1", None, "S1", "Station One", 41.0000, -87.0000, 15, "classic", base, base),
        ("s2", None, "S2", "Station Two", 41.0040, -87.0040, 15, "classic", base, base),
        ("s3", None, "S3", "Far Station", 41.0500, -87.0500, 15, "classic", base, base),
    ]
    conn.executemany(
        """
        INSERT INTO stations (
          station_id, legacy_id, short_name, name, lat, lon, capacity, station_type,
          first_seen_at, last_seen_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        stations,
    )

    rows = []
    for minute in range(36):
        ts = base + timedelta(minutes=minute)
        s1_ebikes = 0 if minute < 8 else (1 if minute < 24 else 0)
        s2_ebikes = 2 if minute % 9 != 0 else 1
        for station_id, ebikes in [("s1", s1_ebikes), ("s2", s2_ebikes)]:
            rows.append((
                station_id,
                ts,
                ts,
                6 + ebikes,
                ebikes,
                0,
                7,
                0,
                True,
                True,
                True,
            ))
    conn.executemany(
        """
        INSERT INTO station_status (
          station_id, last_reported, fetched_at, num_bikes_available,
          num_ebikes_available, num_bikes_disabled, num_docks_available,
          num_docks_disabled, is_installed, is_renting, is_returning
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.close()


def test_reconstruct_minute_series_piecewise_constant() -> None:
    base = datetime(2026, 1, 1, 12, 0, 0)
    status = pd.DataFrame(
        [
            {"station_id": "s1", "last_reported": base, "num_ebikes_available": 0},
            {"station_id": "s1", "last_reported": base + timedelta(minutes=2), "num_ebikes_available": 1},
        ]
    )
    reconstructed = predictor.reconstruct_minute_series(status)
    values = reconstructed.set_index("ts")["num_ebikes_available"].to_dict()
    assert values[pd.Timestamp(base + timedelta(minutes=1))] == 0
    assert values[pd.Timestamp(base + timedelta(minutes=2))] == 1


def test_inventory_dp_outputs_constrained_count_distribution() -> None:
    rollout = inventory_dp.rollout_inventory_distribution(
        capacity=6,
        current_ebikes=0,
        current_total_bikes=3,
        ebike_departure_mean=5.0,
        classic_departure_mean=1.0,
        ebike_arrival_mean=1.5,
        classic_arrival_mean=1.0,
    )

    assert sum(rollout.p_count_ebikes.values()) == pytest.approx(1.0)
    assert sum(rollout.p_count_total.values()) == pytest.approx(1.0)
    assert rollout.p_has_ebike == pytest.approx(1.0 - rollout.p_count_ebikes["0"])
    assert rollout.expected_ebike_departures == pytest.approx(0.0)
    assert 0 <= rollout.expected_ebikes <= rollout.expected_total_bikes <= 6
    assert rollout.p_capacity_violation == 0.0


def test_training_examples_include_temporal_features_and_horizon_labels(tmp_path) -> None:
    path = tmp_path / "divvy.duckdb"
    _seed_db(path)
    conn = duckdb.connect(str(path))
    examples = predictor.build_training_examples(conn, history_hours=1, max_source_rows=5000)
    conn.close()

    assert not examples.empty
    assert {5, 10}.issubset(set(examples["horizon_minutes"]))
    assert examples["has_ebike"].isin([0, 1]).all()
    for column in predictor.FEATURE_COLUMNS:
        assert column in examples.columns
    assert examples["station_same_hour_rate"].between(0, 1).all()
    assert examples["nearby_same_hour_rate"].between(0, 1).all()


def test_score_candidates_uses_sparse_history_fallback(tmp_path) -> None:
    path = tmp_path / "divvy.duckdb"
    _seed_db(path)
    conn = duckdb.connect(str(path))
    candidates = predictor.station_candidates(conn, 41.0, -87.0, radius_km=1.5)
    scored, fitted = predictor.score_candidates(conn, candidates)
    conn.close()

    assert len(scored) >= 2
    assert fitted.n_examples > 0
    assert scored["p_has_ebike_5m"].between(0, 1).all()
    assert scored["p_has_ebike_10m"].between(0, 1).all()
    assert set(scored["confidence"]).issubset({"low", "medium", "high"})
    assert scored["walk_adjusted_score"].between(0, 1).all()
    for model_key in predictor.MODEL_KEYS:
        assert scored[f"p_has_ebike_10m_{model_key}"].between(0, 1).all()
    assert predictor.ACTIVE_MODEL_KEY == "cc_nissm"
    assert fitted.active_key == "cc_nissm"
    assert {"cc_nissm", "dg_nissm", "stg_ncde_inventory", "tft_inventory", "inventory_world"}.issubset(set(predictor.MODEL_KEYS))
    assert fitted.models["inventory_world"].method == "distributional_inventory_world_rollout"
    assert fitted.models["inventory_world"].model_version == "inventory-world-v2"
    assert scored["expected_ebikes_10m"].notna().all()
    assert scored["p_dock_constrained_arrival_10m"].between(0, 1).all()
    inv_dist = scored.iloc[0]["p_count_ebikes_10m_inventory_world"]
    assert isinstance(inv_dist, dict)
    assert sum(inv_dist.values()) == pytest.approx(1.0)
    assert scored.iloc[0]["p_has_ebike_10m_inventory_world"] == pytest.approx(1.0 - inv_dist["0"])
    assert fitted.models["stg_ncde_inventory"].method == "stg_ncde_inventory_flow_fallback"


def test_trip_flow_and_weather_features_are_available_to_experimental_models(tmp_path) -> None:
    path = tmp_path / "divvy.duckdb"
    _seed_db(path)
    conn = duckdb.connect(str(path))
    now = datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0)
    local_now = pd.Timestamp(now, tz="UTC").tz_convert("America/Chicago").tz_localize(None).to_pydatetime()
    local_same_hour = local_now.replace(minute=30, second=0, microsecond=0)
    trip_rows = pd.DataFrame(
        [
            {
                "ride_id": "r1",
                "rideable_type": "electric_bike",
                "started_at": (local_same_hour - timedelta(minutes=10)).strftime("%Y-%m-%d %H:%M:%S"),
                "ended_at": local_same_hour.strftime("%Y-%m-%d %H:%M:%S"),
                "start_station_id": "S2",
                "start_station_name": "Station Two",
                "end_station_id": "S1",
                "end_station_name": "Station One",
                "start_lat": 41.004,
                "start_lng": -87.004,
                "end_lat": 41.0,
                "end_lng": -87.0,
                "member_casual": "member",
            },
            {
                "ride_id": "r2",
                "rideable_type": "classic_bike",
                "started_at": (local_same_hour - timedelta(days=7, minutes=10)).strftime("%Y-%m-%d %H:%M:%S"),
                "ended_at": (local_same_hour - timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S"),
                "start_station_id": "S2",
                "start_station_name": "Station Two",
                "end_station_id": "S1",
                "end_station_name": "Station One",
                "start_lat": 41.004,
                "start_lng": -87.004,
                "end_lat": 41.0,
                "end_lng": -87.0,
                "member_casual": "casual",
            },
        ]
    )
    inserted = tripdata.insert_trips(conn, tripdata.normalize_trip_frame(trip_rows))
    flow = tripdata.rebuild_flow_tables(conn)
    weather_rows = pd.DataFrame(
        {
            "observed_at": [(now + timedelta(minutes=10)).replace(minute=0, second=0, microsecond=0)],
            "temperature_2m": [2.0],
            "relative_humidity_2m": [80.0],
            "apparent_temperature": [-6.0],
            "precipitation": [0.8],
            "rain": [0.8],
            "snowfall": [0.0],
            "snow_depth": [0.0],
            "cloud_cover": [90.0],
            "wind_speed_10m": [24.0],
            "wind_gusts_10m": [42.0],
            "weather_code": [61],
            "source": ["fixture"],
            "fetched_at": [now],
        }
    )
    weather.upsert_weather(conn, weather_rows)

    candidates = predictor.station_candidates(conn, 41.0, -87.0, radius_km=1.5)
    scored, _ = predictor.score_candidates(conn, candidates)
    conn.close()

    assert inserted == 2
    assert flow.flow_rows > 0
    assert flow.route_rows > 0
    s1 = scored[scored["station_id"] == "s1"].iloc[0]
    assert s1["route_inbound_trips_same_hour"] >= 1
    assert s1["route_inbound_ebike_share_same_hour"] > 0
    assert s1["weather_bad_conditions"] == 1
    assert s1["p_appears_10m_inventory_world"] is not None
    assert s1["p_learned_10m_inventory_world"] > 0.1
    for model_key in ["random_forest", "gradient_boosting", "inventory_world", "stg_ncde_inventory"]:
        assert scored[f"p_has_ebike_10m_{model_key}"].between(0, 1).all()


def test_weather_upsert_dedupes_duplicate_hours(tmp_path) -> None:
    path = tmp_path / "weather.duckdb"
    conn = duckdb.connect(str(path))
    db.init_schema(conn)
    observed = datetime(2026, 3, 8, 8, 0, 0)
    weather_rows = pd.DataFrame(
        {
            "observed_at": [observed, observed],
            "temperature_2m": [1.0, 3.0],
            "relative_humidity_2m": [60.0, 80.0],
            "apparent_temperature": [0.0, 2.0],
            "precipitation": [0.0, 0.2],
            "rain": [0.0, 0.2],
            "snowfall": [0.0, 0.0],
            "snow_depth": [0.0, 0.0],
            "cloud_cover": [40.0, 80.0],
            "wind_speed_10m": [10.0, 12.0],
            "wind_gusts_10m": [15.0, 18.0],
            "weather_code": [1, 61],
            "source": ["fixture", "fixture"],
            "fetched_at": [observed, observed],
        }
    )

    inserted = weather.upsert_weather(conn, weather_rows)
    row = conn.execute(
        "SELECT COUNT(*), temperature_2m, weather_code FROM weather_hourly GROUP BY temperature_2m, weather_code"
    ).fetchone()
    conn.close()

    assert inserted == 1
    assert row == (1, 2.0, 61)


def test_walk_adjusted_score_can_favor_a_closer_station() -> None:
    scored = pd.DataFrame(
        [
            {
                "station_id": "close",
                "p_has_ebike_5m": 0.76,
                "p_has_ebike_10m": 0.76,
                "distance_km": 0.10,
                "num_ebikes_available": 1,
            },
            {
                "station_id": "far",
                "p_has_ebike_5m": 0.85,
                "p_has_ebike_10m": 0.85,
                "distance_km": 1.00,
                "num_ebikes_available": 3,
            },
        ]
    )

    ranked = predictor.apply_walk_adjusted_scores(scored, search_radius_km=1.5).sort_values(
        "walk_adjusted_score",
        ascending=False,
    )

    assert ranked.iloc[0]["station_id"] == "close"
    assert ranked.iloc[0]["rank_probability"] < ranked.iloc[1]["rank_probability"]


def test_forecast_logging_resolution_and_metrics(tmp_path) -> None:
    path = tmp_path / "divvy.duckdb"
    _seed_db(path)
    conn = duckdb.connect(str(path))
    candidates = predictor.station_candidates(conn, 41.0, -87.0, radius_km=1.5)
    scored, _ = predictor.score_candidates(conn, candidates)
    scored["forecasted_at"] = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=20)

    logged = model_eval.log_forecasts(
        conn,
        scored,
        user_lat=41.0,
        user_lon=-87.0,
        request_id="test-request",
        recommended_station_ids=[scored.iloc[0]["station_id"]],
    )
    resolved = model_eval.resolve_due_outcomes(conn, tolerance_minutes=20)
    summary = model_eval.performance_summary(conn, window_hours=1, resolve=False)
    inventory_row = conn.execute(
        """
        SELECT p_count_ebikes_json, expected_ebikes
        FROM model_forecasts
        WHERE model_key = 'inventory_world'
          AND p_count_ebikes_json IS NOT NULL
        LIMIT 1
        """
    ).fetchone()
    outcome_row = conn.execute(
        """
        SELECT count_log_prob, crps, observed_total_bikes, observed_docks
        FROM model_outcomes o
        JOIN model_forecasts f USING (forecast_id)
        WHERE f.model_key = 'inventory_world'
        LIMIT 1
        """
    ).fetchone()
    conn.close()

    assert logged == len(scored) * len(predictor.HORIZONS) * len(predictor.MODEL_KEYS)
    assert 0 < resolved <= logged
    assert summary["overall"]["n"] == resolved
    assert summary["overall"]["brier_score"] is not None
    assert summary["overall"]["count_log_loss"] is not None
    assert summary["overall"]["crps"] is not None
    assert summary["coherence"]["capacity_violation_rate"] == 0.0
    assert summary["recommended_precision"]["n"] > 0
    assert len(summary["model_leaderboard"]) == len(predictor.MODEL_KEYS)
    assert inventory_row is not None
    assert sum(json.loads(inventory_row[0]).values()) == pytest.approx(1.0)
    assert inventory_row[1] is not None
    assert outcome_row is not None
    assert outcome_row[0] is not None
    assert outcome_row[1] is not None
    assert outcome_row[2] is not None
    assert outcome_row[3] is not None


def test_recommendation_api_logs_forecasts_with_mocked_free_bikes(tmp_path, monkeypatch) -> None:
    path = tmp_path / "api.duckdb"
    _seed_db(path)
    monkeypatch.setattr(db.config, "DB_PATH", path)
    monkeypatch.setattr(db.config, "READ_DB_PATH", tmp_path / "api_read.duckdb")
    monkeypatch.setattr(db.config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(db.config, "FORECAST_QUEUE_DIR", tmp_path / "forecast_queue")
    monkeypatch.setattr(
        recommendations,
        "fetch_live_free_bikes",
        lambda lat, lon, radius_km=1.5: {
            "bike_id": "bike-1",
            "name": "1001",
            "lat": lat,
            "lon": lon,
            "distance_km": 0.0,
        },
    )

    client = TestClient(app)
    response = client.post(
        "/api/v1/recommendations",
        json={
            "lat": 41.0,
            "lon": -87.0,
            "near_radius_km": 0.5,
            "search_radius_km": 1.5,
            "place_label": "fixture place",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["best_practical_station_5_10m"]["station_id"] in {"s1", "s2"}
    assert data["highest_probability_station_5_10m"]["station_id"] in {"s1", "s2"}
    assert data["best_station_5_10m"] == data["best_practical_station_5_10m"]
    assert "ranking_formula" in data
    assert len(data["best_station_5_10m"]["model_predictions"]) == len(predictor.MODEL_KEYS)
    assert data["best_station_5_10m"]["expected_ebikes_10m"] is not None
    inventory_prediction = next(
        item for item in data["best_station_5_10m"]["model_predictions"]
        if item["model_key"] == "inventory_world"
    )
    assert inventory_prediction["p_count_ebikes_10m"] is not None
    assert data["closest_free_ebike"]["bike_id"] == "bike-1"
    assert data["forecast_rows_logged"] == 0
    assert data["forecast_rows_queued"] == data["candidate_count"] * len(predictor.HORIZONS) * len(predictor.MODEL_KEYS)
    assert data["forecast_logging_mode"] == "queued_for_collector"
    conn = duckdb.connect(str(path))
    drained = forecast_queue.drain_forecast_queue(conn)
    query_row = conn.execute("SELECT query_label, query_place_key FROM prediction_queries").fetchone()
    forecast_count = conn.execute("SELECT COUNT(*) FROM model_forecasts").fetchone()[0]
    inventory_forecast = conn.execute(
        """
        SELECT p_count_ebikes_json
        FROM model_forecasts
        WHERE model_key = 'inventory_world'
          AND p_count_ebikes_json IS NOT NULL
        LIMIT 1
        """
    ).fetchone()
    conn.close()
    assert drained["forecast_rows_logged"] == data["forecast_rows_queued"]
    assert forecast_count == data["forecast_rows_queued"]
    assert query_row == ("fixture place", "41.000,-87.000")
    assert inventory_forecast is not None
    assert sum(json.loads(inventory_forecast[0]).values()) == pytest.approx(1.0)
