from __future__ import annotations

from datetime import datetime, timedelta, timezone

import duckdb
import pytest
from fastapi.testclient import TestClient

from divvy import db, recommendations, tile, tile_predictor
from divvy.api import app


LOOP_LAT = 41.8819
LOOP_LON = -87.6278


def _seed_db(path):
    conn = duckdb.connect(str(path))
    db.init_schema(conn)
    now = datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0)
    conn.execute(
        """
        INSERT INTO model_selection_state
          (computed_at, active_model_key, active_artifact_id, active_model_source,
           best_evaluated_model_key, best_sota_model_key, best_baseline_model_key,
           active_equals_best, selection_metric, selection_window_hours, min_resolved,
           reason, metrics_json)
        VALUES (now(), 'logistic', NULL, 'test', NULL, NULL, NULL, true, 'rank_loss', 24, 0, 'fixture', '{}')
        """
    )
    conn.execute(
        """
        INSERT INTO stations VALUES
        ('s_loop', NULL, 'S1', 'Loop Station', ?, ?, 27, 'classic', ?, ?)
        """,
        [LOOP_LAT, LOOP_LON, now, now],
    )
    last_reported = now - timedelta(minutes=2)
    conn.execute(
        """
        INSERT INTO station_status VALUES
        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ["s_loop", last_reported, last_reported, 11, 3, 0, 16, 0, True, True, True],
    )
    tile_id = tile.tile_id_for(LOOP_LAT, LOOP_LON)
    conn.execute(
        """
        INSERT INTO free_bike_status
          (bike_id, fetched_at, name, lat, lon, is_reserved, is_disabled, tile_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            "bike-1",
            now - timedelta(minutes=5),
            "1001",
            LOOP_LAT,
            LOOP_LON,
            False,
            False,
            tile_id,
        ],
    )
    # bike-2: was free 4 min ago, became reserved 1 min ago — counts as a fresh-reservation event
    # within the last 5 min and shows up in current_reserved_free_ebikes.
    conn.execute(
        """
        INSERT INTO free_bike_status
          (bike_id, fetched_at, name, lat, lon, is_reserved, is_disabled, tile_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ["bike-2", now - timedelta(minutes=4), "1002", LOOP_LAT, LOOP_LON, False, False, tile_id],
    )
    conn.execute(
        """
        INSERT INTO free_bike_status
          (bike_id, fetched_at, name, lat, lon, is_reserved, is_disabled, tile_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ["bike-2", now - timedelta(minutes=1), "1002", LOOP_LAT, LOOP_LON, True, False, tile_id],
    )
    conn.close()


def test_tile_state_endpoint_returns_combined_free_and_dock_blocks(tmp_path, monkeypatch) -> None:
    path = tmp_path / "api.duckdb"
    _seed_db(path)
    monkeypatch.setattr(db.config, "DB_PATH", path)
    monkeypatch.setattr(db.config, "READ_DB_PATH", tmp_path / "api_read.duckdb")
    monkeypatch.setattr(db.config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(db.config, "FORECAST_QUEUE_DIR", tmp_path / "forecast_queue")
    monkeypatch.setattr(recommendations, "fetch_live_free_bike_ids", lambda: {"bike-1", "bike-2"})
    tile_predictor._stations_cache.clear()

    client = TestClient(app)
    response = client.get(
        "/api/v1/free_ebike/tile_state",
        params={"lat": LOOP_LAT, "lon": LOOP_LON, "radius_km": 0.3},
    )
    assert response.status_code == 200, response.text

    data = response.json()
    assert data["center"] == {"lat": LOOP_LAT, "lon": LOOP_LON}
    assert data["radius_km"] == 0.3
    assert len(data["tiles"]) >= 1

    loop_tile_id = tile.tile_id_for(LOOP_LAT, LOOP_LON)
    loop_tile = next(t for t in data["tiles"] if t["tile_id"] == loop_tile_id)

    assert loop_tile["current_free_ebikes"] == 1
    assert loop_tile["current_reserved_free_ebikes"] == 1
    assert loop_tile["current_docked_ebikes"] == 3
    assert loop_tile["n_stations_in_tile"] == 1
    assert len(loop_tile["bikes"]) == 1
    assert loop_tile["bikes"][0]["bike_id"] == "bike-1"
    p_stays = loop_tile["bikes"][0]["p_stays"]
    assert set(p_stays.keys()) == {"5", "10", "15", "20"}
    for v in p_stays.values():
        assert 0.0 <= v <= 1.0

    # Reservation surface: bike-2 was just reserved, so a fresh-reservation event was logged.
    assert loop_tile["reservation_events_5m"] == 1
    assert loop_tile["reservation_events_30m"] == 1
    assert len(loop_tile["reserved_bikes"]) == 1
    assert loop_tile["reserved_bikes"][0]["bike_id"] == "bike-2"

    assert "dock_reservations" in data["notes"]
    assert len(loop_tile["stations"]) == 1
    assert loop_tile["stations"][0]["station_id"] == "s_loop"

    h10 = loop_tile["horizons"]["10"]
    assert {"free", "docked", "combined"} == set(h10.keys())
    free_p = h10["free"]["p_has_bike"]
    dock_p = h10["docked"]["p_any_has_ebike"]
    combined_p = h10["combined"]["p_any_ebike"]
    # Combined = 1 - (1 - free) * (1 - dock). Without a dock prediction seeded,
    # dock_p is 0 and combined collapses to free_p.
    assert combined_p == pytest.approx(1.0 - (1.0 - free_p) * (1.0 - dock_p))


def test_recommendation_attaches_tile_dwell_probability(tmp_path, monkeypatch) -> None:
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
    tile_predictor._stations_cache.clear()

    client = TestClient(app)
    response = client.post(
        "/api/v1/recommendations",
        json={
            "lat": LOOP_LAT,
            "lon": LOOP_LON,
            "near_radius_km": 0.5,
            "search_radius_km": 1.5,
        },
    )
    assert response.status_code == 200, response.text
    data = response.json()
    closest = data.get("closest_free_ebike")
    assert closest is not None
    assert closest["bike_id"] == "bike-1"
    assert closest["tile_id"] == tile.tile_id_for(LOOP_LAT, LOOP_LON)
    dwell = closest["tile_dwell_probability"]
    assert set(dwell.keys()) == {"5", "10", "15", "20"}
    for v in dwell.values():
        assert 0.0 <= v <= 1.0
