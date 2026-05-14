from __future__ import annotations

from datetime import datetime, timedelta, timezone

import duckdb
from fastapi.testclient import TestClient

from divvy import db, tile, tile_predictor
from divvy.api import app


LOOP_LAT = 41.8819
LOOP_LON = -87.6278


def _seed_db(path):
    conn = duckdb.connect(str(path))
    db.init_schema(conn)
    now = datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0)
    # A station inside the LOOP tile with a non-zero num_bikes_disabled history
    # and a disabled dock.
    conn.execute(
        """
        INSERT INTO stations VALUES
        ('s_loop', NULL, 'S1', 'Loop Station', ?, ?, 27, 'classic', ?, ?)
        """,
        [LOOP_LAT, LOOP_LON, now, now],
    )
    # Three station_status rows so we observe transitions:
    #   0 → 2 disabled (disability_events += 2)
    #   2 → 1 disabled (repair_events += 1, last state = 1)
    for offset_h, n_dis, n_dock_dis in [(4, 0, 0), (2, 2, 1), (1, 1, 1)]:
        ts = now - timedelta(hours=offset_h)
        conn.execute(
            """
            INSERT INTO station_status VALUES
            (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ["s_loop", ts, ts, 11, 3, n_dis, 16, n_dock_dis, True, True, True],
        )
    # A free-floating bike that was free 1 hour ago, then disabled 30 minutes ago.
    tile_id = tile.tile_id_for(LOOP_LAT, LOOP_LON)
    conn.execute(
        """
        INSERT INTO free_bike_status
          (bike_id, fetched_at, name, lat, lon, is_reserved, is_disabled, tile_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ["bike-d", now - timedelta(hours=1), "1001", LOOP_LAT, LOOP_LON, False, False, tile_id],
    )
    conn.execute(
        """
        INSERT INTO free_bike_status
          (bike_id, fetched_at, name, lat, lon, is_reserved, is_disabled, tile_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ["bike-d", now - timedelta(minutes=30), "1001", LOOP_LAT, LOOP_LON, False, True, tile_id],
    )
    conn.close()


def test_disabled_tile_state_endpoint_returns_state_and_horizons(tmp_path, monkeypatch) -> None:
    path = tmp_path / "api.duckdb"
    _seed_db(path)
    monkeypatch.setattr(db.config, "DB_PATH", path)
    monkeypatch.setattr(db.config, "READ_DB_PATH", tmp_path / "api_read.duckdb")
    monkeypatch.setattr(db.config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(db.config, "FORECAST_QUEUE_DIR", tmp_path / "forecast_queue")
    tile_predictor._stations_cache.clear()

    client = TestClient(app)
    response = client.get(
        "/api/v1/disabled_bike/tile_state",
        params={"lat": LOOP_LAT, "lon": LOOP_LON, "radius_km": 0.3},
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["center"] == {"lat": LOOP_LAT, "lon": LOOP_LON}
    assert data["radius_km"] == 0.3
    assert "docked_bike_id_unknown" in data["notes"]
    assert "free_floating_undercount" in data["notes"]
    assert "repair_model" in data["notes"]
    assert data["horizons_hours"] == [1.0, 6.0, 24.0]

    loop_tile_id = tile.tile_id_for(LOOP_LAT, LOOP_LON)
    loop_tile = next(t for t in data["tiles"] if t["tile_id"] == loop_tile_id)
    # 1 disabled free bike (bike-d), 1 disabled docked bike (latest state of s_loop).
    assert loop_tile["current_disabled_free_ebikes"] == 1
    assert loop_tile["current_disabled_docked_bikes"] == 1
    assert loop_tile["current_disabled_docks"] == 1
    # Free-floating dwell ~30 min for our one disabled bike.
    assert loop_tile["median_free_dwell_minutes"] > 25
    assert loop_tile["median_free_dwell_minutes"] < 35
    # Disability events (2 dock-bike + 1 free) and repair events (1 dock-bike).
    assert loop_tile["disability_events_24h"] >= 2
    assert loop_tile["repair_events_24h"] >= 1

    # Disabled-bike list surfaces the single bike with its disabled_since.
    free_bikes = loop_tile["disabled_free_bikes"]
    assert len(free_bikes) == 1
    assert free_bikes[0]["bike_id"] == "bike-d"
    assert free_bikes[0]["disabled_since"] is not None

    # Disabled stations surface s_loop with its current count.
    stations = loop_tile["disabled_stations"]
    assert len(stations) == 1
    assert stations[0]["station_id"] == "s_loop"
    assert stations[0]["num_bikes_disabled"] == 1
    assert stations[0]["num_docks_disabled"] == 1

    # Horizons present at all three expected values.
    horizons = loop_tile["horizons"]
    assert set(horizons.keys()) == {"1.0", "6.0", "24.0"}
    for payload in horizons.values():
        assert 0.0 <= payload["p_any_repair"] <= 1.0
        assert 0.0 <= payload["p_all_repaired"] <= 1.0
        assert payload["expected_repairs"] >= 0.0
        assert payload["n_disabled"] == 2  # 1 free + 1 docked


def test_disabled_tile_state_endpoint_empty_when_no_data(tmp_path, monkeypatch) -> None:
    # Schema only; no rows seeded.
    path = tmp_path / "api.duckdb"
    conn = duckdb.connect(str(path))
    db.init_schema(conn)
    conn.close()
    monkeypatch.setattr(db.config, "DB_PATH", path)
    monkeypatch.setattr(db.config, "READ_DB_PATH", tmp_path / "api_read.duckdb")
    monkeypatch.setattr(db.config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(db.config, "FORECAST_QUEUE_DIR", tmp_path / "forecast_queue")
    tile_predictor._stations_cache.clear()

    client = TestClient(app)
    response = client.get(
        "/api/v1/disabled_bike/tile_state",
        params={"lat": LOOP_LAT, "lon": LOOP_LON, "radius_km": 0.3},
    )
    assert response.status_code == 200, response.text
    data = response.json()
    # No stations, no free bikes → every hex returns zero state.
    assert all(t["current_disabled_free_ebikes"] == 0 for t in data["tiles"])
    assert all(t["current_disabled_docked_bikes"] == 0 for t in data["tiles"])
