from __future__ import annotations

import math

from divvy import tile


LOOP_LAT = 41.8819
LOOP_LON = -87.6278


def test_tile_id_for_returns_stable_hex_string() -> None:
    tile_id = tile.tile_id_for(LOOP_LAT, LOOP_LON)
    assert isinstance(tile_id, str)
    assert len(tile_id) == 15  # H3 res-9 hex strings
    # determinism: same input -> same id
    assert tile.tile_id_for(LOOP_LAT, LOOP_LON) == tile_id


def test_tile_id_for_rejects_missing_and_nonfinite() -> None:
    assert tile.tile_id_for(None, LOOP_LON) is None
    assert tile.tile_id_for(LOOP_LAT, None) is None
    assert tile.tile_id_for(float("nan"), LOOP_LON) is None
    assert tile.tile_id_for(LOOP_LAT, float("inf")) is None


def test_tile_center_round_trips_through_tile_id_for() -> None:
    tile_id = tile.tile_id_for(LOOP_LAT, LOOP_LON)
    assert tile_id is not None
    clat, clon = tile.tile_center(tile_id)
    # Centroid should be within one cell radius (~150 m) of the requested point.
    dlat_m = (clat - LOOP_LAT) * 111_000
    dlon_m = (clon - LOOP_LON) * 111_000 * math.cos(math.radians(LOOP_LAT))
    assert math.hypot(dlat_m, dlon_m) < 200
    # And it must map back to the same tile.
    assert tile.tile_id_for(clat, clon) == tile_id


def test_tile_boundary_returns_six_points() -> None:
    tile_id = tile.tile_id_for(LOOP_LAT, LOOP_LON)
    boundary = tile.tile_boundary(tile_id)
    assert len(boundary) == 6
    for lat, lon in boundary:
        assert isinstance(lat, float) and isinstance(lon, float)
        assert -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0


def test_tile_neighbors_k1_returns_center_and_six_neighbors() -> None:
    tile_id = tile.tile_id_for(LOOP_LAT, LOOP_LON)
    neighbors = tile.tile_neighbors(tile_id, k=1)
    assert len(neighbors) == 7
    assert tile_id in neighbors


def test_tile_ids_in_disk_grows_with_radius() -> None:
    small = tile.tile_ids_in_disk(LOOP_LAT, LOOP_LON, radius_km=0.2)
    medium = tile.tile_ids_in_disk(LOOP_LAT, LOOP_LON, radius_km=0.5)
    big = tile.tile_ids_in_disk(LOOP_LAT, LOOP_LON, radius_km=1.5)
    assert len(small) < len(medium) < len(big)
    # Center tile must always be included.
    center = tile.tile_id_for(LOOP_LAT, LOOP_LON)
    assert center in small
    assert center in medium
    assert center in big


def test_tile_ids_in_disk_zero_radius_returns_just_center() -> None:
    center = tile.tile_id_for(LOOP_LAT, LOOP_LON)
    assert tile.tile_ids_in_disk(LOOP_LAT, LOOP_LON, radius_km=0.0) == [center]
