"""H3 hex tile arithmetic for the free-ebike persistence model.

H3 resolution 9: ~174 m edge length, ~0.1 km² per cell. Cell IDs are stored as
hex strings (e.g. ``"8928308280fffff"``) — matches pydeck's H3HexagonLayer
input and the h3-py public API.
"""
from __future__ import annotations

import math

import h3

TILE_RES = 9

# Centroid-to-centroid distance between adjacent res-9 hexes (≈ 2*edge*cos(30°)).
# Used to translate a radius-in-km into a grid_disk k-ring.
_EDGE_KM = h3.average_hexagon_edge_length(TILE_RES, unit="km")
_CENTROID_STEP_KM = _EDGE_KM * math.sqrt(3.0)


def tile_id_for(lat: float | None, lon: float | None) -> str | None:
    if lat is None or lon is None:
        return None
    try:
        flat = float(lat)
        flon = float(lon)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(flat) and math.isfinite(flon)):
        return None
    return h3.latlng_to_cell(flat, flon, TILE_RES)


def tile_center(tile_id: str) -> tuple[float, float]:
    return h3.cell_to_latlng(tile_id)


def tile_boundary(tile_id: str) -> list[tuple[float, float]]:
    return list(h3.cell_to_boundary(tile_id))


def tile_neighbors(tile_id: str, k: int = 1) -> list[str]:
    return list(h3.grid_disk(tile_id, k))


def tile_ids_in_disk(lat: float, lon: float, radius_km: float) -> list[str]:
    """All res-9 cells whose centroid is within `radius_km` of (lat, lon).

    Conservative: returns every cell in the smallest k-ring that covers the disk,
    so the caller may need to post-filter if exact containment matters.
    """
    if radius_km <= 0:
        center = tile_id_for(lat, lon)
        return [center] if center is not None else []
    k = max(1, int(math.ceil(radius_km / _CENTROID_STEP_KM)))
    center = tile_id_for(lat, lon)
    if center is None:
        return []
    return list(h3.grid_disk(center, k))
