from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat()


def _haversine(lat1: float, lon1: float, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    lat1r = np.radians(float(lat1))
    lat2 = np.radians(lats.astype(float))
    dlat = lat2 - lat1r
    dlon = np.radians(lons.astype(float) - float(lon1))
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * 6371.0 * np.arcsin(np.sqrt(a))


def _normalize_inbound(edges: pd.DataFrame) -> pd.DataFrame:
    if edges.empty:
        return edges
    out = edges.copy()
    sums = out.groupby("dst")["weight"].transform("sum").replace(0.0, np.nan)
    out["weight"] = (out["weight"] / sums).fillna(0.0).clip(0.0, 1.0)
    return out[out["weight"] > 0.0].reset_index(drop=True)


def build_graph_cache_from_examples(
    examples: pd.DataFrame,
    *,
    station_id_to_idx: dict[str, int] | None = None,
    top_k: int = 16,
    semantic_top_k: int | None = None,
    tau_km: float = 0.75,
) -> dict[str, Any]:
    """Build a compact multi-graph cache from training examples.

    The semantic graph uses only rows supplied to this function. Callers should
    pass training rows, not validation/test rows, when constructing artifacts.
    """
    if examples.empty or "station_id" not in examples.columns:
        return empty_graph_cache(station_id_to_idx or {})
    if station_id_to_idx is None:
        station_ids = sorted(examples["station_id"].astype(str).dropna().unique().tolist())
        station_id_to_idx = {station_id: idx for idx, station_id in enumerate(station_ids)}
    else:
        station_ids = [station_id for station_id, _ in sorted(station_id_to_idx.items(), key=lambda kv: kv[1])]
    top_k = max(1, int(top_k))
    semantic_top_k = max(1, int(semantic_top_k or top_k))

    edge_frames: dict[str, pd.DataFrame] = {}
    meta_cols = [c for c in ["station_id", "lat", "lon"] if c in examples.columns]
    if {"station_id", "lat", "lon"}.issubset(meta_cols):
        stations = (
            examples[["station_id", "lat", "lon"]]
            .dropna()
            .drop_duplicates("station_id")
            .copy()
        )
        stations["station_id"] = stations["station_id"].astype(str)
        stations = stations[stations["station_id"].isin(station_id_to_idx)]
        ids = stations["station_id"].to_numpy(dtype=str)
        lats = pd.to_numeric(stations["lat"], errors="coerce").to_numpy(dtype=float)
        lons = pd.to_numeric(stations["lon"], errors="coerce").to_numpy(dtype=float)
        rows: list[dict[str, float | int]] = []
        for dst_pos, dst_station_id in enumerate(ids):
            distances = _haversine(float(lats[dst_pos]), float(lons[dst_pos]), lats, lons)
            order = np.argsort(distances)
            rank = 0
            for src_pos in order:
                if src_pos == dst_pos or not math.isfinite(float(distances[src_pos])):
                    continue
                rank += 1
                if rank > top_k:
                    break
                rows.append({
                    "src": int(station_id_to_idx[str(ids[src_pos])]),
                    "dst": int(station_id_to_idx[str(dst_station_id)]),
                    "weight": float(math.exp(-float(distances[src_pos]) / max(1e-3, tau_km))),
                })
        edge_frames["distance"] = _normalize_inbound(pd.DataFrame(rows))

    if {"station_id", "local_hour", "dow", "has_ebike"}.issubset(examples.columns):
        pivot = examples.copy()
        pivot["station_id"] = pivot["station_id"].astype(str)
        pivot = pivot[pivot["station_id"].isin(station_id_to_idx)]
        profile = (
            pivot.groupby(["station_id", "local_hour", "dow"], as_index=False)["has_ebike"]
            .mean()
            .pivot_table(index="station_id", columns=["local_hour", "dow"], values="has_ebike")
            .fillna(0.0)
        )
        rows = []
        if len(profile) >= 2:
            values = profile.to_numpy(dtype=float)
            norms = np.linalg.norm(values, axis=1)
            for dst_pos, dst_station_id in enumerate(profile.index.astype(str)):
                denom = norms[dst_pos] * norms
                sim = np.divide(
                    values @ values[dst_pos],
                    denom,
                    out=np.zeros(len(profile), dtype=float),
                    where=denom > 1e-9,
                )
                order = np.argsort(-sim)
                rank = 0
                for src_pos in order:
                    if src_pos == dst_pos or sim[src_pos] <= 0.0:
                        continue
                    rank += 1
                    if rank > semantic_top_k:
                        break
                    rows.append({
                        "src": int(station_id_to_idx[str(profile.index[src_pos])]),
                        "dst": int(station_id_to_idx[str(dst_station_id)]),
                        "weight": float(sim[src_pos]),
                    })
        edge_frames["semantic"] = _normalize_inbound(pd.DataFrame(rows))

    edge_index_by_type: dict[str, np.ndarray] = {}
    edge_weight_by_type: dict[str, np.ndarray] = {}
    for relation, edges in edge_frames.items():
        if edges.empty:
            continue
        edge_index_by_type[relation] = edges[["src", "dst"]].to_numpy(dtype=np.int64).T
        edge_weight_by_type[relation] = edges["weight"].to_numpy(dtype=np.float32)

    if not edge_index_by_type:
        return empty_graph_cache(station_id_to_idx)
    return {
        "edge_index_by_type": edge_index_by_type,
        "edge_weight_by_type": edge_weight_by_type,
        "station_idx_order": station_ids,
        "created_at": _utc_now_iso(),
        "leakage_policy": "distance graph from station metadata; semantic graph from caller-supplied training rows only",
        "top_k": int(top_k),
    }


def empty_graph_cache(station_id_to_idx: dict[str, int] | None = None) -> dict[str, Any]:
    station_id_to_idx = station_id_to_idx or {}
    station_ids = [station_id for station_id, _ in sorted(station_id_to_idx.items(), key=lambda kv: kv[1])]
    return {
        "edge_index_by_type": {},
        "edge_weight_by_type": {},
        "station_idx_order": station_ids,
        "created_at": _utc_now_iso(),
        "leakage_policy": "empty graph cache",
        "top_k": 0,
    }


def graph_cache_edge_count(graph_cache: dict[str, Any] | None) -> int:
    if not graph_cache:
        return 0
    edges = graph_cache.get("edge_index_by_type") or {}
    total = 0
    for value in edges.values():
        arr = np.asarray(value)
        if arr.ndim == 2:
            total += int(arr.shape[1])
    return total
