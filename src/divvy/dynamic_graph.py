from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import duckdb
import numpy as np
import pandas as pd


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _haversine(lat1: float, lon1: float, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    lat1r = np.radians(lat1)
    lat2 = np.radians(lats.astype(float))
    dlat = lat2 - lat1r
    dlon = np.radians(lons.astype(float) - lon1)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * 6371.0 * np.arcsin(np.sqrt(a))


def _station_rows(conn: duckdb.DuckDBPyConnection, station_ids: list[str] | None) -> pd.DataFrame:
    params: list[object] = []
    station_filter = ""
    if station_ids:
        station_filter = "AND station_id IN (" + ",".join(["?"] * len(station_ids)) + ")"
        params.extend(station_ids)
    return conn.execute(
        f"""
        SELECT station_id, lat, lon
        FROM stations
        WHERE lat IS NOT NULL AND lon IS NOT NULL
          {station_filter}
        """,
        params,
    ).df()


def build_dynamic_graph_edges(
    conn: duckdb.DuckDBPyConnection,
    anchor_ts,
    station_ids=None,
    lookback_days=30,
    top_k=16,
    horizons=(5, 10, 15, 20),
    graph_key=None,
) -> pd.DataFrame:
    """Build sparse top-k dynamic graph edges for station inventory models."""
    anchor = pd.Timestamp(anchor_ts or _utc_now()).to_pydatetime()
    lookback_start = anchor - timedelta(days=int(lookback_days))
    graph_key = graph_key or f"graph-{uuid.uuid4().hex}"
    station_ids_list = [str(s) for s in station_ids] if station_ids is not None else None
    stations = _station_rows(conn, station_ids_list)
    if stations.empty:
        return pd.DataFrame(columns=[
            "graph_key", "anchor_ts", "relation", "src_station_id", "dst_station_id",
            "horizon_minutes", "weight", "edge_rank", "distance_km",
            "median_duration_minutes", "lookback_start", "lookback_end",
        ])

    ids = stations["station_id"].astype(str).to_numpy()
    lats = stations["lat"].to_numpy(dtype=float)
    lons = stations["lon"].to_numpy(dtype=float)
    rows: list[dict] = []
    for idx, dst in enumerate(ids):
        distances = _haversine(float(lats[idx]), float(lons[idx]), lats, lons)
        order = np.argsort(distances)
        rank = 0
        for src_idx in order:
            if src_idx == idx:
                continue
            rank += 1
            if rank > int(top_k):
                break
            distance = float(distances[src_idx])
            rows.append({
                "graph_key": graph_key,
                "anchor_ts": anchor,
                "relation": "distance",
                "src_station_id": str(ids[src_idx]),
                "dst_station_id": str(dst),
                "horizon_minutes": None,
                "weight": float(1.0 / max(0.05, distance)),
                "edge_rank": rank,
                "distance_km": distance,
                "median_duration_minutes": None,
                "lookback_start": lookback_start,
                "lookback_end": anchor,
            })

    try:
        route_rows = conn.execute(
            """
            SELECT
              start_station_id AS src_station_id,
              end_station_id AS dst_station_id,
              SUM(trips) AS trips,
              AVG(median_duration_minutes) AS median_duration_minutes
            FROM station_trip_routes
            GROUP BY start_station_id, end_station_id
            """
        ).df()
    except Exception:
        route_rows = pd.DataFrame()
    if not route_rows.empty:
        route_rows = route_rows.dropna(subset=["src_station_id", "dst_station_id"])
        if station_ids_list:
            route_rows = route_rows[route_rows["dst_station_id"].astype(str).isin(station_ids_list)]
        for dst, group in route_rows.groupby("dst_station_id"):
            g = group.sort_values("trips", ascending=False).head(int(top_k)).reset_index(drop=True)
            for rank, row in enumerate(g.itertuples(index=False), start=1):
                distance = None
                src_match = stations[stations["station_id"].astype(str) == str(row.src_station_id)]
                dst_match = stations[stations["station_id"].astype(str) == str(dst)]
                if not src_match.empty and not dst_match.empty:
                    distance = float(_haversine(float(dst_match.iloc[0]["lat"]), float(dst_match.iloc[0]["lon"]), src_match["lat"].to_numpy(), src_match["lon"].to_numpy())[0])
                for horizon in horizons:
                    rows.append({
                        "graph_key": graph_key,
                        "anchor_ts": anchor,
                        "relation": "od_flow",
                        "src_station_id": str(row.src_station_id),
                        "dst_station_id": str(dst),
                        "horizon_minutes": int(horizon),
                        "weight": float(row.trips),
                        "edge_rank": rank,
                        "distance_km": distance,
                        "median_duration_minutes": float(row.median_duration_minutes) if pd.notna(row.median_duration_minutes) else None,
                        "lookback_start": lookback_start,
                        "lookback_end": anchor,
                    })

    edges = pd.DataFrame(rows)
    if edges.empty:
        return edges
    edges["weight"] = pd.to_numeric(edges["weight"], errors="coerce").fillna(0.0)
    max_by_relation = edges.groupby("relation")["weight"].transform("max").replace(0.0, 1.0)
    edges["weight"] = (edges["weight"] / max_by_relation).clip(0.0, 1.0)
    return edges


def refresh_dynamic_graph_cache(
    conn: duckdb.DuckDBPyConnection,
    lookback_days: int = 30,
    top_k: int = 16,
    horizons=(5, 10, 15, 20),
) -> dict:
    from . import db

    db.init_schema(conn)
    anchor = _utc_now()
    graph_key = f"graph-{uuid.uuid4().hex}"
    edges = build_dynamic_graph_edges(
        conn,
        anchor_ts=anchor,
        lookback_days=lookback_days,
        top_k=top_k,
        horizons=horizons,
        graph_key=graph_key,
    )
    if edges.empty:
        return {"status": "no_edges", "graph_key": graph_key, "edges": 0}
    rows = [
        (
            row.graph_key,
            row.anchor_ts,
            row.relation,
            row.src_station_id,
            row.dst_station_id,
            int(row.horizon_minutes) if pd.notna(row.horizon_minutes) else None,
            float(row.weight),
            int(row.edge_rank),
            float(row.distance_km) if pd.notna(row.distance_km) else None,
            float(row.median_duration_minutes) if pd.notna(row.median_duration_minutes) else None,
            row.lookback_start,
            row.lookback_end,
            anchor,
        )
        for row in edges.itertuples(index=False)
    ]
    conn.executemany(
        """
        INSERT INTO dynamic_graph_edges (
          graph_key, anchor_ts, relation, src_station_id, dst_station_id,
          horizon_minutes, weight, edge_rank, distance_km,
          median_duration_minutes, lookback_start, lookback_end, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    return {"status": "ok", "graph_key": graph_key, "edges": len(rows)}
