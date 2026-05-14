"""Mobility-community detection for the MacFlow-NISSM-lite model.

Builds a partition of Divvy stations into communities derived from OD trip
flows (or from spatial fallback when trip data is missing). The partition is
the substrate for community/role/exchange features that test the world-model
hypothesis that mobility structure improves inventory-constrained availability
prediction beyond station history + geographic neighbors.

Leakage rule: a partition built for a chronological evaluation must use only
trip rows with ``started_at <= training_window_end``.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence

import duckdb
import numpy as np
import pandas as pd


ROLE_CORE = "core"
ROLE_BOUNDARY = "boundary"
ROLE_GATEWAY = "gateway"
ROLE_UNKNOWN = "unknown"

ROLE_TO_INT = {ROLE_CORE: 0, ROLE_BOUNDARY: 1, ROLE_GATEWAY: 2, ROLE_UNKNOWN: 3}
INT_TO_ROLE = {v: k for k, v in ROLE_TO_INT.items()}


@dataclass
class Partition:
    """A frozen snapshot of a station → community mapping with role metadata."""

    partition_id: str
    computed_at: datetime
    source_data_start: datetime | None
    source_data_end: datetime
    algorithm: str
    n_communities: int
    station_to_community: dict[str, int] = field(default_factory=dict)
    station_to_role: dict[str, str] = field(default_factory=dict)
    boundary_score: dict[str, float] = field(default_factory=dict)
    gateway_score: dict[str, float] = field(default_factory=dict)
    inbound_internal_share: dict[str, float] = field(default_factory=dict)
    outbound_internal_share: dict[str, float] = field(default_factory=dict)
    community_to_neighbors: dict[int, list[tuple[int, float]]] = field(default_factory=dict)

    def role_id(self, station_id: str) -> int:
        return ROLE_TO_INT.get(self.station_to_role.get(str(station_id), ROLE_UNKNOWN), 3)

    def community(self, station_id: str) -> int:
        return int(self.station_to_community.get(str(station_id), 0))

    def stations(self) -> list[str]:
        return list(self.station_to_community.keys())

    def community_sizes(self) -> dict[int, int]:
        sizes: dict[int, int] = {}
        for community_id in self.station_to_community.values():
            sizes[int(community_id)] = sizes.get(int(community_id), 0) + 1
        return sizes


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _stable_partition_id(algorithm: str, source_data_end: datetime, station_ids: Sequence[str]) -> str:
    hasher = hashlib.sha1()
    hasher.update(algorithm.encode("utf-8"))
    hasher.update(b"|")
    hasher.update(str(source_data_end).encode("utf-8"))
    hasher.update(b"|")
    for station_id in sorted(station_ids):
        hasher.update(str(station_id).encode("utf-8"))
        hasher.update(b",")
    return hasher.hexdigest()[:16]


def _load_od_edges(
    conn: duckdb.DuckDBPyConnection | None,
    *,
    training_window_end: datetime,
    lookback_days: int,
) -> pd.DataFrame:
    """Return DataFrame[start_station_id, end_station_id, weight] aggregated over the lookback.

    Prefers ``station_trip_routes`` (already aggregated) and falls back to ``divvy_trips``.
    Returns an empty frame when neither table is populated.
    """

    if conn is None:
        return pd.DataFrame(columns=["start_station_id", "end_station_id", "weight"])
    start = training_window_end - pd.Timedelta(days=int(max(1, lookback_days)))
    try:
        flows = conn.execute(
            """
            SELECT
              start_station_id,
              end_station_id,
              SUM(trips) AS weight
            FROM station_trip_routes
            WHERE start_station_id IS NOT NULL AND end_station_id IS NOT NULL
              AND start_station_id != end_station_id
            GROUP BY start_station_id, end_station_id
            """,
        ).df()
    except Exception:
        flows = pd.DataFrame(columns=["start_station_id", "end_station_id", "weight"])
    if not flows.empty and float(flows["weight"].sum()) > 0:
        return flows.reset_index(drop=True)
    try:
        rows = conn.execute(
            """
            SELECT
              start_station_id,
              end_station_id,
              COUNT(*) AS weight
            FROM divvy_trips
            WHERE started_at <= ?
              AND started_at >= ?
              AND start_station_id IS NOT NULL AND end_station_id IS NOT NULL
              AND start_station_id != end_station_id
            GROUP BY start_station_id, end_station_id
            """,
            [training_window_end, start],
        ).df()
    except Exception:
        rows = pd.DataFrame(columns=["start_station_id", "end_station_id", "weight"])
    return rows.reset_index(drop=True) if not rows.empty else rows


def _load_station_coords(conn: duckdb.DuckDBPyConnection | None) -> pd.DataFrame:
    if conn is None:
        return pd.DataFrame(columns=["station_id", "lat", "lon"])
    try:
        return conn.execute(
            "SELECT station_id, lat, lon FROM stations WHERE lat IS NOT NULL AND lon IS NOT NULL"
        ).df()
    except Exception:
        return pd.DataFrame(columns=["station_id", "lat", "lon"])


def _build_symmetric_matrix(
    edges: pd.DataFrame, station_ids: list[str]
) -> tuple[np.ndarray, dict[str, int]]:
    """Return (W_sym, station_index) where W_sym[i,j] = w_ij + w_ji."""

    n = len(station_ids)
    idx = {station_id: i for i, station_id in enumerate(station_ids)}
    matrix = np.zeros((n, n), dtype=np.float64)
    if edges.empty or n == 0:
        return matrix, idx
    for src, dst, weight in edges[["start_station_id", "end_station_id", "weight"]].itertuples(index=False):
        if src in idx and dst in idx and src != dst:
            matrix[idx[src], idx[dst]] += float(weight or 0.0)
    return matrix + matrix.T, idx


def _label_propagation(W_sym: np.ndarray, *, max_iters: int = 50, seed: int = 42) -> np.ndarray:
    """Deterministic label propagation on a symmetric weight matrix.

    Each node starts in its own community; each iteration relabels every node
    to the community with the largest sum of edge weights from its neighbors.
    Ties are broken by smaller current community id (deterministic).
    """

    n = W_sym.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    labels = np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    for _ in range(int(max_iters)):
        order = rng.permutation(n)
        changed = 0
        for node in order:
            row = W_sym[node]
            if not np.any(row > 0):
                continue
            neighbors = np.nonzero(row)[0]
            if len(neighbors) == 0:
                continue
            scores: dict[int, float] = {}
            for nb in neighbors:
                lbl = int(labels[nb])
                scores[lbl] = scores.get(lbl, 0.0) + float(row[nb])
            best_label, best_score = labels[node], -1.0
            for lbl, score in scores.items():
                if score > best_score or (score == best_score and lbl < best_label):
                    best_label, best_score = lbl, score
            if best_label != labels[node]:
                labels[node] = best_label
                changed += 1
        if changed == 0:
            break
    return _compact_labels(labels)


def _compact_labels(labels: np.ndarray) -> np.ndarray:
    """Renumber labels to 0..K-1 in order of first appearance."""

    seen: dict[int, int] = {}
    out = np.zeros_like(labels)
    for i, lbl in enumerate(labels):
        key = int(lbl)
        if key not in seen:
            seen[key] = len(seen)
        out[i] = seen[key]
    return out


def _spectral_kmeans(W_sym: np.ndarray, *, target_k: int, seed: int) -> np.ndarray:
    """Spectral embedding via top eigenvectors of the normalized Laplacian, then KMeans.

    Returns single-community labels if eigendecomposition or KMeans fails.
    """

    n = W_sym.shape[0]
    if n <= 1:
        return np.zeros(max(0, n), dtype=np.int64)
    try:
        from sklearn.cluster import KMeans
    except Exception:
        return np.zeros(n, dtype=np.int64)
    degree = W_sym.sum(axis=1)
    if not np.any(degree > 0):
        return np.zeros(n, dtype=np.int64)
    d_inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(degree), 0.0)
    L_norm = np.eye(n) - (W_sym * d_inv_sqrt[None, :]) * d_inv_sqrt[:, None]
    try:
        eigvals, eigvecs = np.linalg.eigh(L_norm)
    except np.linalg.LinAlgError:
        return np.zeros(n, dtype=np.int64)
    k = max(2, min(int(target_k), n))
    embedding = eigvecs[:, :k]
    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    norms = np.where(norms > 1e-9, norms, 1.0)
    embedding = embedding / norms
    try:
        km = KMeans(n_clusters=k, random_state=int(seed), n_init=4)
        labels = km.fit_predict(embedding)
    except Exception:
        return np.zeros(n, dtype=np.int64)
    return _compact_labels(np.asarray(labels, dtype=np.int64))


def _spatial_kmeans(coords: pd.DataFrame, *, target_k: int, seed: int) -> dict[str, int]:
    if coords.empty:
        return {}
    try:
        from sklearn.cluster import KMeans
    except Exception:
        return {sid: 0 for sid in coords["station_id"].astype(str)}
    n = len(coords)
    k = max(1, min(int(target_k), n))
    if k <= 1:
        return {sid: 0 for sid in coords["station_id"].astype(str)}
    arr = coords[["lat", "lon"]].to_numpy(dtype=np.float64)
    try:
        km = KMeans(n_clusters=k, random_state=int(seed), n_init=4)
        labels = km.fit_predict(arr)
    except Exception:
        return {sid: 0 for sid in coords["station_id"].astype(str)}
    return {
        str(sid): int(lbl)
        for sid, lbl in zip(coords["station_id"].astype(str), _compact_labels(np.asarray(labels, dtype=np.int64)))
    }


def _suggest_target_k(n_stations: int) -> int:
    if n_stations <= 0:
        return 1
    # Aim for ~30–200 stations per community; smaller graphs use fewer clusters.
    if n_stations <= 20:
        target = 2 if n_stations <= 8 else max(2, n_stations // 4)
    else:
        target = max(10, min(40, int(round(math.sqrt(n_stations) / 2.0) + 6)))
    return max(2, min(target, n_stations))


def _compute_roles_and_scores(
    W_directed: np.ndarray,
    labels: np.ndarray,
    station_ids: list[str],
) -> tuple[dict[str, str], dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
    """Compute boundary/gateway scores + role assignment from directed OD weights.

    ``W_directed[i, j]`` = trips from station i to station j (not symmetrized).
    """

    eps = 1e-9
    n = len(station_ids)
    boundary_score: dict[str, float] = {}
    gateway_score: dict[str, float] = {}
    inbound_internal: dict[str, float] = {}
    outbound_internal: dict[str, float] = {}
    station_to_role: dict[str, str] = {}
    if n == 0:
        return station_to_role, boundary_score, gateway_score, inbound_internal, outbound_internal

    out_total = W_directed.sum(axis=1)
    in_total = W_directed.sum(axis=0)
    out_internal = np.zeros(n, dtype=np.float64)
    in_internal = np.zeros(n, dtype=np.float64)
    for i in range(n):
        same = labels == labels[i]
        out_internal[i] = float(W_directed[i, same].sum())
        in_internal[i] = float(W_directed[same, i].sum())

    out_share = np.where(out_total > 0, out_internal / np.maximum(out_total, eps), np.nan)
    in_share = np.where(in_total > 0, in_internal / np.maximum(in_total, eps), np.nan)
    # Boundary score uses only the shares that are defined; stations with no
    # outbound (or no inbound) flow fall back to the single available share so we
    # don't artificially inflate boundary_score by treating a missing share as 0.
    avg_share = np.where(
        np.isnan(out_share) & np.isnan(in_share),
        np.nan,
        np.where(
            np.isnan(out_share),
            in_share,
            np.where(np.isnan(in_share), out_share, 0.5 * (out_share + in_share)),
        ),
    )
    bs = np.where(np.isnan(avg_share), 0.0, 1.0 - avg_share)
    # External flow magnitude (z-scored across stations with any flow).
    external = (out_total - out_internal) + (in_total - in_internal)
    if np.any(external > 0):
        mean = float(external[external > 0].mean())
        std = float(external[external > 0].std() or 1.0)
        gateway_z = np.where(external > 0, (external - mean) / max(std, 1.0), 0.0)
    else:
        gateway_z = np.zeros(n, dtype=np.float64)

    for i, sid in enumerate(station_ids):
        score = float(bs[i])
        boundary_score[sid] = score
        gateway_score[sid] = float(gateway_z[i])
        outbound_internal[sid] = float(out_share[i]) if out_total[i] > 0 else 0.0
        inbound_internal[sid] = float(in_share[i]) if in_total[i] > 0 else 0.0
        if out_total[i] == 0 and in_total[i] == 0:
            station_to_role[sid] = ROLE_UNKNOWN
        elif score >= 0.35 or gateway_z[i] >= 1.5:
            station_to_role[sid] = ROLE_GATEWAY
        elif score >= 0.15:
            station_to_role[sid] = ROLE_BOUNDARY
        else:
            station_to_role[sid] = ROLE_CORE
    return station_to_role, boundary_score, gateway_score, inbound_internal, outbound_internal


def _compute_community_neighbors(
    W_directed: np.ndarray,
    labels: np.ndarray,
    *,
    top_k: int = 5,
) -> dict[int, list[tuple[int, float]]]:
    """For each community c, return a list of (other_community, normalized weight) of top-k strongest external links."""

    n = labels.shape[0]
    if n == 0:
        return {}
    n_comm = int(labels.max()) + 1 if n > 0 else 0
    comm_matrix = np.zeros((n_comm, n_comm), dtype=np.float64)
    for i in range(n):
        ci = int(labels[i])
        for j in range(n):
            if i == j or W_directed[i, j] == 0:
                continue
            cj = int(labels[j])
            comm_matrix[ci, cj] += float(W_directed[i, j])
    out: dict[int, list[tuple[int, float]]] = {}
    for c in range(n_comm):
        row = comm_matrix[c].copy()
        row[c] = 0.0
        total = float(row.sum())
        if total <= 0:
            out[c] = []
            continue
        order = np.argsort(-row)
        neighbors: list[tuple[int, float]] = []
        for nb in order[:top_k]:
            weight = float(row[nb]) / total
            if weight <= 0:
                break
            neighbors.append((int(nb), weight))
        out[c] = neighbors
    return out


def build_partition(
    conn: duckdb.DuckDBPyConnection | None,
    *,
    training_window_end: datetime,
    lookback_days: int = 30,
    seed: int = 42,
    target_k: int | None = None,
    algorithm: str = "auto",
) -> Partition:
    """Build a station → community partition <= training_window_end.

    Algorithm ladder ('auto' tries each in turn):
      1. label_propagation on symmetric OD weights
      2. spectral_kmeans on the same weights
      3. spatial_kmeans on lat/lon
      4. single_community fallback
    """

    computed_at = _utc_now()
    training_window_end = pd.Timestamp(training_window_end).to_pydatetime().replace(tzinfo=None)
    edges = _load_od_edges(conn, training_window_end=training_window_end, lookback_days=lookback_days)
    coords = _load_station_coords(conn)
    station_ids = sorted({str(s) for s in pd.concat([
        edges["start_station_id"], edges["end_station_id"], coords["station_id"]
    ], ignore_index=True).dropna().astype(str).unique().tolist()})
    if not station_ids:
        return Partition(
            partition_id=_stable_partition_id("single_community", training_window_end, []),
            computed_at=computed_at,
            source_data_start=None,
            source_data_end=training_window_end,
            algorithm="single_community",
            n_communities=0,
        )

    W_directed, idx = _build_symmetric_matrix(
        pd.DataFrame(columns=["start_station_id", "end_station_id", "weight"]), station_ids
    )
    # Build directed matrix separately to keep symmetric vs directed separate.
    W_directed = np.zeros((len(station_ids), len(station_ids)), dtype=np.float64)
    if not edges.empty:
        for src, dst, weight in edges[["start_station_id", "end_station_id", "weight"]].itertuples(index=False):
            if src in idx and dst in idx and src != dst:
                W_directed[idx[src], idx[dst]] += float(weight or 0.0)
    W_sym = W_directed + W_directed.T

    target_k_eff = target_k or _suggest_target_k(len(station_ids))
    algo_choice = (algorithm or "auto").lower()

    labels: np.ndarray | None = None
    chosen_algo = "single_community"
    has_flows = bool(np.any(W_sym > 0))

    if algo_choice in ("auto", "label_propagation", "louvain") and has_flows:
        try:
            labels = _label_propagation(W_sym, seed=int(seed))
            if labels.size and int(labels.max()) + 1 >= 2:
                chosen_algo = "label_propagation"
        except Exception:
            labels = None
    if (labels is None or (labels.size and int(labels.max()) + 1 < 2)) and algo_choice in (
        "auto",
        "spectral",
        "spectral_kmeans",
    ) and has_flows:
        try:
            labels = _spectral_kmeans(W_sym, target_k=target_k_eff, seed=int(seed))
            if labels.size and int(labels.max()) + 1 >= 2:
                chosen_algo = "spectral_kmeans"
        except Exception:
            labels = None
    if (labels is None or (labels.size and int(labels.max()) + 1 < 2)) and algo_choice in (
        "auto",
        "spatial",
        "spatial_kmeans",
    ):
        spatial = _spatial_kmeans(coords, target_k=target_k_eff, seed=int(seed))
        if spatial:
            labels = np.array([spatial.get(sid, 0) for sid in station_ids], dtype=np.int64)
            if labels.size and int(labels.max()) + 1 >= 2:
                chosen_algo = "spatial_kmeans"
    if labels is None or labels.size == 0:
        labels = np.zeros(len(station_ids), dtype=np.int64)
        chosen_algo = "single_community"

    labels = _compact_labels(labels)
    station_to_community = {sid: int(lbl) for sid, lbl in zip(station_ids, labels)}
    role_map, boundary, gateway, in_share, out_share = _compute_roles_and_scores(
        W_directed, labels, station_ids
    )
    if not has_flows:
        # When we have no flow data the boundary/gateway scores are meaningless;
        # mark every station as unknown so downstream features take the neutral defaults.
        role_map = {sid: ROLE_UNKNOWN for sid in station_ids}
        boundary = {sid: 0.0 for sid in station_ids}
        gateway = {sid: 0.0 for sid in station_ids}
        in_share = {sid: 0.0 for sid in station_ids}
        out_share = {sid: 0.0 for sid in station_ids}

    neighbors = _compute_community_neighbors(W_directed, labels)
    return Partition(
        partition_id=_stable_partition_id(chosen_algo, training_window_end, station_ids),
        computed_at=computed_at,
        source_data_start=(training_window_end - pd.Timedelta(days=int(lookback_days))).replace(tzinfo=None)
        if has_flows
        else None,
        source_data_end=training_window_end,
        algorithm=chosen_algo,
        n_communities=int(labels.max()) + 1 if labels.size else 0,
        station_to_community=station_to_community,
        station_to_role=role_map,
        boundary_score=boundary,
        gateway_score=gateway,
        inbound_internal_share=in_share,
        outbound_internal_share=out_share,
        community_to_neighbors=neighbors,
    )


def make_random_partition(partition: Partition, *, seed: int = 7) -> Partition:
    """Return a size-preserving random shuffle of community labels (ablation control)."""

    rng = np.random.default_rng(int(seed))
    stations = list(partition.station_to_community.keys())
    community_ids = [partition.station_to_community[s] for s in stations]
    permuted = rng.permutation(community_ids)
    new_map = {s: int(c) for s, c in zip(stations, permuted)}
    return Partition(
        partition_id=_stable_partition_id("random", partition.source_data_end, stations),
        computed_at=_utc_now(),
        source_data_start=partition.source_data_start,
        source_data_end=partition.source_data_end,
        algorithm="random",
        n_communities=partition.n_communities,
        station_to_community=new_map,
        station_to_role=dict(partition.station_to_role),
        boundary_score=dict(partition.boundary_score),
        gateway_score=dict(partition.gateway_score),
        inbound_internal_share=dict(partition.inbound_internal_share),
        outbound_internal_share=dict(partition.outbound_internal_share),
        community_to_neighbors=dict(partition.community_to_neighbors),
    )


def upsert_station_communities(conn: duckdb.DuckDBPyConnection, partition: Partition) -> int:
    """Persist a Partition to the station_communities table. Returns row count."""

    if not partition.station_to_community:
        return 0
    rows = []
    for station_id, community_id in partition.station_to_community.items():
        rows.append(
            (
                partition.partition_id,
                partition.computed_at,
                partition.source_data_start,
                partition.source_data_end,
                partition.algorithm,
                int(partition.n_communities),
                str(station_id),
                int(community_id),
                str(partition.station_to_role.get(station_id, ROLE_UNKNOWN)),
                float(partition.inbound_internal_share.get(station_id, 0.0)),
                float(partition.outbound_internal_share.get(station_id, 0.0)),
                float(partition.boundary_score.get(station_id, 0.0)),
                float(partition.gateway_score.get(station_id, 0.0)),
            )
        )
    conn.execute("DELETE FROM station_communities WHERE partition_id = ?", [partition.partition_id])
    conn.executemany(
        """
        INSERT INTO station_communities (
          partition_id, computed_at, source_data_start, source_data_end, algorithm,
          n_communities, station_id, community_id, role,
          inbound_internal_share, outbound_internal_share, boundary_score, gateway_score
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    return len(rows)
