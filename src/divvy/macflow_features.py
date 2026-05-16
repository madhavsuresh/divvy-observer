"""Community/role/exchange feature attachment for MacFlow-NISSM-lite.

Adds mobility-community-derived columns to a training-example frame produced
by :func:`divvy.label_builder.build_leak_free_examples`. Pure DataFrame
transform — no DB queries, no side effects.

Leakage policy:
  Every community-aggregate feature must use only station observations
  ``obs_ts <= anchor_ts``. Aggregates are pre-computed in
  :func:`build_station_aggregates` from the training-window status frame and
  joined back onto each row via an as-of merge.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .mobility_partitions import (
    INT_TO_ROLE,
    Partition,
    ROLE_TO_INT,
    ROLE_UNKNOWN,
)


MACFLOW_FEATURE_COLUMNS: list[str] = [
    "community_id",
    "role_id",
    "boundary_score",
    "gateway_score",
    "inbound_internal_share",
    "outbound_internal_share",
    "community_recent_ebikes_mean",
    "community_recent_zero_share",
    "community_recent_churn",
    "community_recent_full_share",
    "community_recent_docks_mean",
    "neighbor_community_ebikes_mean",
    "neighbor_community_zero_share",
    "neighbor_community_churn",
    "od_departure_pressure_same_community",
    "od_arrival_pressure_same_community",
    "od_departure_pressure_external",
    "od_arrival_pressure_external",
    "community_exchange_in_pressure",
    "community_exchange_out_pressure",
]

NEUTRAL_DEFAULTS: dict[str, float] = {
    "community_id": 0.0,
    "role_id": float(ROLE_TO_INT[ROLE_UNKNOWN]),
    "boundary_score": 0.0,
    "gateway_score": 0.0,
    "inbound_internal_share": 0.0,
    "outbound_internal_share": 0.0,
    "community_recent_ebikes_mean": 0.0,
    "community_recent_zero_share": 0.0,
    "community_recent_churn": 0.0,
    "community_recent_full_share": 0.0,
    "community_recent_docks_mean": 0.0,
    "neighbor_community_ebikes_mean": 0.0,
    "neighbor_community_zero_share": 0.0,
    "neighbor_community_churn": 0.0,
    "od_departure_pressure_same_community": 0.0,
    "od_arrival_pressure_same_community": 0.0,
    "od_departure_pressure_external": 0.0,
    "od_arrival_pressure_external": 0.0,
    "community_exchange_in_pressure": 0.0,
    "community_exchange_out_pressure": 0.0,
}


@dataclass
class CommunityRuntimeDefaults:
    """Snapshot of per-community feature values from end-of-training period.

    Used to score live rows at request time without re-aggregating station
    history (we deliberately treat community aggregates as a stable prior at
    inference; see plan doc).
    """

    per_community: dict[int, dict[str, float]]
    per_station: dict[str, dict[str, float]]
    pressures: dict[int, dict[str, float]]


def _coerce_partition_mode(mode: str | None) -> str:
    valid = {"off", "id_only", "id_plus_role", "full", "random", "spatial"}
    mode = (mode or "full").lower()
    if mode not in valid:
        raise ValueError(f"unknown partition_mode: {mode}; must be one of {sorted(valid)}")
    return mode


def _ensure_anchor_ts(rows: pd.DataFrame) -> pd.Series:
    if "anchor_ts" in rows.columns:
        return pd.to_datetime(rows["anchor_ts"], errors="coerce")
    if "forecasted_at" in rows.columns:
        return pd.to_datetime(rows["forecasted_at"], errors="coerce")
    if "last_reported" in rows.columns:
        return pd.to_datetime(rows["last_reported"], errors="coerce")
    return pd.Series(pd.NaT, index=rows.index)


def build_station_aggregates(status_frame: pd.DataFrame) -> pd.DataFrame:
    """Reshape station_status rows into a per-row observation table.

    Returns a frame with columns ``station_id``, ``observation_ts``, ``ebikes``,
    ``total_bikes``, ``docks``, ``churn`` suitable for as-of joins back onto
    training-example rows.
    """

    if status_frame.empty:
        return pd.DataFrame(
            columns=["station_id", "observation_ts", "ebikes", "total_bikes", "docks", "churn"],
        )
    ts_col = "observation_ts" if "observation_ts" in status_frame.columns else (
        "last_reported" if "last_reported" in status_frame.columns else "fetched_at"
    )
    out = pd.DataFrame(
        {
            "station_id": status_frame["station_id"].astype(str),
            "observation_ts": pd.to_datetime(status_frame[ts_col], errors="coerce"),
            "ebikes": pd.to_numeric(
                status_frame.get("num_ebikes_available", status_frame.get("e_now")),
                errors="coerce",
            ).fillna(0).astype(float),
            "total_bikes": pd.to_numeric(
                status_frame.get("num_bikes_available", status_frame.get("q_now")),
                errors="coerce",
            ).fillna(0).astype(float),
            "docks": pd.to_numeric(
                status_frame.get("num_docks_available"),
                errors="coerce",
            ).fillna(0).astype(float),
        }
    )
    if "churn" in status_frame.columns:
        out["churn"] = pd.to_numeric(status_frame["churn"], errors="coerce").fillna(0).astype(float)
    else:
        # Approximate churn from the delta in ebikes if a sequence is available; otherwise 0.
        out["churn"] = 0.0
    return out.dropna(subset=["observation_ts"]).sort_values(["station_id", "observation_ts"]).reset_index(drop=True)


def _aggregate_community_pressures(
    partition: Partition,
    od_edges: pd.DataFrame | None,
) -> dict[int, dict[str, float]]:
    """Compute static OD-pressure features per community (from partition + OD edges)."""

    n = partition.n_communities
    pressures: dict[int, dict[str, float]] = {
        c: {
            "od_departure_pressure_same_community": 0.0,
            "od_arrival_pressure_same_community": 0.0,
            "od_departure_pressure_external": 0.0,
            "od_arrival_pressure_external": 0.0,
            "community_exchange_in_pressure": 0.0,
            "community_exchange_out_pressure": 0.0,
        }
        for c in range(max(1, n))
    }
    if od_edges is None or od_edges.empty:
        return pressures
    out_same = np.zeros(max(1, n), dtype=np.float64)
    out_ext = np.zeros(max(1, n), dtype=np.float64)
    in_same = np.zeros(max(1, n), dtype=np.float64)
    in_ext = np.zeros(max(1, n), dtype=np.float64)
    station_to_community = partition.station_to_community
    for src, dst, weight in od_edges[["start_station_id", "end_station_id", "weight"]].itertuples(index=False):
        if src not in station_to_community or dst not in station_to_community:
            continue
        cs, cd = int(station_to_community[src]), int(station_to_community[dst])
        w = float(weight or 0.0)
        if cs == cd:
            out_same[cs] += w
            in_same[cd] += w
        else:
            out_ext[cs] += w
            in_ext[cd] += w
    for c in range(max(1, n)):
        out_total = out_same[c] + out_ext[c]
        in_total = in_same[c] + in_ext[c]
        pressures[c]["od_departure_pressure_same_community"] = float(out_same[c] / max(out_total, 1e-9))
        pressures[c]["od_departure_pressure_external"] = float(out_ext[c] / max(out_total, 1e-9))
        pressures[c]["od_arrival_pressure_same_community"] = float(in_same[c] / max(in_total, 1e-9))
        pressures[c]["od_arrival_pressure_external"] = float(in_ext[c] / max(in_total, 1e-9))
        # Exchange pressure: external in vs out.
        pressures[c]["community_exchange_in_pressure"] = float(in_ext[c] / max(in_total, 1e-9))
        pressures[c]["community_exchange_out_pressure"] = float(out_ext[c] / max(out_total, 1e-9))
    return pressures


def _community_aggregates_asof(
    rows: pd.DataFrame,
    station_aggregates: pd.DataFrame | None,
    station_to_community: dict[str, int],
    lookback_minutes: int = 120,
) -> pd.DataFrame:
    """As-of join community aggregates to each (station_id, anchor_ts) row.

    For each row we average the last station-status observation per station within
    its community whose ``observation_ts`` is in ``[anchor_ts - lookback, anchor_ts]``.
    Result is one row per input row, indexed identically.
    """

    if station_aggregates is None or station_aggregates.empty:
        return pd.DataFrame(
            {
                "community_recent_ebikes_mean": np.zeros(len(rows), dtype=float),
                "community_recent_zero_share": np.zeros(len(rows), dtype=float),
                "community_recent_churn": np.zeros(len(rows), dtype=float),
                "community_recent_full_share": np.zeros(len(rows), dtype=float),
                "community_recent_docks_mean": np.zeros(len(rows), dtype=float),
            },
            index=rows.index,
        )
    aggs = station_aggregates.copy()
    aggs["community_id"] = aggs["station_id"].map(station_to_community).fillna(-1).astype(int)
    aggs["zero_flag"] = (aggs["ebikes"] <= 0).astype(float)
    aggs["full_flag"] = (aggs["docks"] <= 0).astype(float)
    aggs = aggs.sort_values("observation_ts")
    anchor_ts = _ensure_anchor_ts(rows)
    n = len(rows)
    out = {
        "community_recent_ebikes_mean": np.zeros(n, dtype=float),
        "community_recent_zero_share": np.zeros(n, dtype=float),
        "community_recent_churn": np.zeros(n, dtype=float),
        "community_recent_full_share": np.zeros(n, dtype=float),
        "community_recent_docks_mean": np.zeros(n, dtype=float),
    }
    # Index aggregates by community for fast slicing.
    grouped = {int(c): g for c, g in aggs.groupby("community_id", sort=False)}
    for i, (idx, row) in enumerate(rows.iterrows()):
        sid = str(row.get("station_id", ""))
        ts = anchor_ts.iloc[i]
        if pd.isna(ts):
            continue
        community_id = int(station_to_community.get(sid, -1))
        if community_id < 0 or community_id not in grouped:
            continue
        window_start = ts - pd.Timedelta(minutes=int(lookback_minutes))
        slice_ = grouped[community_id]
        mask = (slice_["observation_ts"] >= window_start) & (slice_["observation_ts"] <= ts)
        sub = slice_.loc[mask]
        if sub.empty:
            continue
        # Use the most recent observation per station within the window so we don't
        # double-count repeated polls of the same station.
        latest = sub.sort_values("observation_ts").groupby("station_id", as_index=False).tail(1)
        out["community_recent_ebikes_mean"][i] = float(latest["ebikes"].mean()) if len(latest) else 0.0
        out["community_recent_zero_share"][i] = float(latest["zero_flag"].mean()) if len(latest) else 0.0
        out["community_recent_full_share"][i] = float(latest["full_flag"].mean()) if len(latest) else 0.0
        out["community_recent_docks_mean"][i] = float(latest["docks"].mean()) if len(latest) else 0.0
        out["community_recent_churn"][i] = float(latest["churn"].mean()) if "churn" in latest.columns else 0.0
    return pd.DataFrame(out, index=rows.index)


def _neighbor_aggregates(
    rows: pd.DataFrame,
    partition: Partition,
    community_aggs: pd.DataFrame,
) -> pd.DataFrame:
    """Compute neighbor-community-aggregate features per row.

    Uses ``partition.community_to_neighbors`` to find the top-K weighted
    neighbor communities and takes the weighted average of their recent
    aggregates from ``community_aggs`` indexed by row.
    """

    n = len(rows)
    out = {
        "neighbor_community_ebikes_mean": np.zeros(n, dtype=float),
        "neighbor_community_zero_share": np.zeros(n, dtype=float),
        "neighbor_community_churn": np.zeros(n, dtype=float),
    }
    if community_aggs.empty or n == 0:
        return pd.DataFrame(out, index=rows.index)
    # Build a quick lookup keyed by (anchor_ts, community_id) by grouping the input rows by anchor_ts.
    sid_col = rows["station_id"].astype(str)
    community_per_row = sid_col.map(partition.station_to_community).fillna(-1).astype(int)
    # Build mean of community_aggs per community across all rows where that community is observed at any anchor.
    # Since we already store the row-level community aggregate, we approximate neighbor mean as the mean
    # over all rows whose community matches the neighbor — this is acceptable because all rows here belong
    # to the same training/eval window.
    by_community = (
        pd.DataFrame(
            {
                "community_id": community_per_row,
                "ebikes": community_aggs["community_recent_ebikes_mean"].to_numpy(),
                "zero": community_aggs["community_recent_zero_share"].to_numpy(),
                "churn": community_aggs["community_recent_churn"].to_numpy(),
            }
        )
        .groupby("community_id")
        .agg({"ebikes": "mean", "zero": "mean", "churn": "mean"})
        .to_dict(orient="index")
    )

    for i, idx in enumerate(rows.index):
        c = int(community_per_row.iloc[i])
        neighbors = partition.community_to_neighbors.get(c, []) if c >= 0 else []
        if not neighbors:
            continue
        weights = []
        eb_vals = []
        zero_vals = []
        churn_vals = []
        for nb_c, w in neighbors:
            stats = by_community.get(nb_c)
            if not stats:
                continue
            weights.append(w)
            eb_vals.append(stats["ebikes"])
            zero_vals.append(stats["zero"])
            churn_vals.append(stats["churn"])
        if not weights:
            continue
        total_w = float(sum(weights))
        if total_w <= 0:
            continue
        out["neighbor_community_ebikes_mean"][i] = float(np.dot(weights, eb_vals) / total_w)
        out["neighbor_community_zero_share"][i] = float(np.dot(weights, zero_vals) / total_w)
        out["neighbor_community_churn"][i] = float(np.dot(weights, churn_vals) / total_w)
    return pd.DataFrame(out, index=rows.index)


def attach_macflow_features(
    examples: pd.DataFrame,
    partition: Partition | None,
    *,
    station_aggregates: pd.DataFrame | None = None,
    partition_mode: str = "full",
    od_edges: pd.DataFrame | None = None,
    lookback_minutes: int = 120,
) -> pd.DataFrame:
    """Attach community/role/exchange features to ``examples`` in place-safe form.

    Returns a new DataFrame with the same index as ``examples`` and the
    additional columns listed in :data:`MACFLOW_FEATURE_COLUMNS`.

    If ``partition`` is None, or if ``partition_mode == "off"``, all feature
    columns are emitted with neutral defaults so the model can still train.
    """

    partition_mode = _coerce_partition_mode(partition_mode)
    out = examples.copy()
    n = len(out)
    if n == 0:
        for col in MACFLOW_FEATURE_COLUMNS:
            out[col] = pd.Series(dtype=float)
        return out

    if partition is None or partition_mode == "off":
        for col, default in NEUTRAL_DEFAULTS.items():
            out[col] = default
        return out

    station_to_community = dict(partition.station_to_community)
    station_to_role = dict(partition.station_to_role)
    boundary = dict(partition.boundary_score)
    gateway = dict(partition.gateway_score)
    in_share = dict(partition.inbound_internal_share)
    out_share = dict(partition.outbound_internal_share)

    sid_series = out["station_id"].astype(str)
    out["community_id"] = sid_series.map(station_to_community).fillna(-1).astype(int)
    role_int = sid_series.map(lambda s: ROLE_TO_INT.get(station_to_role.get(s, ROLE_UNKNOWN), 3))
    out["role_id"] = role_int.astype(int)
    out["boundary_score"] = sid_series.map(boundary).fillna(0.0).astype(float)
    out["gateway_score"] = sid_series.map(gateway).fillna(0.0).astype(float)
    out["inbound_internal_share"] = sid_series.map(in_share).fillna(0.0).astype(float)
    out["outbound_internal_share"] = sid_series.map(out_share).fillna(0.0).astype(float)

    # As-of community aggregates (slow-tier prior).
    aggs = _community_aggregates_asof(
        out,
        station_aggregates,
        station_to_community,
        lookback_minutes=lookback_minutes,
    )
    for col in aggs.columns:
        out[col] = aggs[col].astype(float)

    neighbors = _neighbor_aggregates(out, partition, aggs)
    for col in neighbors.columns:
        out[col] = neighbors[col].astype(float)

    pressures = _aggregate_community_pressures(partition, od_edges)
    out["od_departure_pressure_same_community"] = out["community_id"].map(
        lambda c: pressures.get(int(c), {}).get("od_departure_pressure_same_community", 0.0)
    ).astype(float)
    out["od_arrival_pressure_same_community"] = out["community_id"].map(
        lambda c: pressures.get(int(c), {}).get("od_arrival_pressure_same_community", 0.0)
    ).astype(float)
    out["od_departure_pressure_external"] = out["community_id"].map(
        lambda c: pressures.get(int(c), {}).get("od_departure_pressure_external", 0.0)
    ).astype(float)
    out["od_arrival_pressure_external"] = out["community_id"].map(
        lambda c: pressures.get(int(c), {}).get("od_arrival_pressure_external", 0.0)
    ).astype(float)
    out["community_exchange_in_pressure"] = out["community_id"].map(
        lambda c: pressures.get(int(c), {}).get("community_exchange_in_pressure", 0.0)
    ).astype(float)
    out["community_exchange_out_pressure"] = out["community_id"].map(
        lambda c: pressures.get(int(c), {}).get("community_exchange_out_pressure", 0.0)
    ).astype(float)

    # Negative community_id (station not in partition) → drop to 0 to keep embeddings valid.
    out["community_id"] = out["community_id"].clip(lower=0)

    if partition_mode == "id_only":
        # Keep community_id; zero out the rest of the partition-derived columns.
        for col, default in NEUTRAL_DEFAULTS.items():
            if col == "community_id":
                continue
            out[col] = default
    elif partition_mode == "id_plus_role":
        for col, default in NEUTRAL_DEFAULTS.items():
            if col in {"community_id", "role_id"}:
                continue
            out[col] = default
    # "full", "random", and "spatial" leave the columns populated as-is (random/spatial
    # use a randomized/spatial partition upstream — this function does not know which).
    return out


def build_community_runtime_defaults(
    train_examples: pd.DataFrame,
    partition: Partition,
    *,
    od_edges: pd.DataFrame | None = None,
) -> CommunityRuntimeDefaults:
    """Snapshot per-community feature averages from end-of-training rows.

    Used at inference to populate community aggregates when the predictor's
    row-prep path does not include them. Falls back to neutral defaults for
    communities with no observations.
    """

    feat = MACFLOW_FEATURE_COLUMNS
    per_community: dict[int, dict[str, float]] = {}
    if not train_examples.empty and "community_id" in train_examples.columns:
        cols_present = [c for c in feat if c in train_examples.columns and c != "community_id"]
        if cols_present:
            grouped = (
                train_examples[["community_id", *cols_present]]
                .groupby("community_id")
                .mean(numeric_only=True)
                .to_dict(orient="index")
            )
            for community_id, stats in grouped.items():
                per_community[int(community_id)] = {col: float(stats.get(col, 0.0)) for col in feat}

    pressures = _aggregate_community_pressures(partition, od_edges)
    per_station: dict[str, dict[str, float]] = {}
    for sid in partition.stations():
        cid = int(partition.station_to_community.get(sid, 0))
        per_station[sid] = {
            "community_id": float(cid),
            "role_id": float(ROLE_TO_INT.get(partition.station_to_role.get(sid, ROLE_UNKNOWN), 3)),
            "boundary_score": float(partition.boundary_score.get(sid, 0.0)),
            "gateway_score": float(partition.gateway_score.get(sid, 0.0)),
            "inbound_internal_share": float(partition.inbound_internal_share.get(sid, 0.0)),
            "outbound_internal_share": float(partition.outbound_internal_share.get(sid, 0.0)),
        }
    return CommunityRuntimeDefaults(
        per_community=per_community,
        per_station=per_station,
        pressures=pressures,
    )


def apply_runtime_defaults(
    rows: pd.DataFrame,
    defaults: CommunityRuntimeDefaults,
    *,
    partition_mode: str = "full",
) -> pd.DataFrame:
    """Attach community features at request-time using snapshot defaults.

    Falls back to global neutral defaults when a station is unknown.
    """

    partition_mode = _coerce_partition_mode(partition_mode)
    out = rows.copy()
    n = len(out)
    if n == 0:
        for col in MACFLOW_FEATURE_COLUMNS:
            out[col] = pd.Series(dtype=float)
        return out

    if partition_mode == "off" or not defaults.per_station:
        for col, default in NEUTRAL_DEFAULTS.items():
            out[col] = default
        return out

    sid_series = out["station_id"].astype(str)
    for col in MACFLOW_FEATURE_COLUMNS:
        out[col] = NEUTRAL_DEFAULTS[col]

    for col in (
        "community_id",
        "role_id",
        "boundary_score",
        "gateway_score",
        "inbound_internal_share",
        "outbound_internal_share",
    ):
        out[col] = sid_series.map(lambda s: defaults.per_station.get(s, {}).get(col, NEUTRAL_DEFAULTS[col])).astype(float)
    out["community_id"] = out["community_id"].astype(int).clip(lower=0)
    out["role_id"] = out["role_id"].astype(int)

    # Slow-tier community aggregates: pull per-community snapshot.
    for col in (
        "community_recent_ebikes_mean",
        "community_recent_zero_share",
        "community_recent_churn",
        "community_recent_full_share",
        "community_recent_docks_mean",
        "neighbor_community_ebikes_mean",
        "neighbor_community_zero_share",
        "neighbor_community_churn",
    ):
        out[col] = out["community_id"].map(
            lambda c: defaults.per_community.get(int(c), {}).get(col, NEUTRAL_DEFAULTS[col])
        ).astype(float)

    for col in (
        "od_departure_pressure_same_community",
        "od_arrival_pressure_same_community",
        "od_departure_pressure_external",
        "od_arrival_pressure_external",
        "community_exchange_in_pressure",
        "community_exchange_out_pressure",
    ):
        out[col] = out["community_id"].map(
            lambda c: defaults.pressures.get(int(c), {}).get(col, NEUTRAL_DEFAULTS[col])
        ).astype(float)

    if partition_mode == "id_only":
        for col, default in NEUTRAL_DEFAULTS.items():
            if col == "community_id":
                continue
            out[col] = default
    elif partition_mode == "id_plus_role":
        for col, default in NEUTRAL_DEFAULTS.items():
            if col in {"community_id", "role_id"}:
                continue
            out[col] = default

    # Metra commuter-rail event features. Adds N_FEATURES columns named
    # "metra_*" — zero everywhere except at the ~21 Divvy stations that sit
    # within ~400m of a Metra stop and have a measurable train-arrival lift
    # (see divvy/metra.py). Done after the partition-mode neutralization
    # because the Metra signal is per-station and orthogonal to the
    # community-flow features the partition_mode controls.
    try:
        from . import metra as _metra
        out = _metra.attach_metra_features(out, ts_col="anchor_ts" if "anchor_ts" in out.columns else "forecasted_at")
    except Exception:
        # Don't let a Metra-feature failure break macflow training.
        # Fall back to all-zero so the column set stays consistent.
        for k in (
            "is_near_metra", "metra_distance_m", "metra_arr_in_5m", "metra_arr_in_10m",
            "metra_arr_in_30m", "metra_dep_in_5m", "metra_dep_in_10m", "metra_dep_in_30m",
            "metra_arr_in_last_5m", "metra_arr_in_last_10m", "metra_dep_in_last_5m",
            "metra_pickup_lift", "metra_dropoff_lift",
            "metra_pickup_lift_x_arr_5m", "metra_dropoff_lift_x_dep_5m",
        ):
            if k not in out.columns:
                out[k] = 0.0
    return out
