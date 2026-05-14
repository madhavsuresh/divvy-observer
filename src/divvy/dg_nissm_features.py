from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


SHIFTED_PRIOR_COLUMNS = [
    "station_hour_dow_has_ebike_rate_shifted",
    "station_hour_dow_e_mean_shifted",
    "station_hour_dow_q_mean_shifted",
    "station_hour_dow_depart_rate_shifted",
    "station_hour_dow_arrive_rate_shifted",
    "hour_dow_global_rate_shifted",
    "global_rate_shifted",
    "neighbor_hour_dow_rate_shifted",
]

FLOW_LABEL_COLUMNS = [
    "obs_e_depart",
    "obs_e_arrive",
    "obs_c_depart",
    "obs_c_arrive",
    "flow_label_outlier",
    "example_weight",
]

SEQUENCE_COLUMN = "sequence_features"


@dataclass(frozen=True)
class SequenceSpec:
    seq_len: int = 24
    seq_step_minutes: int = 2

    @property
    def channels(self) -> tuple[str, ...]:
        return (
            "ebikes_frac",
            "classic_frac",
            "total_frac",
            "docks_frac",
            "is_renting",
            "is_returning",
            "status_age_log",
            "delta_ebikes",
            "delta_total",
            "zero_flag",
            "full_flag",
        )


def _as_utc_naive(ts) -> pd.Timestamp:
    out = pd.Timestamp(ts)
    if out.tzinfo is not None:
        out = out.tz_convert("UTC").tz_localize(None)
    return out


def _finite_series(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default).astype(float)


def _local_hour_dow(anchor_ts: pd.Series) -> tuple[pd.Series, pd.Series]:
    ts = pd.to_datetime(anchor_ts, utc=True, errors="coerce")
    local = ts.dt.tz_convert("America/Chicago")
    return local.dt.hour.fillna(0).astype(int), local.dt.dayofweek.fillna(0).astype(int)


def _shifted_group_stats(
    frame: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
    *,
    default_value: float,
) -> tuple[pd.Series, pd.Series]:
    values = _finite_series(frame, value_col, default_value)
    grouped = values.groupby([frame[col] for col in group_cols], sort=False)
    count = grouped.cumcount()
    cumsum = grouped.cumsum() - values
    mean = cumsum / count.replace(0, np.nan)
    return mean.astype(float), count.astype(float)


def _eb_rate(successes: pd.Series, counts: pd.Series, parent: pd.Series | float, alpha: float) -> pd.Series:
    parent_series = (
        pd.Series(float(parent), index=successes.index, dtype=float)
        if np.isscalar(parent)
        else pd.to_numeric(parent, errors="coerce").astype(float)
    )
    out = (successes.astype(float) + alpha * parent_series) / (counts.astype(float) + alpha)
    return out.replace([np.inf, -np.inf], np.nan).fillna(parent_series).clip(0.001, 0.999)


def add_shifted_empirical_priors(
    examples: pd.DataFrame,
    *,
    alpha: float = 50.0,
    global_default: float = 0.35,
    sort_columns: Iterable[str] = ("anchor_ts", "station_id", "horizon_minutes"),
) -> pd.DataFrame:
    """Add chronological empirical-Bayes priors with the current row shifted out.

    The returned priors use labels only from rows ordered strictly before each
    row in the provided frame. For validation/test leakage control, call
    ``apply_train_valid_shifted_priors`` so validation priors are initialized
    from training history only.
    """
    if examples.empty:
        return examples.copy()
    out = examples.copy()
    if "anchor_ts" not in out.columns:
        raise ValueError("examples must include anchor_ts")
    out["anchor_ts"] = pd.to_datetime(out["anchor_ts"], errors="coerce")
    if "local_hour" not in out.columns or "dow" not in out.columns:
        out["local_hour"], out["dow"] = _local_hour_dow(out["anchor_ts"])
    if "has_ebike" not in out.columns and "y_has_ebike" in out.columns:
        out["has_ebike"] = out["y_has_ebike"]
    if "has_ebike" not in out.columns:
        out["has_ebike"] = 0
    if "future_ebikes" not in out.columns and "e_future" in out.columns:
        out["future_ebikes"] = out["e_future"]
    if "future_total_bikes" not in out.columns and "q_future" in out.columns:
        out["future_total_bikes"] = out["q_future"]
    for column in ["future_ebikes", "future_total_bikes"]:
        if column not in out.columns:
            out[column] = 0.0
    for column in ["obs_e_depart", "obs_c_depart", "obs_e_arrive", "obs_c_arrive"]:
        if column not in out.columns:
            out[column] = 0.0

    order_cols = [c for c in sort_columns if c in out.columns]
    ordered = out.sort_values(order_cols, kind="mergesort").copy()
    y = _finite_series(ordered, "has_ebike", 0.0).clip(0.0, 1.0)

    global_count = pd.Series(np.arange(len(ordered)), index=ordered.index, dtype=float)
    global_success = y.cumsum() - y
    global_rate = _eb_rate(global_success, global_count, global_default, alpha)

    hd_mean, hd_n = _shifted_group_stats(
        ordered,
        ["local_hour", "dow"],
        "has_ebike",
        default_value=global_default,
    )
    hd_success = hd_mean.fillna(global_rate) * hd_n
    hour_dow = _eb_rate(hd_success, hd_n, global_rate, alpha)

    station_mean, station_n = _shifted_group_stats(
        ordered,
        ["station_id"],
        "has_ebike",
        default_value=global_default,
    )
    station_success = station_mean.fillna(global_rate) * station_n
    station_rate = _eb_rate(station_success, station_n, hour_dow, alpha)

    st_hour_mean, st_hour_n = _shifted_group_stats(
        ordered,
        ["station_id", "local_hour"],
        "has_ebike",
        default_value=global_default,
    )
    st_hour_success = st_hour_mean.fillna(station_rate) * st_hour_n
    station_hour_rate = _eb_rate(st_hour_success, st_hour_n, station_rate, alpha)

    shd_mean, shd_n = _shifted_group_stats(
        ordered,
        ["station_id", "local_hour", "dow"],
        "has_ebike",
        default_value=global_default,
    )
    shd_success = shd_mean.fillna(station_hour_rate) * shd_n
    station_hour_dow = _eb_rate(shd_success, shd_n, station_hour_rate, alpha)

    e_mean, _ = _shifted_group_stats(
        ordered,
        ["station_id", "local_hour", "dow"],
        "future_ebikes",
        default_value=float(_finite_series(ordered, "future_ebikes", 0.0).mean() or 0.0),
    )
    q_mean, _ = _shifted_group_stats(
        ordered,
        ["station_id", "local_hour", "dow"],
        "future_total_bikes",
        default_value=float(_finite_series(ordered, "future_total_bikes", 0.0).mean() or 0.0),
    )
    dep_success = _finite_series(ordered, "obs_e_depart", 0.0) + _finite_series(ordered, "obs_c_depart", 0.0)
    arr_success = _finite_series(ordered, "obs_e_arrive", 0.0) + _finite_series(ordered, "obs_c_arrive", 0.0)
    ordered["_depart_success_tmp"] = dep_success
    ordered["_arrive_success_tmp"] = arr_success
    dep_mean, _ = _shifted_group_stats(ordered, ["station_id", "local_hour", "dow"], "_depart_success_tmp", default_value=0.0)
    arr_mean, _ = _shifted_group_stats(ordered, ["station_id", "local_hour", "dow"], "_arrive_success_tmp", default_value=0.0)
    ordered = ordered.drop(columns=["_depart_success_tmp", "_arrive_success_tmp"])

    ordered["global_rate_shifted"] = global_rate
    ordered["hour_dow_global_rate_shifted"] = hour_dow
    ordered["station_hour_dow_has_ebike_rate_shifted"] = station_hour_dow
    ordered["station_hour_dow_e_mean_shifted"] = e_mean.fillna(_finite_series(ordered, "future_ebikes", 0.0).expanding().mean().shift(1)).fillna(0.0)
    ordered["station_hour_dow_q_mean_shifted"] = q_mean.fillna(_finite_series(ordered, "future_total_bikes", 0.0).expanding().mean().shift(1)).fillna(0.0)
    ordered["station_hour_dow_depart_rate_shifted"] = dep_mean.fillna(0.0).clip(lower=0.0)
    ordered["station_hour_dow_arrive_rate_shifted"] = arr_mean.fillna(0.0).clip(lower=0.0)

    if "station_neighbor_same_hour_rate" in ordered.columns:
        ordered["neighbor_hour_dow_rate_shifted"] = pd.to_numeric(
            ordered["station_neighbor_same_hour_rate"], errors="coerce"
        ).fillna(hour_dow)
    else:
        ordered["neighbor_hour_dow_rate_shifted"] = hour_dow

    ordered["station_same_hour_rate"] = ordered["station_hour_dow_has_ebike_rate_shifted"]
    ordered["station_same_hour_n"] = shd_n.fillna(0).astype(int)
    ordered["nearby_same_hour_rate"] = ordered["hour_dow_global_rate_shifted"]
    ordered["station_neighbor_same_hour_rate"] = ordered["neighbor_hour_dow_rate_shifted"]
    ordered["station_neighbor_recent_zero_rate"] = 1.0 - ordered["station_neighbor_same_hour_rate"].clip(0.0, 1.0)
    for column in SHIFTED_PRIOR_COLUMNS:
        ordered[column] = pd.to_numeric(ordered[column], errors="coerce").fillna(global_default)
    return ordered.sort_index()


def apply_train_valid_shifted_priors(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    *,
    alpha: float = 50.0,
    global_default: float = 0.35,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute train priors chronologically and validation priors walk-forward.

    Validation rows are appended after training rows for the shifted expanding
    calculation, so each validation row can see training history and earlier
    validation anchors only. It cannot see its own or later labels.
    """
    train_in = train.copy()
    valid_in = valid.copy()
    train_in["_dg_prior_split"] = "train"
    valid_in["_dg_prior_split"] = "valid"
    combined = pd.concat([train_in, valid_in], ignore_index=False, sort=False)
    if combined.empty:
        return train.copy(), valid.copy()
    combined = combined.sort_values(["anchor_ts", "station_id", "horizon_minutes"], kind="mergesort")
    enriched = add_shifted_empirical_priors(combined, alpha=alpha, global_default=global_default)
    train_out = enriched[enriched["_dg_prior_split"] == "train"].drop(columns=["_dg_prior_split"])
    valid_out = enriched[enriched["_dg_prior_split"] == "valid"].drop(columns=["_dg_prior_split"])
    return train_out.sort_index(), valid_out.sort_index()


def observed_flow_labels_from_states(
    e_now: int,
    q_now: int,
    e_future: int,
    q_future: int,
) -> dict[str, float]:
    c_now = max(0, int(q_now) - int(e_now))
    c_future = max(0, int(q_future) - int(e_future))
    return {
        "obs_e_depart": float(max(int(e_now) - int(e_future), 0)),
        "obs_e_arrive": float(max(int(e_future) - int(e_now), 0)),
        "obs_c_depart": float(max(c_now - c_future, 0)),
        "obs_c_arrive": float(max(c_future - c_now, 0)),
    }


def build_sequence_from_status(
    group: pd.DataFrame,
    anchor_ts,
    *,
    spec: SequenceSpec = SequenceSpec(),
) -> list[list[float]]:
    """Build a compact backward-as-of status sequence ending at anchor_ts."""
    if group.empty:
        return [[0.0] * len(spec.channels) for _ in range(spec.seq_len)]
    anchor = _as_utc_naive(anchor_ts)
    g = group.copy()
    if "observation_ts" not in g.columns:
        if "last_reported" in g.columns:
            g["observation_ts"] = g["last_reported"]
        elif "fetched_at" in g.columns:
            g["observation_ts"] = g["fetched_at"]
        else:
            return [[0.0] * len(spec.channels) for _ in range(spec.seq_len)]
    g["observation_ts"] = pd.to_datetime(g["observation_ts"], errors="coerce")
    g = g.dropna(subset=["observation_ts"]).sort_values("observation_ts").reset_index(drop=True)
    g = g[g["observation_ts"] <= anchor]
    if g.empty:
        return [[0.0] * len(spec.channels) for _ in range(spec.seq_len)]
    obs = g["observation_ts"].to_numpy(dtype="datetime64[ns]")
    out: list[list[float]] = []
    prev_e: float | None = None
    prev_q: float | None = None
    for pos in range(spec.seq_len):
        minutes_back = (spec.seq_len - 1 - pos) * int(spec.seq_step_minutes)
        lookup = np.datetime64((anchor - pd.Timedelta(minutes=minutes_back)).to_datetime64(), "ns")
        idx = int(np.searchsorted(obs, lookup, side="right") - 1)
        if idx < 0:
            row = g.iloc[0]
        else:
            row = g.iloc[idx]
        capacity = float(pd.to_numeric(pd.Series([row.get("capacity", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
        capacity = max(1.0, capacity)
        e = float(pd.to_numeric(pd.Series([row.get("num_ebikes_available", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
        q = float(pd.to_numeric(pd.Series([row.get("num_bikes_available", e)]), errors="coerce").fillna(e).iloc[0])
        docks = float(pd.to_numeric(pd.Series([row.get("num_docks_available", max(capacity - q, 0.0))]), errors="coerce").fillna(max(capacity - q, 0.0)).iloc[0])
        classic = max(q - e, 0.0)
        delta_e = 0.0 if prev_e is None else e - prev_e
        delta_q = 0.0 if prev_q is None else q - prev_q
        prev_e, prev_q = e, q
        status_age = row.get("status_age_minutes", 0.0)
        try:
            status_age_float = float(status_age)
        except (TypeError, ValueError):
            status_age_float = 0.0
        out.append([
            float(np.clip(e / capacity, 0.0, 1.0)),
            float(np.clip(classic / capacity, 0.0, 1.0)),
            float(np.clip(q / capacity, 0.0, 1.0)),
            float(np.clip(docks / capacity, 0.0, 1.0)),
            float(bool(row.get("is_renting", True))),
            float(bool(row.get("is_returning", True))),
            float(np.log1p(max(0.0, status_age_float)) / math.log1p(120.0)),
            float(np.clip(delta_e / capacity, -1.0, 1.0)),
            float(np.clip(delta_q / capacity, -1.0, 1.0)),
            float(e <= 0.0),
            float(q >= capacity),
        ])
    return out


def fallback_sequence_from_aggregate(rows: pd.DataFrame, *, spec: SequenceSpec = SequenceSpec()) -> np.ndarray:
    """Generate a safe aggregate-history fallback sequence with no DB access."""
    if rows.empty:
        return np.zeros((0, spec.seq_len, len(spec.channels)), dtype=np.float32)
    capacity = _finite_series(rows, "capacity", np.nan).fillna(_finite_series(rows, "capacity_clipped", 15.0)).clip(1.0, 80.0)
    e_now = _finite_series(rows, "num_ebikes_available", np.nan).fillna(_finite_series(rows, "current_ebikes_clipped", 0.0)).clip(0.0, 80.0)
    q_now = _finite_series(rows, "num_bikes_available", np.nan).fillna(_finite_series(rows, "current_total_bikes_clipped", 0.0)).clip(lower=e_now, upper=80.0)
    docks = _finite_series(rows, "num_docks_available", np.nan).fillna((capacity - q_now).clip(lower=0.0)).clip(0.0, 80.0)
    trend = (
        0.50 * _finite_series(rows, "trend_5m", 0.0)
        + 0.30 * _finite_series(rows, "trend_10m", 0.0)
        + 0.20 * _finite_series(rows, "trend_15m", 0.0)
    ).clip(-6.0, 6.0)
    is_renting = _finite_series(rows, "is_renting", 1.0).clip(0.0, 1.0)
    is_returning = _finite_series(rows, "is_returning", 1.0).clip(0.0, 1.0)
    status_age = _finite_series(rows, "status_age_minutes", 0.0).clip(0.0, 120.0)
    arr = np.zeros((len(rows), spec.seq_len, len(spec.channels)), dtype=np.float32)
    for k in range(spec.seq_len):
        frac = (spec.seq_len - 1 - k) / max(1, spec.seq_len - 1)
        e = (e_now - trend * frac).clip(0.0, capacity)
        q = (q_now - trend * frac).clip(lower=e, upper=capacity)
        d = (capacity - q).clip(lower=0.0)
        classic = (q - e).clip(lower=0.0)
        delta_e = np.zeros(len(rows)) if k == 0 else trend / max(1.0, spec.seq_len - 1)
        delta_q = delta_e
        arr[:, k, 0] = (e / capacity).to_numpy(dtype=np.float32)
        arr[:, k, 1] = (classic / capacity).to_numpy(dtype=np.float32)
        arr[:, k, 2] = (q / capacity).to_numpy(dtype=np.float32)
        arr[:, k, 3] = (d / capacity).to_numpy(dtype=np.float32)
        arr[:, k, 4] = is_renting.to_numpy(dtype=np.float32)
        arr[:, k, 5] = is_returning.to_numpy(dtype=np.float32)
        arr[:, k, 6] = (np.log1p(status_age) / math.log1p(120.0)).to_numpy(dtype=np.float32)
        arr[:, k, 7] = (delta_e / capacity).clip(-1.0, 1.0).to_numpy(dtype=np.float32)
        arr[:, k, 8] = (delta_q / capacity).clip(-1.0, 1.0).to_numpy(dtype=np.float32)
        arr[:, k, 9] = (e <= 0.0).astype(float).to_numpy(dtype=np.float32)
        arr[:, k, 10] = (q >= capacity).astype(float).to_numpy(dtype=np.float32)
    return arr


def sequence_array_from_rows(rows: pd.DataFrame, *, spec: SequenceSpec = SequenceSpec()) -> tuple[np.ndarray, np.ndarray]:
    """Return sequence tensor and mask; falls back to aggregate trends when absent."""
    expected_shape = (spec.seq_len, len(spec.channels))
    if SEQUENCE_COLUMN not in rows.columns:
        seq = fallback_sequence_from_aggregate(rows, spec=spec)
        return seq, np.ones((len(rows),), dtype=np.float32)
    seq = np.zeros((len(rows), spec.seq_len, len(spec.channels)), dtype=np.float32)
    fallback_mask = np.zeros((len(rows),), dtype=np.float32)
    fallback = fallback_sequence_from_aggregate(rows, spec=spec)
    for i, value in enumerate(rows[SEQUENCE_COLUMN].tolist()):
        try:
            arr = np.asarray(value, dtype=np.float32)
        except Exception:
            arr = np.empty((0, 0), dtype=np.float32)
        if arr.shape != expected_shape or not np.isfinite(arr).all():
            seq[i] = fallback[i]
            fallback_mask[i] = 1.0
        else:
            seq[i] = arr
    return seq, fallback_mask
