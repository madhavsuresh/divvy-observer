from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Iterable

import duckdb
import numpy as np
import pandas as pd

from .dg_nissm_features import (
    SequenceSpec,
    add_shifted_empirical_priors,
    build_sequence_from_status,
    observed_flow_labels_from_states,
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _ids_clause(ids: Iterable[str]) -> tuple[str, list[str]]:
    values = [str(value) for value in ids if value is not None]
    if not values:
        return "", []
    return ",".join(["?"] * len(values)), values


def _as_timestamp(value) -> pd.Timestamp | None:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        return None
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def _table_has_column(conn: duckdb.DuckDBPyConnection, table_name: str, column_name: str) -> bool:
    try:
        row = conn.execute(
            """
            SELECT COUNT(*)
            FROM information_schema.columns
            WHERE table_name = ? AND column_name = ?
            """,
            [table_name, column_name],
        ).fetchone()
        return bool(row and row[0])
    except Exception:
        return False


def _status_bounds(
    conn: duckdb.DuckDBPyConnection,
    station_ids: list[str] | None,
    clock_col: str = "fetched_at",
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    station_filter = ""
    params: list[object] = []
    if station_ids:
        placeholders, ids = _ids_clause(station_ids)
        station_filter = f"WHERE station_id IN ({placeholders})"
        params.extend(ids)
    if clock_col == "fetched_at" and _table_has_column(conn, "station_status", "fetched_at"):
        clock_expr = "COALESCE(fetched_at, last_reported)"
    else:
        clock_expr = "last_reported"
    row = conn.execute(
        f"""
        SELECT MIN({clock_expr}), MAX({clock_expr})
        FROM station_status
        {station_filter}
        """,
        params,
    ).fetchone()
    if not row:
        return None, None
    return _as_timestamp(row[0]), _as_timestamp(row[1])


def _fetch_status(
    conn: duckdb.DuckDBPyConnection,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    station_ids: list[str] | None,
    max_source_rows: int,
    clock_col: str = "fetched_at",
) -> pd.DataFrame:
    station_filter = ""
    params: list[object] = [start.to_pydatetime(), end.to_pydatetime()]
    if station_ids:
        placeholders, ids = _ids_clause(station_ids)
        station_filter = f"AND ss.station_id IN ({placeholders})"
        params.extend(ids)
    params.append(int(max_source_rows))
    has_fetched_at = _table_has_column(conn, "station_status", "fetched_at")
    fetched_expr = "ss.fetched_at" if has_fetched_at else "ss.last_reported"
    if clock_col == "fetched_at" and has_fetched_at:
        observation_expr = "COALESCE(ss.fetched_at, ss.last_reported)"
    else:
        observation_expr = "ss.last_reported"
    rows = conn.execute(
        f"""
        SELECT
          ss.station_id,
          ss.last_reported AS source_last_reported,
          {fetched_expr} AS fetched_at,
          {observation_expr} AS observation_ts,
          ss.num_bikes_available,
          ss.num_ebikes_available,
          ss.num_docks_available,
          ss.is_renting,
          ss.is_returning,
          s.capacity,
          s.lat,
          s.lon
        FROM station_status ss
        LEFT JOIN stations s USING (station_id)
        WHERE {observation_expr} >= ?
          AND {observation_expr} <= ?
          AND ss.num_ebikes_available IS NOT NULL
          {station_filter}
        ORDER BY ss.station_id, observation_ts, source_last_reported, fetched_at
        LIMIT ?
        """,
        params,
    ).df()
    if rows.empty:
        return rows
    for column in ["source_last_reported", "fetched_at", "observation_ts"]:
        rows[column] = pd.to_datetime(rows[column], utc=True).dt.tz_localize(None)
    rows["last_reported"] = rows["observation_ts"]
    rows["status_age_minutes"] = (
        rows["observation_ts"] - rows["source_last_reported"]
    ).dt.total_seconds() / 60.0
    for column in [
        "num_bikes_available",
        "num_ebikes_available",
        "num_docks_available",
        "capacity",
    ]:
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    return rows.sort_values(["station_id", "observation_ts", "source_last_reported", "fetched_at"]).reset_index(drop=True)


def _latest_current_index(group: pd.DataFrame, anchor: pd.Timestamp) -> int | None:
    reported = group["observation_ts"].to_numpy(dtype="datetime64[ns]")
    anchor64 = np.datetime64(anchor.to_datetime64(), "ns")
    idx = int(np.searchsorted(reported, anchor64, side="right") - 1)
    return idx if idx >= 0 else None


def _latest_label_index(group: pd.DataFrame, target: pd.Timestamp) -> int | None:
    reported = group["observation_ts"].to_numpy(dtype="datetime64[ns]")
    target64 = np.datetime64(target.to_datetime64(), "ns")
    idx = int(np.searchsorted(reported, target64, side="right") - 1)
    return idx if idx >= 0 else None


def _core_examples_for_station(
    station_id: str,
    group: pd.DataFrame,
    *,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    horizons: tuple[int, ...],
    anchor_every_min: int,
    max_current_status_age_min: int,
    max_label_status_age_min: int,
    include_sequences: bool = False,
    sequence_spec: SequenceSpec = SequenceSpec(),
) -> list[dict]:
    anchors = pd.date_range(start_ts, end_ts, freq=f"{int(anchor_every_min)}min")
    if anchors.empty:
        return []
    out: list[dict] = []
    for anchor in anchors:
        cur_idx = _latest_current_index(group, anchor)
        if cur_idx is None:
            continue
        current = group.iloc[cur_idx]
        current_observation_ts = pd.Timestamp(current["observation_ts"])
        current_age = (anchor - current_observation_ts).total_seconds() / 60.0
        if current_age < -1e-9 or current_age > max_current_status_age_min:
            continue
        for horizon in horizons:
            target = anchor + pd.Timedelta(minutes=int(horizon))
            label_idx = _latest_label_index(group, target)
            if label_idx is None:
                continue
            label = group.iloc[label_idx]
            label_observation_ts = pd.Timestamp(label["observation_ts"])
            label_age = (target - label_observation_ts).total_seconds() / 60.0
            if label_age < -1e-9 or label_age > max_label_status_age_min:
                continue
            e_now = int(max(0, current.get("num_ebikes_available") or 0))
            q_now = int(max(e_now, current.get("num_bikes_available") or e_now))
            docks_now = int(max(0, current.get("num_docks_available") or 0))
            e_future = int(max(0, label.get("num_ebikes_available") or 0))
            q_future = int(max(e_future, label.get("num_bikes_available") or e_future))
            docks_future = int(max(0, label.get("num_docks_available") or 0))
            flow = _observed_flow_labels_between(group, cur_idx, label_idx)
            row = {
                "station_id": station_id,
                "anchor_ts": anchor.to_pydatetime(),
                "target_at": target.to_pydatetime(),
                "current_reported_at": current_observation_ts.to_pydatetime(),
                "label_reported_at": label_observation_ts.to_pydatetime(),
                "current_observation_ts": current_observation_ts.to_pydatetime(),
                "label_observation_ts": label_observation_ts.to_pydatetime(),
                "current_status_age_minutes": float(current_age),
                "label_status_age_minutes": float(label_age),
                "status_age_minutes": float(current.get("status_age_minutes") or 0.0),
                "horizon_minutes": int(horizon),
                "e_now": e_now,
                "q_now": q_now,
                "docks_now": docks_now,
                "e_future": e_future,
                "q_future": q_future,
                "docks_future": docks_future,
                "y_has_ebike": int(e_future >= 1),
                "y_appears": int(e_now == 0 and e_future >= 1),
                "y_survives": int(e_now > 0 and e_future >= 1),
                "num_ebikes_available": e_now,
                "num_bikes_available": q_now,
                "num_docks_available": docks_now,
                "future_ebikes": e_future,
                "future_total_bikes": q_future,
                "future_docks": docks_future,
                "future_reported": label_observation_ts.to_pydatetime(),
                "has_ebike": int(e_future >= 1),
                "last_reported": anchor.to_pydatetime(),
                "fetched_at": current_observation_ts.to_pydatetime(),
                "label_fetched_at": label_observation_ts.to_pydatetime(),
                "source_last_reported": pd.Timestamp(current["source_last_reported"]).to_pydatetime(),
                "label_source_last_reported": pd.Timestamp(label["source_last_reported"]).to_pydatetime(),
                "capacity": current.get("capacity"),
                "lat": current.get("lat"),
                "lon": current.get("lon"),
                "is_renting": bool(current.get("is_renting", True)),
                "is_returning": bool(current.get("is_returning", True)),
                **flow,
            }
            total_flow = (
                row["obs_e_depart"]
                + row["obs_e_arrive"]
                + row["obs_c_depart"]
                + row["obs_c_arrive"]
            )
            one_min_jump = total_flow / max(1.0, float(horizon))
            row["flow_label_outlier"] = bool(total_flow > max(8.0, 4.0 * math.sqrt(max(1.0, float(q_now + docks_now)))))
            row["example_weight"] = 0.25 if row["flow_label_outlier"] or one_min_jump > 8.0 else 1.0
            if include_sequences:
                row["sequence_features"] = build_sequence_from_status(
                    group,
                    anchor,
                    spec=sequence_spec,
                )
            out.append(row)
    return out


def _observed_flow_labels_between(group: pd.DataFrame, cur_idx: int, label_idx: int) -> dict[str, float]:
    if label_idx <= cur_idx:
        current = group.iloc[cur_idx]
        return observed_flow_labels_from_states(
            int(max(0, current.get("num_ebikes_available") or 0)),
            int(max(0, current.get("num_bikes_available") or 0)),
            int(max(0, current.get("num_ebikes_available") or 0)),
            int(max(0, current.get("num_bikes_available") or 0)),
        )
    rows = group.iloc[cur_idx : label_idx + 1]
    obs_e_depart = 0.0
    obs_e_arrive = 0.0
    obs_c_depart = 0.0
    obs_c_arrive = 0.0
    prev_e = int(max(0, rows.iloc[0].get("num_ebikes_available") or 0))
    prev_q = int(max(prev_e, rows.iloc[0].get("num_bikes_available") or prev_e))
    prev_c = max(prev_q - prev_e, 0)
    for row in rows.iloc[1:].itertuples(index=False):
        e = int(max(0, getattr(row, "num_ebikes_available", 0) or 0))
        q = int(max(e, getattr(row, "num_bikes_available", e) or e))
        c = max(q - e, 0)
        obs_e_depart += max(prev_e - e, 0)
        obs_e_arrive += max(e - prev_e, 0)
        obs_c_depart += max(prev_c - c, 0)
        obs_c_arrive += max(c - prev_c, 0)
        prev_e, prev_q, prev_c = e, q, c
    return {
        "obs_e_depart": float(obs_e_depart),
        "obs_e_arrive": float(obs_e_arrive),
        "obs_c_depart": float(obs_c_depart),
        "obs_c_arrive": float(obs_c_arrive),
    }


def _add_predictor_compatible_features(conn: duckdb.DuckDBPyConnection, examples: pd.DataFrame) -> pd.DataFrame:
    if examples.empty:
        return examples
    from . import predictor

    out = examples.copy()
    out = predictor.add_temporal_features(out, "anchor_ts")
    out = predictor.add_calendar_features(out, "anchor_ts")
    out = predictor._add_inventory_features(out)
    out["current_ebikes_clipped"] = out["num_ebikes_available"].fillna(0).clip(0, 6)
    out["current_bucket"] = out["num_ebikes_available"].fillna(0).map(predictor.current_bucket)
    out["has_ebike_now"] = (out["num_ebikes_available"].fillna(0) >= 1).astype(int)

    out = add_shifted_empirical_priors(out)
    global_mean = float(out["global_rate_shifted"].iloc[-1]) if len(out) else 0.35
    out["station_neighbor_count_500m"] = 0
    out["station_neighbor_capacity_500m"] = 0.0
    out["station_neighbor_recent_ebikes"] = 0.0
    out["station_neighbor_recent_zero_rate"] = 1.0 - out["station_neighbor_same_hour_rate"].clip(0.0, 1.0)

    out["trend_5m"] = 0.0
    out["trend_10m"] = 0.0
    out["trend_15m"] = 0.0
    out["churn_rate"] = 0.0
    for column in predictor.LIVE_INFLIGHT_FEATURE_COLUMNS:
        out[column] = 0.0
    for column in predictor.FREE_FLOATING_FEATURE_COLUMNS:
        out[column] = 0.0
    if "status_age_minutes" not in out.columns:
        out["status_age_minutes"] = out["current_status_age_minutes"]
    out["station_closed_penalty_flag"] = (~out["is_renting"].fillna(True).astype(bool)).astype(int)
    out["stale_status_penalty_flag"] = (out["status_age_minutes"].fillna(999.0) > 10.0).astype(int)
    out = predictor._add_trip_features(conn, out, now=None)
    out = predictor._add_weather_features(conn, out, "anchor_ts")
    out = predictor._fill_feature_defaults(out, predictor.CALENDAR_FEATURE_COLUMNS)
    out = predictor._fill_feature_defaults(out, predictor.TRIP_FEATURE_COLUMNS)
    out = predictor._fill_feature_defaults(out, predictor.LIVE_INFLIGHT_FEATURE_COLUMNS)
    out = predictor._fill_feature_defaults(out, predictor.FREE_FLOATING_FEATURE_COLUMNS)
    out = predictor._fill_feature_defaults(out, predictor.STATUS_QUALITY_FEATURE_COLUMNS)
    out = predictor._weather_defaults(out)
    for column in predictor.FEATURE_COLUMNS:
        if column not in out.columns:
            out[column] = 0.0
    out[predictor.FEATURE_COLUMNS] = out[predictor.FEATURE_COLUMNS].fillna(0.0)
    return out


def build_leak_free_examples(
    conn: duckdb.DuckDBPyConnection,
    start_ts=None,
    end_ts=None,
    station_ids=None,
    horizons=(5, 10, 15, 20),
    anchor_every_min=2,
    history_hours=24 * 30,
    max_current_status_age_min=10,
    max_label_status_age_min=20,
    max_source_rows=2_000_000,
    clock_col: str = "fetched_at",
    include_sequences: bool = False,
    seq_len: int = 24,
    seq_step_minutes: int = 2,
) -> pd.DataFrame:
    """Build offline examples using backward-as-of current and label state.

    ``station_status`` rows are state-change rows, so a target between changes
    inherits the latest status at or before the target. No label row after
    ``target_at`` is used.
    """
    station_ids_list = [str(s) for s in station_ids] if station_ids is not None else None
    min_status, max_status = _status_bounds(conn, station_ids_list, clock_col=clock_col)
    if min_status is None or max_status is None:
        return pd.DataFrame()

    end = _as_timestamp(end_ts) or min(max_status, pd.Timestamp(_utc_now()))
    start = _as_timestamp(start_ts) or max(min_status, end - pd.Timedelta(hours=int(history_hours)))
    if end < start:
        return pd.DataFrame()

    max_horizon = max(int(h) for h in horizons) if horizons else 0
    fetch_start = start - pd.Timedelta(minutes=max_current_status_age_min + 1)
    fetch_end = end + pd.Timedelta(minutes=max_horizon)
    status = _fetch_status(
        conn,
        start=fetch_start,
        end=fetch_end,
        station_ids=station_ids_list,
        max_source_rows=max_source_rows,
        clock_col=clock_col,
    )
    if status.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    horizons_tuple = tuple(int(h) for h in horizons)
    sequence_spec = SequenceSpec(seq_len=int(seq_len), seq_step_minutes=int(seq_step_minutes))
    for station_id, group in status.groupby("station_id", sort=False):
        g = group.sort_values(["observation_ts", "source_last_reported", "fetched_at"]).reset_index(drop=True)
        rows.extend(
            _core_examples_for_station(
                str(station_id),
                g,
                start_ts=start,
                end_ts=end,
                horizons=horizons_tuple,
                anchor_every_min=int(anchor_every_min),
                max_current_status_age_min=int(max_current_status_age_min),
                max_label_status_age_min=int(max_label_status_age_min),
                include_sequences=bool(include_sequences),
                sequence_spec=sequence_spec,
            )
        )
    if not rows:
        return pd.DataFrame()
    examples = pd.DataFrame(rows)
    return _add_predictor_compatible_features(conn, examples)
