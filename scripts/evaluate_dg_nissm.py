"""Offline evaluation runner for DG-NISSM v2: CDG-NMIP.

Trains a DG-NISSM model on a configurable training window, evaluates on a
held-out window, and produces a CSV report with one row per
(horizon, current-state bucket, fallback flag) slice.

Two data modes:
 - `--synthetic`: build an in-memory toy dataset (fast smoke test).
 - default: pull examples from the project's DuckDB via
   `divvy.label_builder.build_leak_free_examples`.

Usage examples:

    # Smoke test against synthetic data (no DB required):
    uv run python scripts/evaluate_dg_nissm.py --synthetic --output /tmp/eval.csv

    # Real run against the DuckDB store:
    uv run python scripts/evaluate_dg_nissm.py \
      --train-hours 168 --valid-hours 24 \
      --horizons 5 10 15 20 \
      --output diagnostics/dg_nissm_eval.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Make `src/` importable when running as a script
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from divvy.cdg_nmip import CDGNMIPConfig  # noqa: E402
from divvy.dg_nissm import DGNISSMModel  # noqa: E402
from divvy.evaluation import (  # noqa: E402
    brier_score,
    crps_count_pmf_mean,
    ece,
    empirical_count_pmf,
    log_loss,
    stratified_metrics,
)


def _build_synthetic_examples(
    n_stations: int = 12,
    minutes: int = 600,
    horizons: tuple[int, ...] = (5, 10, 15, 20),
    seed: int = 2026,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = datetime(2026, 1, 1, 6, 0, 0)
    types = rng.choice(["downtown", "residential", "edge"], size=n_stations)
    caps = rng.choice([10, 15, 20], size=n_stations)
    rows = []
    for sidx in range(n_stations):
        stype = types[sidx]
        cap = int(caps[sidx])
        e = int(rng.integers(0, max(1, cap // 3)))
        q = int(rng.integers(e, cap + 1))
        for m in range(minutes):
            ts = base + timedelta(minutes=m)
            hour = ts.hour
            commute = 1 if hour in (7, 8, 17, 18) else 0
            type_factor = {"downtown": 1.4, "residential": 0.8, "edge": 0.5}[stype]
            depart_rate = type_factor * (0.6 + 0.9 * commute)
            arrive_rate = type_factor * (0.6 + 0.9 * commute)
            updated_e, updated_q = e, q
            for horizon in horizons:
                de = int(rng.poisson(depart_rate * horizon * (e / max(1, q)) if q else 0))
                dc = int(rng.poisson(depart_rate * horizon * ((q - e) / max(1, q)) if q else 0))
                ae = int(rng.poisson(arrive_rate * horizon * 0.45))
                ac = int(rng.poisson(arrive_rate * horizon * 0.55))
                de = min(de, e)
                dc = min(dc, q - e)
                e_future = int(np.clip(e - de + ae, 0, cap))
                q_future = int(np.clip(q - de - dc + ae + ac, e_future, cap))
                rows.append({
                    "station_id": f"s{sidx}",
                    "anchor_ts": ts,
                    "target_at": ts + timedelta(minutes=horizon),
                    "horizon_minutes": horizon,
                    "capacity": cap,
                    "num_ebikes_available": e,
                    "num_bikes_available": q,
                    "num_docks_available": cap - q,
                    "future_ebikes": e_future,
                    "future_total_bikes": q_future,
                    "has_ebike": int(e_future >= 1),
                    "lat": 41.0 + sidx * 0.005,
                    "lon": -87.0 + (sidx % 4) * 0.005,
                    "is_renting": True,
                    "is_returning": True,
                })
                if horizon == min(horizons):
                    updated_e, updated_q = e_future, q_future
            e, q = updated_e, updated_q
    return pd.DataFrame(rows)


def _load_duckdb_examples(
    db_path: Path,
    train_hours: int,
    valid_hours: int,
    horizons: tuple[int, ...],
    anchor_every_min: int,
    include_sequences: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    import duckdb  # local import — only needed in DB mode

    from divvy.label_builder import build_leak_free_examples

    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        end = pd.Timestamp.now(tz="UTC").tz_localize(None)
        valid_start = end - pd.Timedelta(hours=int(valid_hours))
        train_start = valid_start - pd.Timedelta(hours=int(train_hours))
        print(f"[evaluate_dg_nissm] building train examples ({train_start} → {valid_start})", flush=True)
        t0 = time.perf_counter()
        train = build_leak_free_examples(
            conn,
            start_ts=train_start,
            end_ts=valid_start,
            horizons=tuple(int(h) for h in horizons),
            anchor_every_min=int(anchor_every_min),
            include_sequences=bool(include_sequences),
        )
        print(f"[evaluate_dg_nissm] train examples: {len(train)} in {time.perf_counter() - t0:.1f}s", flush=True)
        print(f"[evaluate_dg_nissm] building valid examples ({valid_start} → {end})", flush=True)
        t0 = time.perf_counter()
        valid = build_leak_free_examples(
            conn,
            start_ts=valid_start,
            end_ts=end,
            horizons=tuple(int(h) for h in horizons),
            anchor_every_min=int(anchor_every_min),
            include_sequences=bool(include_sequences),
        )
        print(f"[evaluate_dg_nissm] valid examples: {len(valid)} in {time.perf_counter() - t0:.1f}s", flush=True)
    finally:
        conn.close()
    return train, valid


def _evaluate(
    model: DGNISSMModel,
    valid: pd.DataFrame,
    train: pd.DataFrame,
) -> pd.DataFrame:
    out = model.predict_distribution(valid, debug=False)
    labels = valid.copy()
    labels["current_bucket"] = np.where(
        labels["num_ebikes_available"] <= 0, "empty",
        np.where(
            labels["num_ebikes_available"] >= labels["capacity"] * 0.75,
            "full",
            "mid",
        ),
    )
    labels["used_sequence_fallback"] = out["used_sequence_fallback"].to_numpy()

    table = stratified_metrics(
        out, labels,
        by=["horizon_minutes", "current_bucket", "used_sequence_fallback"],
        p_col="p_has_ebike",
        y_col="has_ebike",
        pmf_col="p_count_ebikes",
        obs_count_col="future_ebikes",
    )

    # Baselines for delta computation
    empirical = empirical_count_pmf(train["future_ebikes"].astype(int).tolist())
    obs = labels["future_ebikes"].astype(int).to_numpy()
    baseline_crps = crps_count_pmf_mean([empirical] * len(obs), obs)

    # Overall row at the top
    overall_p = out["p_has_ebike"].to_numpy(dtype=float)
    overall_y = labels["has_ebike"].astype(float).to_numpy()
    overall = pd.DataFrame([{
        "horizon_minutes": "ALL",
        "current_bucket": "ALL",
        "used_sequence_fallback": "ALL",
        "n": int(len(labels)),
        "brier": brier_score(overall_p, overall_y),
        "log_loss": log_loss(overall_p, overall_y),
        "ece": ece(overall_p, overall_y),
        "p_mean": float(np.mean(overall_p)) if len(overall_p) else float("nan"),
        "y_mean": float(np.mean(overall_y)) if len(overall_y) else float("nan"),
        "crps": crps_count_pmf_mean(out["p_count_ebikes"].tolist(), obs),
    }])
    out_table = pd.concat([overall, table], ignore_index=True, sort=False)
    out_table["crps_baseline_empirical"] = baseline_crps
    if "crps" in out_table.columns:
        out_table["crps_delta_vs_baseline"] = out_table["crps"] - baseline_crps
    return out_table


def main() -> int:
    p = argparse.ArgumentParser(description="Offline DG-NISSM CDG-NMIP evaluation")
    p.add_argument("--synthetic", action="store_true", help="Use synthetic data (no DB needed)")
    p.add_argument("--db-path", type=Path, default=Path("data/divvy.duckdb"))
    p.add_argument("--train-hours", type=int, default=24)
    p.add_argument("--valid-hours", type=int, default=4)
    p.add_argument("--horizons", type=int, nargs="+", default=[5, 10, 15, 20])
    p.add_argument("--anchor-every-min", type=int, default=2)
    p.add_argument("--include-sequences", action="store_true",
                   help="Build per-row backward sequence features (slow; off by default — model has a trend-based fallback)")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=512)
    args = p.parse_args()

    if args.synthetic:
        print("[evaluate_dg_nissm] generating synthetic dataset")
        df = _build_synthetic_examples(
            n_stations=10,
            minutes=400,
            horizons=tuple(args.horizons),
        )
        df = df.sort_values("anchor_ts").reset_index(drop=True)
        split = int(0.8 * len(df))
        train, valid = df.iloc[:split], df.iloc[split:]
    else:
        if not args.db_path.exists():
            print(f"[evaluate_dg_nissm] db not found at {args.db_path}; use --synthetic for a smoke test", file=sys.stderr)
            return 2
        print(f"[evaluate_dg_nissm] loading examples from {args.db_path}", flush=True)
        train, valid = _load_duckdb_examples(
            db_path=args.db_path,
            train_hours=args.train_hours,
            valid_hours=args.valid_hours,
            horizons=tuple(args.horizons),
            anchor_every_min=args.anchor_every_min,
            include_sequences=args.include_sequences,
        )

    print(f"[evaluate_dg_nissm] train rows: {len(train)}, valid rows: {len(valid)}")
    if train.empty or valid.empty:
        print("[evaluate_dg_nissm] no data to evaluate", file=sys.stderr)
        return 2

    config = CDGNMIPConfig(
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        device="auto",
        runtime_device="auto",
        min_train_examples=min(200, len(train)),
        min_valid_examples=min(30, len(valid)),
        min_positive_examples=min(20, int(train.get("has_ebike", pd.Series(dtype=int)).sum() or 20)),
        min_zero_future_examples=min(20, int((1 - train.get("has_ebike", pd.Series(dtype=int)).fillna(0)).sum() or 20)),
        top_k=4,
        hidden_dim=32,
        sequence_hidden_dim=16,
        station_embedding_dim=8,
        horizon_embedding_dim=4,
    )

    started = time.perf_counter()
    model = DGNISSMModel(config).fit(train, valid)
    fit_seconds = time.perf_counter() - started
    if not model.trained:
        print(f"[evaluate_dg_nissm] model failed quality gate: {model.model_warning}", file=sys.stderr)
        return 3

    print(f"[evaluate_dg_nissm] training complete in {fit_seconds:.1f}s; method={model.method}")
    table = _evaluate(model, valid, train)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print(f"[evaluate_dg_nissm] wrote {len(table)} rows to {args.output}")
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
