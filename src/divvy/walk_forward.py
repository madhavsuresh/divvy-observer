from __future__ import annotations

import argparse
from datetime import datetime, timedelta

import pandas as pd

from . import db, label_builder, model_registry, predictor
from .cc_nissm import CCNISSMModel


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _score_holdout(model, test: pd.DataFrame) -> dict:
    if test.empty:
        return {"n": 0}
    scored = model.predict_distribution(test)
    p = scored["p_has_ebike"].astype(float).clip(0.001, 0.999)
    y = test["has_ebike"].astype(float)
    brier = float(((p - y) ** 2).mean())
    log_loss = float(-(y * p.apply(__import__("math").log) + (1.0 - y) * (1.0 - p).apply(__import__("math").log)).mean())
    return {"n": int(len(test)), "brier_score": brier, "log_loss": log_loss, "rank_loss": brier + 0.05 * log_loss}


def run_walk_forward(args: argparse.Namespace) -> list[dict]:
    start = _dt(args.start)
    end = _dt(args.end)
    folds = []
    cursor = start
    embargo = timedelta(minutes=int(args.embargo_minutes))
    horizons = tuple(int(h) for h in args.horizons)
    with db.session(read_only=False) as conn:
        db.init_schema(conn)
        while cursor < end:
            train_start = cursor
            train_end = train_start + timedelta(days=args.train_days)
            valid_start = train_end + embargo
            valid_end = valid_start + timedelta(days=args.valid_days)
            test_start = valid_end + embargo
            test_end = test_start + timedelta(days=args.test_days)
            if test_end > end:
                break
            train = label_builder.build_leak_free_examples(conn, train_start, train_end, horizons=horizons)
            valid = label_builder.build_leak_free_examples(conn, valid_start, valid_end, horizons=horizons)
            test = label_builder.build_leak_free_examples(conn, test_start, test_end, horizons=horizons)
            model = CCNISSMModel().fit(train, valid)
            metrics = _score_holdout(model, test)
            artifact = model_registry.save_artifact(
                conn,
                "cc_nissm",
                model,
                {
                    "model_family": "cc_nissm",
                    "model_version": model.model_version,
                    "train_start": train_start,
                    "train_end": train_end,
                    "valid_start": valid_start,
                    "valid_end": valid_end,
                    "horizons": list(horizons),
                    "feature_columns": predictor.FEATURE_COLUMNS,
                    "is_primary_eligible": True,
                },
                metrics,
            )
            folds.append({
                "train_start": train_start.isoformat(),
                "train_end": train_end.isoformat(),
                "valid_start": valid_start.isoformat(),
                "valid_end": valid_end.isoformat(),
                "test_start": test_start.isoformat(),
                "test_end": test_end.isoformat(),
                **metrics,
                **artifact,
            })
            cursor += timedelta(days=args.step_days)
    return folds


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Walk-forward Divvy model evaluation")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--train-days", type=int, default=28)
    parser.add_argument("--valid-days", type=int, default=7)
    parser.add_argument("--test-days", type=int, default=7)
    parser.add_argument("--step-days", type=int, default=7)
    parser.add_argument("--embargo-minutes", type=int, default=20)
    parser.add_argument("--horizons", type=int, nargs="+", default=list(predictor.HORIZONS))
    args = parser.parse_args(argv)
    print(run_walk_forward(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
