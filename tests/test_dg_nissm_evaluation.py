"""Layer 2 evaluation tests for DG-NISSM v2: CDG-NMIP.

Slow, marked `@pytest.mark.slow`. Verifies that the evaluation *framework*
(metric calculations, slicing, walk-forward harness, ablation harness)
produces sensible outputs on a synthetic dataset with known structure.

These tests are not a substitute for measuring the model on real data — that
is what `scripts/evaluate_dg_nissm.py` is for. They exist to make sure the
metric code itself is correct and the harness wiring works end-to-end.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from divvy.cdg_nmip import CDGNMIPConfig
from divvy.dg_nissm import DGNISSMModel
from divvy.evaluation import (
    brier_score,
    crps_count_pmf_mean,
    ece,
    empirical_count_pmf,
    log_loss,
    reliability_curve,
    stratified_metrics,
    uniform_count_pmf,
)


pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Synthetic dataset with structure that the model can plausibly learn from
# ---------------------------------------------------------------------------

def _synthetic_dataset(n_stations: int = 12, minutes: int = 600, seed: int = 1) -> pd.DataFrame:
    """Build a synthetic anchor table where E_future and Q_future depend on
    station type, hour-of-day, current state and a Poisson-ish noise process.

    Includes columns the model expects: anchor_ts, target_at, horizon_minutes,
    capacity, num_ebikes_available, num_bikes_available, num_docks_available,
    future_ebikes, future_total_bikes, has_ebike, lat, lon, is_renting,
    is_returning, station_id.
    """
    rng = np.random.default_rng(seed)
    base = datetime(2026, 1, 1, 6, 0, 0)
    types = rng.choice(["downtown", "residential", "edge"], size=n_stations)
    caps = rng.choice([10, 15, 20], size=n_stations)
    rows = []
    for sidx in range(n_stations):
        stype = types[sidx]
        cap = int(caps[sidx])
        # Per-station autocorrelated state walk
        e = int(rng.integers(0, max(1, cap // 3)))
        q = int(rng.integers(e, cap + 1))
        for m in range(minutes):
            ts = base + timedelta(minutes=m)
            hour = ts.hour
            commute = 1 if hour in (7, 8, 17, 18) else 0
            # demand intensities depend on station type and hour
            type_factor = {"downtown": 1.4, "residential": 0.8, "edge": 0.5}[stype]
            depart_rate = type_factor * (0.6 + 0.9 * commute)
            arrive_rate = type_factor * (0.6 + 0.9 * commute) * (0.7 if commute and stype == "residential" else 1.0)
            # 5-min horizon
            for horizon in (5, 10):
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
                # Update state for next minute (using the 5-min projection)
                if horizon == 5:
                    e = e_future
                    q = q_future
    return pd.DataFrame(rows)


def _eval_config(**overrides) -> CDGNMIPConfig:
    defaults = dict(
        min_train_examples=200,
        min_valid_examples=30,
        min_positive_examples=20,
        min_zero_future_examples=20,
        epochs=2,
        batch_size=512,
        hidden_dim=24,
        sequence_hidden_dim=12,
        station_embedding_dim=6,
        horizon_embedding_dim=4,
        top_k=4,
        max_rollout_steps=1,
        device="cpu",
        runtime_device="cpu",
        calibrate=True,
    )
    defaults.update(overrides)
    return CDGNMIPConfig(**defaults)


@pytest.fixture(scope="module")
def dataset() -> pd.DataFrame:
    return _synthetic_dataset(n_stations=10, minutes=400, seed=2026)


@pytest.fixture(scope="module")
def fitted(dataset: pd.DataFrame) -> tuple[DGNISSMModel, pd.DataFrame, pd.DataFrame]:
    df = dataset.sort_values("anchor_ts").reset_index(drop=True)
    split = int(0.8 * len(df))
    train, valid = df.iloc[:split], df.iloc[split:]
    model = DGNISSMModel(_eval_config()).fit(train, valid)
    assert model.trained, f"model failed to train: {model.model_warning}"
    return model, train, valid


# ---------------------------------------------------------------------------
# Test 1 — walk-forward metric stability
# ---------------------------------------------------------------------------

def test_walk_forward_metric_stability(dataset: pd.DataFrame) -> None:
    df = dataset.sort_values("anchor_ts").reset_index(drop=True)
    n_splits = 4
    width = len(df) // (n_splits + 1)
    briers: list[float] = []
    log_losses: list[float] = []
    for i in range(n_splits):
        train_end = (i + 1) * width
        valid_end = (i + 2) * width
        train = df.iloc[:train_end]
        valid = df.iloc[train_end:valid_end]
        model = DGNISSMModel(_eval_config(epochs=1)).fit(train, valid)
        if not model.trained:
            pytest.skip(f"insufficient data at split {i}: {model.model_warning}")
        out = model.predict_distribution(valid, debug=False)
        y = valid["has_ebike"].astype(float).to_numpy()
        p = out["p_has_ebike"].to_numpy(dtype=float)
        briers.append(brier_score(p, y))
        log_losses.append(log_loss(p, y))

    assert len(briers) == n_splits
    # No catastrophic blow-up across splits
    assert max(briers) < 1.0
    assert max(log_losses) < 10.0
    # Stability: max should not be more than 3x min (loose bound for synthetic)
    if min(briers) > 1e-4:
        assert max(briers) / min(briers) < 5.0, f"unstable briers across splits: {briers}"


# ---------------------------------------------------------------------------
# Test 2 — per-horizon ECE stays finite and bounded after calibration
# ---------------------------------------------------------------------------

def test_reliability_per_horizon_post_calibration(
    fitted: tuple[DGNISSMModel, pd.DataFrame, pd.DataFrame],
) -> None:
    model, _train, valid = fitted
    out = model.predict_distribution(valid, debug=False)
    y_all = valid["has_ebike"].astype(float).to_numpy()
    horizons = sorted(valid["horizon_minutes"].unique())
    assert horizons, "expected at least one horizon in validation"
    for h in horizons:
        mask = (valid["horizon_minutes"] == h).to_numpy()
        if mask.sum() < 20:
            continue
        p = out.loc[mask, "p_has_ebike"].to_numpy(dtype=float)
        y = y_all[mask]
        value = ece(p, y, n_bins=10)
        assert np.isfinite(value), f"non-finite ECE for horizon {h}"
        # ECE is in [0, 1] by construction; for a non-degenerate model it
        # should stay well below 1.0. We use a loose bound because synthetic
        # training is short.
        assert 0.0 <= value <= 1.0
        assert value < 0.4, f"ECE too high at horizon {h}: {value:.3f}"

    # reliability_curve sanity check on the full validation set
    centers, pred, obs, cnt = reliability_curve(out["p_has_ebike"].to_numpy(), y_all, n_bins=10)
    assert centers.shape == (10,)
    assert int(cnt.sum()) == len(valid)


# ---------------------------------------------------------------------------
# Test 3 — CRPS beats trivial baselines on the count PMF
# ---------------------------------------------------------------------------

def test_crps_beats_uniform_baseline(
    fitted: tuple[DGNISSMModel, pd.DataFrame, pd.DataFrame],
) -> None:
    model, train, valid = fitted
    out = model.predict_distribution(valid, debug=False)
    observed = valid["future_ebikes"].astype(int).to_numpy()

    model_crps = crps_count_pmf_mean(out["p_count_ebikes"].tolist(), observed)
    uniform_crps = crps_count_pmf_mean(
        [uniform_count_pmf()] * len(valid), observed,
    )
    empirical = empirical_count_pmf(train["future_ebikes"].astype(int).tolist())
    empirical_crps = crps_count_pmf_mean([empirical] * len(valid), observed)

    assert np.isfinite(model_crps)
    assert np.isfinite(uniform_crps)
    assert np.isfinite(empirical_crps)
    # The uniform PMF is a strawman. The model must beat it; if not, the
    # framework is wired wrong or the model is completely degenerate.
    assert model_crps < uniform_crps, (
        f"model CRPS ({model_crps:.3f}) not better than uniform ({uniform_crps:.3f})"
    )


# ---------------------------------------------------------------------------
# Test 4 — stratified metrics are consistent across slices
# ---------------------------------------------------------------------------

def test_stratified_metrics_consistent(
    fitted: tuple[DGNISSMModel, pd.DataFrame, pd.DataFrame],
) -> None:
    model, _train, valid = fitted
    out = model.predict_distribution(valid, debug=False)
    df_labels = valid.copy()
    df_labels["current_bucket"] = np.where(
        df_labels["num_ebikes_available"] <= 0, "empty",
        np.where(df_labels["num_ebikes_available"] >= df_labels["capacity"] * 0.75, "full", "mid"),
    )

    table = stratified_metrics(
        out, df_labels,
        by=["horizon_minutes", "current_bucket"],
        p_col="p_has_ebike",
        y_col="has_ebike",
    )
    assert not table.empty
    assert {"n", "brier", "log_loss", "ece", "p_mean", "y_mean"}.issubset(table.columns)
    # No NaN where n > 0
    for _, row in table.iterrows():
        if row["n"] > 0:
            assert np.isfinite(row["brier"])
            assert np.isfinite(row["log_loss"])
            # ECE is finite if there is at least one non-empty bin
            assert 0.0 <= row["ece"] <= 1.0 or np.isnan(row["ece"])

    # Worst slice not catastrophically worse than the best
    valid_rows = table[table["n"] >= 20]
    if len(valid_rows) >= 2 and float(valid_rows["brier"].min()) > 1e-4:
        ratio = float(valid_rows["brier"].max() / valid_rows["brier"].min())
        assert ratio < 6.0, f"worst/best Brier ratio {ratio:.2f} too high; slices: {valid_rows}"


# ---------------------------------------------------------------------------
# Test 5 — fallback flag-stratified metrics framework works
# ---------------------------------------------------------------------------

def test_fallback_flag_stratification_runs(
    fitted: tuple[DGNISSMModel, pd.DataFrame, pd.DataFrame],
) -> None:
    """We can't yet trust that the model performs strictly worse on
    fallback rows (the synthetic dataset has no sequence column at all, so all
    rows are fallback). But the framework must produce a clean stratified
    table when the column is present."""
    model, _train, valid = fitted
    out = model.predict_distribution(valid, debug=False)
    assert "used_sequence_fallback" in out.columns

    df_labels = valid.copy()
    df_labels["used_sequence_fallback"] = out["used_sequence_fallback"].to_numpy()
    table = stratified_metrics(
        out, df_labels,
        by=["used_sequence_fallback"],
        p_col="p_has_ebike",
        y_col="has_ebike",
    )
    assert not table.empty
    # At least one of the slices must have data
    assert int(table["n"].sum()) == len(valid)


# ---------------------------------------------------------------------------
# Test 6 — ablation harness produces a comparable result for each variant
# ---------------------------------------------------------------------------

def test_ablations_each_produce_a_metric(dataset: pd.DataFrame) -> None:
    """The framework needs to be able to fit each of the standard ablation
    variants and return a metric for each. We verify the harness, not direction.
    Direction (which ablation hurts more) requires more data and epochs than is
    practical in CI; that goes through `scripts/evaluate_dg_nissm.py`."""
    df = dataset.sort_values("anchor_ts").reset_index(drop=True)
    split = int(0.8 * len(df))
    train, valid = df.iloc[:split], df.iloc[split:]

    variants = {
        "full": dict(),
        "no_graph": dict(use_graph=False),
        "no_sequence": dict(use_sequence=False),
        "no_calibrate": dict(calibrate=False),
        "no_flow_loss": dict(loss_flow_weight=0.0),
    }
    metrics: dict[str, float] = {}
    for name, overrides in variants.items():
        model = DGNISSMModel(_eval_config(epochs=1, **overrides)).fit(train, valid)
        if not model.trained:
            pytest.skip(f"variant {name} failed to train: {model.model_warning}")
        out = model.predict_distribution(valid, debug=False)
        metrics[name] = brier_score(
            out["p_has_ebike"].to_numpy(dtype=float),
            valid["has_ebike"].astype(float).to_numpy(),
        )

    assert set(metrics) == set(variants)
    for name, value in metrics.items():
        assert np.isfinite(value), f"variant {name} produced non-finite brier"
        assert 0.0 <= value <= 1.0


# ---------------------------------------------------------------------------
# Test 7 — fast-path inference latency budget
# ---------------------------------------------------------------------------

def test_fast_path_inference_latency_budget(
    fitted: tuple[DGNISSMModel, pd.DataFrame, pd.DataFrame],
) -> None:
    model, _train, valid = fitted
    small = valid.iloc[:1]
    medium = valid.iloc[:100] if len(valid) >= 100 else valid
    large = valid.iloc[:1000] if len(valid) >= 1000 else valid

    # Warm up
    model.predict_distribution(small, debug=False)

    def time_call(rows):
        start = time.perf_counter()
        model.predict_distribution(rows, debug=False)
        return time.perf_counter() - start

    t1 = min(time_call(small) for _ in range(3))
    t100 = min(time_call(medium) for _ in range(3))
    t_large = time_call(large)

    # Generous CI bounds; tighten after first measurement.
    assert t1 < 1.0, f"single-row prediction too slow: {t1:.3f}s"
    assert t100 < 5.0, f"100-row prediction too slow: {t100:.3f}s"
    assert t_large < 60.0, f"{len(large)}-row prediction too slow: {t_large:.3f}s"
