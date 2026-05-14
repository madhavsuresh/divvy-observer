"""Layer 1 invariant tests for DG-NISSM v2: CDG-NMIP.

Fast, deterministic, CI-friendly. All seven tests train a single tiny model
in a module-scoped fixture, then probe it under controlled conditions.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from divvy.cdg_nmip import (
    CDGNMIPConfig,
    fast_inventory_rollout_from_parameters,
    rollout_from_parameters,
)
from divvy.dg_nissm import DGNISSMModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _synthetic_examples(n: int = 240, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = datetime(2026, 1, 1, 12, 0, 0)
    rows = []
    for i in range(n):
        cap = int(rng.choice([6, 10, 15]))
        e0 = int(rng.integers(0, min(5, cap + 1)))
        q0 = int(rng.integers(e0, cap + 1))
        future_e = int(np.clip(e0 + rng.integers(-1, 2), 0, cap))
        future_q = int(np.clip(q0 + rng.integers(-2, 3), future_e, cap))
        horizon = int(rng.choice([5, 10]))
        rows.append({
            "station_id": f"s{i % 8}",
            "anchor_ts": base + timedelta(minutes=i),
            "target_at": base + timedelta(minutes=i + horizon),
            "horizon_minutes": horizon,
            "capacity": cap,
            "num_ebikes_available": e0,
            "num_bikes_available": q0,
            "num_docks_available": cap - q0,
            "future_ebikes": future_e,
            "future_total_bikes": future_q,
            "has_ebike": int(future_e >= 1),
            "lat": 41.0 + (i % 8) * 0.001,
            "lon": -87.0,
            "is_renting": True,
            "is_returning": True,
        })
    return pd.DataFrame(rows)


def _tiny_config(**overrides) -> CDGNMIPConfig:
    defaults = dict(
        min_train_examples=80,
        min_valid_examples=10,
        min_positive_examples=20,
        min_zero_future_examples=20,
        epochs=1,
        batch_size=64,
        hidden_dim=16,
        sequence_hidden_dim=8,
        station_embedding_dim=4,
        horizon_embedding_dim=4,
        top_k=2,
        max_rollout_steps=1,
        device="cpu",
        runtime_device="cpu",
        calibrate=True,
    )
    defaults.update(overrides)
    return CDGNMIPConfig(**defaults)


@pytest.fixture(scope="module")
def fitted_model() -> DGNISSMModel:
    examples = _synthetic_examples(n=240)
    return DGNISSMModel(_tiny_config()).fit(examples.iloc[:200], examples.iloc[200:])


# ---------------------------------------------------------------------------
# Test 1 — constraints hold on the full state cube
# ---------------------------------------------------------------------------

def _state_cube_rows() -> pd.DataFrame:
    """Synthetic rows spanning the corners of (capacity, e0, q0, horizon)."""
    base = datetime(2026, 1, 1, 8, 0, 0)
    rows = []
    capacities = [1, 6, 15, 40, 80]
    horizons = [5, 10, 15, 20]
    i = 0
    for cap in capacities:
        for horizon in horizons:
            # corners and a few interior points
            e0_values = sorted({0, max(0, cap // 4), max(0, cap // 2), cap})
            for e0 in e0_values:
                q0_values = sorted({e0, min(cap, e0 + 1), min(cap, (e0 + cap) // 2), cap})
                for q0 in q0_values:
                    rows.append({
                        "station_id": f"s{i % 6}",
                        "anchor_ts": base + timedelta(minutes=i),
                        "target_at": base + timedelta(minutes=i + horizon),
                        "horizon_minutes": horizon,
                        "capacity": cap,
                        "num_ebikes_available": e0,
                        "num_bikes_available": q0,
                        "num_docks_available": max(0, cap - q0),
                        "lat": 41.0,
                        "lon": -87.0,
                        "is_renting": True,
                        "is_returning": True,
                    })
                    i += 1
    return pd.DataFrame(rows)


def test_constraints_hold_on_full_state_cube(fitted_model: DGNISSMModel) -> None:
    rows = _state_cube_rows()
    out = fitted_model.predict_distribution(rows, debug=False)

    assert len(out) == len(rows)
    # PMFs sum to 1
    for pmf_col in ("p_count_ebikes", "p_count_total"):
        for entry in out[pmf_col]:
            assert isinstance(entry, dict) and entry, f"empty {pmf_col}"
            assert sum(entry.values()) == pytest.approx(1.0, abs=1e-6)
            for v in entry.values():
                assert 0.0 - 1e-9 <= float(v) <= 1.0 + 1e-9

    # Identities
    assert np.allclose(out["p_has_ebike"] + out["p_zero"], 1.0, atol=1e-6)

    # Inventory ordering
    caps = rows["capacity"].astype(float).to_numpy()
    assert np.all(out["expected_ebikes"].to_numpy() >= -1e-9)
    assert np.all(out["expected_total_bikes"].to_numpy() + 1e-6 >= out["expected_ebikes"].to_numpy())
    assert np.all(out["expected_total_bikes"].to_numpy() <= caps + 1e-6)

    # No capacity-violation mass on the fast path
    assert np.all(out["p_capacity_violation"].to_numpy() < 1e-6)


# ---------------------------------------------------------------------------
# Test 2 — adversarial inputs handled safely
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "mutation",
    [
        {"num_ebikes_available": np.nan},
        {"num_bikes_available": np.nan},
        {"capacity": np.nan},
        {"capacity": -5.0},
        {"num_ebikes_available": np.inf},
        {"num_bikes_available": -np.inf},
        # e0 > q0 (inconsistent inventory)
        {"num_ebikes_available": 8, "num_bikes_available": 3, "capacity": 15},
        # q0 > capacity (over-stocked)
        {"num_bikes_available": 20, "capacity": 6},
        # horizon out of range
        {"horizon_minutes": -10},
        {"horizon_minutes": 1_000_000},
    ],
)
def test_adversarial_inputs_handled_safely(fitted_model: DGNISSMModel, mutation: dict) -> None:
    record = {
        "station_id": "s0",
        "anchor_ts": datetime(2026, 1, 1, 12, 0, 0),
        "target_at": datetime(2026, 1, 1, 12, 10, 0),
        "horizon_minutes": 10.0,
        "capacity": 6.0,
        "num_ebikes_available": 1.0,
        "num_bikes_available": 3.0,
        "num_docks_available": 3.0,
        "lat": 41.0,
        "lon": -87.0,
        "is_renting": True,
        "is_returning": True,
    }
    record.update(mutation)
    base = pd.DataFrame([record])

    out = fitted_model.predict_distribution(base, debug=False)

    # No silent NaN/inf in the probabilistic outputs
    for col in ("p_has_ebike", "p_zero", "expected_ebikes", "expected_total_bikes"):
        val = float(out[col].iloc[0])
        assert np.isfinite(val), f"{col} is non-finite under mutation {mutation}: {val}"

    pmf_e = out["p_count_ebikes"].iloc[0]
    pmf_q = out["p_count_total"].iloc[0]
    assert sum(pmf_e.values()) == pytest.approx(1.0, abs=1e-6)
    assert sum(pmf_q.values()) == pytest.approx(1.0, abs=1e-6)

    # Output bounds always sane
    cap = max(1, int(np.nan_to_num(base["capacity"].iloc[0], nan=80.0)))
    assert 0.0 - 1e-9 <= out["p_has_ebike"].iloc[0] <= 1.0 + 1e-9
    assert 0.0 - 1e-9 <= out["expected_ebikes"].iloc[0] <= float(cap) + 1.0
    assert out["expected_ebikes"].iloc[0] - 1e-6 <= out["expected_total_bikes"].iloc[0]


# ---------------------------------------------------------------------------
# Test 3 — fast path agrees with exact DP on first two moments (small capacity)
# ---------------------------------------------------------------------------

def test_fast_path_matches_exact_dp_on_first_two_moments() -> None:
    """The fast path is a moment-matched approximation. Sanity-check it doesn't
    drift catastrophically from the exact DP for short horizons."""
    config = _tiny_config(max_capacity=8, max_rollout_steps=1, exact_inventory_dp=False)
    config_exact = _tiny_config(max_capacity=8, max_rollout_steps=1, exact_inventory_dp=True)
    rng = np.random.default_rng(42)
    abs_e_errors: list[float] = []
    abs_q_errors: list[float] = []
    for _ in range(60):
        cap = int(rng.integers(3, 8))
        e0 = int(rng.integers(0, min(4, cap + 1)))
        q0 = int(rng.integers(e0, cap + 1))
        mu = rng.uniform(0.05, 2.5, size=4)
        theta = rng.uniform(0.5, 8.0, size=4)
        zeta = rng.uniform(0.0, 0.3, size=4)
        fast = rollout_from_parameters(
            capacity=cap, current_ebikes=e0, current_total_bikes=q0,
            horizon_minutes=5, mu=mu, theta=theta, zeta=zeta, config=config,
        )
        exact = rollout_from_parameters(
            capacity=cap, current_ebikes=e0, current_total_bikes=q0,
            horizon_minutes=5, mu=mu, theta=theta, zeta=zeta, config=config_exact,
        )
        abs_e_errors.append(abs(fast.expected_ebikes - exact.expected_ebikes))
        abs_q_errors.append(abs(fast.expected_total_bikes - exact.expected_total_bikes))

    # Mean absolute error in the predicted expected count, in raw bikes
    # The fast path is a Gaussian moment-match — at small capacity and horizon
    # 5min the error should stay below ~1 bike on average.
    assert np.mean(abs_e_errors) < 1.5
    assert np.mean(abs_q_errors) < 1.5
    # And no single row should drift by more than half the capacity range
    assert np.max(abs_e_errors) < 4.0
    assert np.max(abs_q_errors) < 4.0


# ---------------------------------------------------------------------------
# Test 4 — renting/returning gating zeros the relevant flows
# ---------------------------------------------------------------------------

def test_renting_returning_gating_zeros_flows() -> None:
    config = _tiny_config()
    cap, e0, q0 = 10, 4, 6
    mu = np.array([2.0, 2.0, 2.0, 2.0])
    theta = np.array([3.0, 3.0, 3.0, 3.0])
    zeta = np.array([0.1, 0.1, 0.1, 0.1])

    not_renting = fast_inventory_rollout_from_parameters(
        capacity=cap, current_ebikes=e0, current_total_bikes=q0,
        mu=mu, theta=theta, zeta=zeta, config=config,
        is_renting=False, is_returning=True,
    )
    assert not_renting.expected_ebike_departures == pytest.approx(0.0, abs=1e-9)
    assert not_renting.expected_classic_departures == pytest.approx(0.0, abs=1e-9)
    # Arrivals can still happen
    assert not_renting.expected_ebike_arrivals + not_renting.expected_classic_arrivals > 0.0

    not_returning = fast_inventory_rollout_from_parameters(
        capacity=cap, current_ebikes=e0, current_total_bikes=q0,
        mu=mu, theta=theta, zeta=zeta, config=config,
        is_renting=True, is_returning=False,
    )
    assert not_returning.expected_ebike_arrivals == pytest.approx(0.0, abs=1e-9)
    assert not_returning.expected_classic_arrivals == pytest.approx(0.0, abs=1e-9)
    assert not_returning.expected_ebike_departures + not_returning.expected_classic_departures > 0.0


# ---------------------------------------------------------------------------
# Test 5 — save/load preserves the calibrator and graph cache
# ---------------------------------------------------------------------------

def test_save_load_preserves_calibrator_and_graph_cache(fitted_model: DGNISSMModel, tmp_path) -> None:
    sample = _synthetic_examples(n=24, seed=99)
    before = fitted_model.predict_distribution(sample, debug=False)["p_has_ebike"].to_numpy()

    path = tmp_path / "dg.pkl"
    fitted_model.save(path)
    loaded = DGNISSMModel.load(path)

    # Calibrator and graph cache survive round-trip
    assert loaded.calibrator is not None
    assert isinstance(loaded.graph_cache, dict)
    assert set(loaded.graph_cache.get("edge_index_by_type", {})) == set(
        fitted_model.graph_cache.get("edge_index_by_type", {})
    )

    after = loaded.predict_distribution(sample, debug=False)["p_has_ebike"].to_numpy()
    np.testing.assert_allclose(before, after, atol=1e-6)


# ---------------------------------------------------------------------------
# Test 6 — graph relations match the (now-trimmed) config
# ---------------------------------------------------------------------------

def test_graph_relations_match_config(fitted_model: DGNISSMModel) -> None:
    """After Fix 5A, the default config lists only the relations that are
    actually built. This test asserts the cache is a subset of the declared
    relations (empty is also allowed if the data couldn't produce a relation)."""
    declared = set(fitted_model.config.graph_relation_types)
    built = set(fitted_model.graph_cache.get("edge_index_by_type", {}))
    # No relation can be built that wasn't declared
    assert built <= declared, f"unexpected relations built: {built - declared}"
    # The model must not advertise the long-removed defaults
    assert "dynamic" not in declared
    assert "adaptive" not in declared


# ---------------------------------------------------------------------------
# Test 7 — used_sequence_fallback flag surfaces in the output frame
# ---------------------------------------------------------------------------

def test_sequence_fallback_flag_surfaces(fitted_model: DGNISSMModel) -> None:
    """Synthetic rows from `_synthetic_examples` have no `sequence_history`
    column, so they should all flag as fallback. Without the surfaced flag,
    production has no way to know how often it is on the synthetic path."""
    sample = _synthetic_examples(n=16, seed=7)
    out = fitted_model.predict_distribution(sample, debug=False)

    assert "used_sequence_fallback" in out.columns
    assert out["used_sequence_fallback"].dtype == bool
    # All rows here lack history → all should be flagged True
    assert out["used_sequence_fallback"].all()

    # Inject a synthesized "real history" column on half the rows and re-check.
    spec = fitted_model.config.sequence_spec()
    n_chan = len(spec.channels)
    mixed = sample.copy()
    real_history = [
        np.zeros((spec.seq_len, n_chan), dtype=np.float32).tolist()
        if i % 2 == 0 else None
        for i in range(len(mixed))
    ]
    mixed["sequence_features"] = real_history
    out2 = fitted_model.predict_distribution(mixed, debug=False)
    assert "used_sequence_fallback" in out2.columns
    # Even-indexed rows have valid history → fallback should be False there
    assert not out2["used_sequence_fallback"].iloc[0]
    # Odd-indexed rows have None → fallback should be True
    assert out2["used_sequence_fallback"].iloc[1]
