from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from divvy.cdg_nmip import CDG_DEBUG_COLUMNS, CDGNMIPConfig
from divvy.dg_nissm import DGNISSMModel


def _synthetic_examples(n: int = 120) -> pd.DataFrame:
    base = datetime(2026, 1, 1, 12, 0, 0)
    rows = []
    for i in range(n):
        e0 = i % 4
        q0 = min(6, 3 + e0)
        future_e = 0 if i % 2 == 0 else min(4, e0 + 1)
        future_q = min(6, max(future_e, q0 + (1 if i % 3 else -1)))
        rows.append(
            {
                "station_id": f"s{i % 6}",
                "anchor_ts": base + timedelta(minutes=i),
                "target_at": base + timedelta(minutes=i + (5 if i % 2 else 10)),
                "horizon_minutes": 5 if i % 2 else 10,
                "capacity": 6,
                "num_ebikes_available": e0,
                "num_bikes_available": q0,
                "num_docks_available": 6 - q0,
                "future_ebikes": future_e,
                "future_total_bikes": future_q,
                "has_ebike": int(future_e >= 1),
                "lat": 41.0 + (i % 6) * 0.001,
                "lon": -87.0,
                "is_renting": True,
                "is_returning": True,
            }
        )
    return pd.DataFrame(rows)


def _tiny_config() -> CDGNMIPConfig:
    return CDGNMIPConfig(
        min_train_examples=60,
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


def test_dg_nissm_predicts_inventory_consistent_pmfs_and_debug_columns() -> None:
    examples = _synthetic_examples()
    model = DGNISSMModel(_tiny_config()).fit(examples.iloc[:100], examples.iloc[100:])

    assert model.trained
    assert model.method == "dg_nissm_cdg_nmip_trained_v1"
    assert "bootstrap" not in model.method
    assert "fallback" not in model.method
    assert model.model_warning is None

    out = model.predict_distribution(examples.iloc[:12], debug=True)
    required = {
        "p_has_ebike",
        "p_zero",
        "p_appears",
        "p_survives",
        "expected_ebikes",
        "expected_total_bikes",
        "p_count_ebikes",
        "p_count_total",
        "p_count_ebikes_json",
        "p_count_total_json",
    }
    assert required.issubset(out.columns)
    assert set(CDG_DEBUG_COLUMNS).issubset(out.columns)
    assert np.allclose(out["p_has_ebike"] + out["p_zero"], 1.0)
    for _, row in out.iterrows():
        assert sum(row["p_count_ebikes"].values()) == pytest.approx(1.0)
        assert sum(row["p_count_total"].values()) == pytest.approx(1.0)
        assert 0.0 <= row["expected_ebikes"] <= row["expected_total_bikes"] <= 6.0
    assert out.loc[examples.iloc[:12]["num_ebikes_available"].to_numpy() == 0, "p_appears"].notna().all()
    assert out.loc[examples.iloc[:12]["num_ebikes_available"].to_numpy() > 0, "p_appears"].isna().all()
    assert out.loc[examples.iloc[:12]["num_ebikes_available"].to_numpy() > 0, "p_survives"].notna().all()


def test_dg_nissm_save_load_preserves_predictions(tmp_path) -> None:
    examples = _synthetic_examples()
    model = DGNISSMModel(_tiny_config()).fit(examples.iloc[:100], examples.iloc[100:])
    before = model.predict_distribution(examples.iloc[:8])["p_has_ebike"].to_numpy()

    path = tmp_path / "dg.pkl"
    model.save(path)
    loaded = DGNISSMModel.load(path)
    after = loaded.predict_distribution(examples.iloc[:8])["p_has_ebike"].to_numpy()

    assert loaded.method == "dg_nissm_cdg_nmip_trained_v1"
    assert loaded.model_warning is None
    assert np.allclose(before, after)


def test_dg_nissm_fit_skips_tiny_data_cleanly() -> None:
    config = _tiny_config()
    config.min_train_examples = 1_000
    model = DGNISSMModel(config).fit(_synthetic_examples(20), None)

    assert not model.trained
    assert model.method == "dg_nissm_skipped_insufficient_data"
    assert model.model_warning
