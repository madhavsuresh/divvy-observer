from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

from divvy.dg_nissm_features import (
    SequenceSpec,
    add_shifted_empirical_priors,
    apply_train_valid_shifted_priors,
    build_sequence_from_status,
)


def test_shifted_station_prior_excludes_current_and_future_labels() -> None:
    base = datetime(2026, 1, 1, 10, 0, 0)
    examples = pd.DataFrame(
        [
            {"station_id": "s1", "anchor_ts": base, "horizon_minutes": 5, "has_ebike": 0, "future_ebikes": 0, "future_total_bikes": 2},
            {"station_id": "s1", "anchor_ts": base + timedelta(minutes=5), "horizon_minutes": 5, "has_ebike": 0, "future_ebikes": 0, "future_total_bikes": 2},
            {"station_id": "s1", "anchor_ts": base + timedelta(minutes=10), "horizon_minutes": 5, "has_ebike": 1, "future_ebikes": 3, "future_total_bikes": 5},
        ]
    )
    enriched = add_shifted_empirical_priors(examples, alpha=1.0, global_default=0.35)

    assert enriched.iloc[0]["station_hour_dow_has_ebike_rate_shifted"] == pytest.approx(0.35, abs=0.05)
    assert enriched.iloc[1]["station_hour_dow_has_ebike_rate_shifted"] < 0.35
    assert enriched.iloc[2]["station_hour_dow_has_ebike_rate_shifted"] < 0.35

    mutated = examples.copy()
    mutated["future_only_signal"] = [999, 999, -999]
    mutated.loc[2, "future_ebikes"] = 99
    mutated.loc[2, "future_total_bikes"] = 99
    mutated_enriched = add_shifted_empirical_priors(mutated, alpha=1.0, global_default=0.35)
    assert mutated_enriched.iloc[1]["station_hour_dow_e_mean_shifted"] == pytest.approx(
        enriched.iloc[1]["station_hour_dow_e_mean_shifted"]
    )


def test_validation_priors_start_from_training_history_only() -> None:
    base = datetime(2026, 1, 1, 10, 0, 0)
    train = pd.DataFrame(
        [
            {"station_id": "s1", "anchor_ts": base, "horizon_minutes": 5, "has_ebike": 0, "future_ebikes": 0, "future_total_bikes": 2},
            {"station_id": "s1", "anchor_ts": base + timedelta(minutes=5), "horizon_minutes": 5, "has_ebike": 0, "future_ebikes": 0, "future_total_bikes": 2},
        ]
    )
    valid = pd.DataFrame(
        [
            {"station_id": "s1", "anchor_ts": base + timedelta(minutes=10), "horizon_minutes": 5, "has_ebike": 1, "future_ebikes": 5, "future_total_bikes": 5},
            {"station_id": "s1", "anchor_ts": base + timedelta(minutes=15), "horizon_minutes": 5, "has_ebike": 1, "future_ebikes": 5, "future_total_bikes": 5},
        ]
    )
    _train_out, valid_out = apply_train_valid_shifted_priors(train, valid, alpha=1.0, global_default=0.35)

    first_valid_rate = valid_out.sort_values("anchor_ts").iloc[0]["station_hour_dow_has_ebike_rate_shifted"]
    second_valid_rate = valid_out.sort_values("anchor_ts").iloc[1]["station_hour_dow_has_ebike_rate_shifted"]
    assert first_valid_rate < 0.35
    assert second_valid_rate > first_valid_rate


def test_sequence_builder_uses_only_observations_at_or_before_anchor() -> None:
    base = datetime(2026, 1, 1, 10, 0, 0)
    status = pd.DataFrame(
        [
            {
                "station_id": "s1",
                "observation_ts": base,
                "num_ebikes_available": 1,
                "num_bikes_available": 3,
                "num_docks_available": 7,
                "capacity": 10,
                "is_renting": True,
                "is_returning": True,
                "status_age_minutes": 0,
            },
            {
                "station_id": "s1",
                "observation_ts": base + timedelta(minutes=10),
                "num_ebikes_available": 9,
                "num_bikes_available": 9,
                "num_docks_available": 1,
                "capacity": 10,
                "is_renting": True,
                "is_returning": True,
                "status_age_minutes": 0,
            },
        ]
    )
    seq = build_sequence_from_status(status, base + timedelta(minutes=5), spec=SequenceSpec(seq_len=3, seq_step_minutes=2))

    assert seq[-1][0] == pytest.approx(0.1)
    assert all(abs(row[0] - 0.9) > 1e-6 for row in seq)
