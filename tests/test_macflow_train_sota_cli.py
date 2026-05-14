"""CLI plumbing tests for the macflow-nissm-lite subcommand."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from divvy import train_sota


def _parser():
    import argparse

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    train_sota._add_shared_args(sub.add_parser("macflow-nissm-lite"))
    train_sota._add_shared_args(sub.add_parser("dg-nissm"))
    return parser


def test_macflow_subcommand_parses_with_default_args():
    parser = _parser()
    args = parser.parse_args([
        "macflow-nissm-lite",
        "--history-hours", "24",
        "--valid-hours", "4",
        "--anchor-every-min", "10",
        "--horizons", "5", "10", "15", "20",
        "--device", "auto",
        "--max-examples", "50000",
        "--partition-mode", "full",
    ])
    assert args.command == "macflow-nissm-lite"
    assert args.partition_mode == "full"
    assert args.device == "auto"
    assert args.horizons == [5, 10, 15, 20]
    assert args.max_examples == 50_000


def test_dg_nissm_subcommand_still_works():
    parser = _parser()
    args = parser.parse_args([
        "dg-nissm",
        "--device", "auto",
        "--max-examples", "200000",
    ])
    assert args.command == "dg-nissm"


def test_command_to_key_maps_correctly():
    assert train_sota._single_command_to_key("macflow-nissm-lite") == "macflow_nissm_lite"
    assert train_sota._single_command_to_key("dg-nissm") == "dg_nissm"
    assert train_sota._single_command_to_key("cc-nissm") == "cc_nissm"


def test_partition_mode_choices_are_validated():
    parser = _parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "macflow-nissm-lite",
            "--partition-mode", "invalid_mode",
        ])


def test_partition_mode_options_all_accepted():
    parser = _parser()
    for mode in ("off", "id_only", "id_plus_role", "full", "random", "spatial"):
        args = parser.parse_args([
            "macflow-nissm-lite",
            "--partition-mode", mode,
        ])
        assert args.partition_mode == mode


def test_train_single_dispatches_to_macflow_handler():
    import argparse

    args = argparse.Namespace(
        command="macflow-nissm-lite",
        history_hours=24,
        valid_hours=4,
        anchor_every_min=10,
        horizons=[5, 10],
        max_source_rows=1000,
        device="cpu",
        register=False,
        activate=None,
        activate_best_sota=False,
        coexist_live=False,
        time_budget_hours=8.0,
        strict=False,
        epochs=1,
        batch_size=128,
        max_examples=1000,
        hidden_dim=64,
        station_embedding_dim=8,
        seq_len=24,
        seq_step_minutes=2,
        top_k=16,
        lr=1e-3,
        weight_decay=1e-4,
        seed=42,
        no_sequence=False,
        no_graph=False,
        calibrate=False,
        benchmark_runtime=False,
        stg_max_examples=None,
        stg_epochs=None,
        stg_batch_size=None,
        partition_mode="off",
    )

    sentinel = {"called": False}

    def fake_train(train, valid, horizons, args=None):
        sentinel["called"] = True
        sentinel["args"] = args
        return {
            "model_key": "macflow_nissm_lite",
            "status": "trained",
            "model_obj": object(),
            "model_family": "macflow_nissm_lite",
            "model_version": "macflow-nissm-lite-v1",
            "feature_columns": [],
            "is_primary_eligible": True,
            "metrics": {},
        }

    import pandas as pd

    def fake_build(_args):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}

    with patch.object(train_sota, "_build_examples", fake_build), patch.object(
        train_sota, "_train_macflow_nissm_lite", fake_train
    ):
        result = train_sota.train_single(args)
    assert sentinel["called"] is True
    assert result["models"][0]["model_key"] == "macflow_nissm_lite"
