from __future__ import annotations

from divvy import train_sota


def test_best_trained_sota_key_uses_registered_primary_eligible_metric() -> None:
    results = [
        {
            "model_key": "cc_nissm",
            "status": "trained",
            "artifact_id": "cc-1",
            "is_primary_eligible": True,
            "metrics": {"decision_rank_loss": 0.20},
        },
        {
            "model_key": "dg_nissm",
            "status": "trained",
            "artifact_id": "dg-1",
            "is_primary_eligible": True,
            "metrics": {"decision_rank_loss": 0.10},
        },
        {
            "model_key": "stg_ncde_inventory",
            "status": "trained",
            "artifact_id": None,
            "is_primary_eligible": True,
            "metrics": {"decision_rank_loss": 0.01},
        },
        {
            "model_key": "tft_inventory",
            "status": "trained",
            "artifact_id": "tft-1",
            "is_primary_eligible": False,
            "metrics": {"decision_rank_loss": 0.01},
        },
    ]

    assert train_sota._best_trained_sota_key(results) == "dg_nissm"


def test_run_nightly_activates_best_sota_not_cc_default(monkeypatch) -> None:
    captured = {}

    def fake_train_all(args):
        captured["activate"] = args.activate
        captured["activate_best_sota"] = args.activate_best_sota
        captured["benchmark_runtime"] = args.benchmark_runtime
        return {"status": "ok"}

    monkeypatch.setattr(train_sota, "train_all", fake_train_all)

    result = train_sota.run_nightly(train_days=1, valid_days=1)

    assert result == {"status": "ok"}
    assert captured["activate"] is None
    assert captured["activate_best_sota"] is True
    assert captured["benchmark_runtime"] is True
