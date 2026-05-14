from __future__ import annotations

import pandas as pd

from divvy import config, forecast_queue, predictor


def test_forecast_queue_v2_counts_all_models_and_horizons(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config, "FORECAST_QUEUE_DIR", tmp_path / "queue")
    row = {
        "station_id": "s1",
        "active_model_key": "cc_nissm",
        "active_model_source": "default_cc_nissm",
    }
    for model_key in predictor.MODEL_KEYS:
        for horizon in predictor.HORIZONS:
            row[f"p_has_ebike_{horizon}m_{model_key}"] = 0.5
    scored = pd.DataFrame([row])

    result = forecast_queue.enqueue_recommendation_forecasts(
        scored,
        request_id="r1",
        source="test",
        user_lat=41.0,
        user_lon=-87.0,
        near_radius_km=0.5,
        search_radius_km=1.5,
        query_label=None,
        query_place_key="41.000,-87.000",
        candidate_count=1,
        best_station_id="s1",
        recommended_station_ids=["s1"],
        recommended_station_ids_by_model={"cc_nissm": ["s1"]},
    )

    assert result["queued_forecast_rows"] == len(predictor.MODEL_KEYS) * len(predictor.HORIZONS)
