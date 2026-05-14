from __future__ import annotations

import pandas as pd

from divvy.cc_nissm import CCNISSMModel


def test_cc_nissm_predict_distribution_is_constrained() -> None:
    rows = pd.DataFrame(
        [
            {
                "station_id": "s1",
                "horizon_minutes": 10,
                "capacity": 10,
                "capacity_clipped": 10,
                "num_ebikes_available": 0,
                "num_bikes_available": 5,
                "num_docks_available": 5,
                "station_same_hour_rate": 0.4,
                "station_neighbor_same_hour_rate": 0.5,
                "trip_arrivals_same_hour_10m": 1.0,
                "trip_departures_same_hour_10m": 0.5,
            }
        ]
    )

    pred = CCNISSMModel().fit(rows.assign(has_ebike=1)).predict_distribution(rows)

    assert pred["p_has_ebike"].between(0, 1).all()
    assert pred.iloc[0]["p_capacity_violation"] == 0.0
    assert sum(pred.iloc[0]["p_count_ebikes"].values()) == 1.0
