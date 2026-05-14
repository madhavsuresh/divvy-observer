from __future__ import annotations

from datetime import datetime, timedelta, timezone

import duckdb
import pandas as pd

from divvy import db, live_inflight


def test_live_inflight_arrivals_from_station_deltas() -> None:
    conn = duckdb.connect(":memory:")
    db.init_schema(conn)
    base = datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0) - timedelta(minutes=5)
    conn.executemany(
        "INSERT INTO stations VALUES (?, NULL, ?, ?, ?, ?, 15, 'classic', ?, ?)",
        [
            ("origin", "O", "Origin", 41.0, -87.0, base, base),
            ("dest", "D", "Dest", 41.001, -87.001, base, base),
        ],
    )
    conn.executemany(
        "INSERT INTO station_status VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            ("origin", base, base, 6, 3, 0, 9, 0, True, True, True),
            ("origin", base + timedelta(minutes=1), base + timedelta(minutes=1), 4, 1, 0, 11, 0, True, True, True),
        ],
    )
    local = pd.Timestamp(base, tz="UTC").tz_convert("America/Chicago")
    conn.execute(
        """
        INSERT INTO station_trip_routes
        VALUES ('origin', 'dest', ?, ?, 10, 8, 8.0, 8.0)
        """,
        [int(local.hour), int(local.dayofweek)],
    )

    result = live_inflight.update_live_inflight_arrivals(conn, lookback_minutes=10)
    features = live_inflight.get_live_inflight_features(conn, ["dest"], base + timedelta(minutes=2))

    assert result["rows_inserted"] > 0
    assert features.iloc[0]["live_inflight_ebike_due_10m"] > 0
    conn.close()
