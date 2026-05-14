# Divvy Observer

A long-running observatory for Chicago Divvy bike-share stations. The collector polls the GBFS feed every 60 seconds and owns DuckDB writes. The prediction API and Streamlit dashboard run separately, read the latest state, and queue forecast logs for the collector to persist.

Self-contained Python project; only requires [uv](https://github.com/astral-sh/uv) (`brew install uv`).

## Quick start

Start the full local system in one terminal:

```bash
./run.sh
```

That single command starts the collector, automation supervisor, API, and dashboard, and prints the local URLs. Leave that terminal open while using the dashboard.

Check or stop the system:

```bash
uv run divvy status
uv run divvy health
uv run divvy launchd-status
uv run divvy stop
```

The local prediction surface defaults to API http://127.0.0.1:8000 and dashboard http://127.0.0.1:8501.

For set-and-forget launchd services:

```bash
./run.sh --launchd
```

The older split foreground scripts are still available for manual development:

```bash
./run_collector.sh
./run_prediction.sh
```

`run_prediction.sh` binds services to `0.0.0.0`, so other devices on your LAN can open the dashboard at `http://<computer-ip>:8503`. The app's browser-location button uses `navigator.geolocation`, which mobile browsers only allow from secure origins. For GPS on your phone, open the app from an HTTPS URL, for example:

```bash
# keep ./run_prediction.sh running, then in another shell expose Streamlit through HTTPS
cloudflared tunnel --url http://localhost:8503
# or:
ngrok http 8503
```

Open the HTTPS tunnel URL on the phone, tap the location button, and allow location access. If you open the LAN HTTP URL directly, the dashboard still works, but mobile browser GPS usually will not prompt.

If you'd rather drive the pieces yourself:

```bash
uv sync                                 # creates .venv, installs deps
uv run python -m divvy.collector        # foreground collector; Ctrl+C to stop
# in another shell:
uv run uvicorn divvy.api:app --host 0.0.0.0 --port 8001
# in another shell:
uv run streamlit run src/divvy/dashboard.py --server.address=0.0.0.0 --server.port=8503
```

Divvy currently exposes ~2,000 stations in the GBFS feed.

## Prediction API

Start the FastAPI service separately from the collector:

```bash
uv run uvicorn divvy.api:app --host 0.0.0.0 --port 8001
```

Useful endpoints:

- `GET /health`
- `POST /api/v1/recommendations` with `{"lat": 41.88, "lon": -87.63}`
- `GET /api/v1/model/performance?window_hours=24`
- `POST /api/v1/model/backtest` with `{"history_hours": 168}`

Recommendation requests score nearby stations server-side, queue forecast logs in `data/forecast_queue/pending`, and return the closest docked eBike, closest live free-floating eBike, best likely docked station in 5-10 minutes, and ranked reliable alternatives. The collector drains that queue into DuckDB and resolves outcomes. It also publishes `data/divvy_readonly.duckdb`, and the API/dashboard read that replica so they do not compete for the main DuckDB writer lock. The original calibrated logistic model remains the baseline. Experimental models run beside it with trip-flow, route, weather, seasonality, holiday, station-neighborhood, and live-state features:

- `random_forest` — calibrated flow/weather random forest.
- `gradient_boosting` — calibrated flow/weather gradient boosting.
- `inventory_world` — active recommender; a constrained rollout that predicts departures, arrivals, accepted inbound bikes, and future docked eBike inventory under capacity/dock limits.
- `stg_ncde` — sidecar STG-NCDE graph controlled-differential-equation model using `torchcde` on larger history sets; for tiny fixtures or unavailable torch dependencies it falls back to the explicit Euler state evolution plus logistic calibration.

The STG-NCDE sidecar is bounded by default so the API stays usable: `DIVVY_STG_NCDE_MIN_EXAMPLES=2000`, `DIVVY_STG_NCDE_MAX_EXAMPLES=8000`, `DIVVY_STG_NCDE_EPOCHS=4`, and `DIVVY_DISABLE_TORCH_STG_NCDE=1` can be used to force the deterministic fallback. `DIVVY_STG_NCDE_DEVICE=auto` prefers Apple MPS on Apple Silicon, then CUDA, then CPU; set it to `cpu` for the most conservative path.

Rolling model rankings use `rank_loss = Brier score + 0.05 × log loss`, where lower is better.

Optional richer features:

```bash
# Divvy monthly historical trip files: station-to-station arrivals, departures, and route durations
uv run python -m divvy.tripdata sync --months 6

# Chicago hourly weather history and near-future forecast features
uv run python -m divvy.weather sync-recent --days 365
uv run python -m divvy.weather sync-forecast --days 3
```

The prediction service still works without these optional tables, but the experimental models will fall back to neutral defaults until the data is loaded.

For continuous self-evaluation independent of app traffic:

```bash
uv run python -m divvy.model_eval loop --window-hours 24 --interval-seconds 300
```

That command is still available for offline experiments, but do not run it continuously beside the collector because it also wants the DuckDB writer lock. In normal rollout, the collector drains API forecast logs, resolves due outcomes, and snapshots rolling metrics itself. Recommendation queries are logged by rounded lat/lon place key so model performance can be compared overall and around places you actually search.

## Legacy Collector-Only LaunchAgent

The preferred macOS service path is now `./run.sh` or `uv run divvy start`, which manages collector, automation, API, and dashboard together. The older collector-only plist flow remains here only for manual collector experiments.

```bash
PROJECT_DIR="$(pwd)"
UV_PATH="$(command -v uv)"
sed -e "s|{{PROJECT_DIR}}|${PROJECT_DIR}|g" \
    -e "s|{{UV_PATH}}|${UV_PATH}|g" \
    launchd/net.thoughtbison.divvy-collector.plist \
    > ~/Library/LaunchAgents/net.thoughtbison.divvy-collector.plist

launchctl load ~/Library/LaunchAgents/net.thoughtbison.divvy-collector.plist
launchctl list | grep divvy-collector    # confirm it's running
tail -f data/collector.log               # watch the ticks
```

To stop / uninstall:

```bash
launchctl unload ~/Library/LaunchAgents/net.thoughtbison.divvy-collector.plist
rm ~/Library/LaunchAgents/net.thoughtbison.divvy-collector.plist
```

**External-drive caveat:** the project lives on `/Volumes/GameDrive`. If the drive is unmounted, the collector will fail with I/O errors and `KeepAlive` will keep restarting it; data collection pauses until the drive is back. To pin data to your internal disk, point `DIVVY_DB_PATH` at, say, `~/.divvy-observer/divvy.duckdb` and `DIVVY_FORECAST_QUEUE_DIR` at `~/.divvy-observer/forecast_queue` in the plist's `EnvironmentVariables` block.

## Configuration (env vars)

| Variable                  | Default                             |
|---------------------------|-------------------------------------|
| `DIVVY_POLL_INTERVAL`     | `60` (seconds)                      |
| `DIVVY_INFO_REFRESH`      | `21600` (6 h — station metadata)    |
| `DIVVY_DATA_DIR`          | `<project>/data`                    |
| `DIVVY_DB_PATH`           | `<DATA_DIR>/divvy.duckdb`           |
| `DIVVY_READ_DB_PATH`      | `<DATA_DIR>/divvy_readonly.duckdb`  |
| `DIVVY_ENABLE_READ_REPLICA` | `1`                               |
| `DIVVY_READ_REPLICA_REFRESH_SECONDS` | `60`                    |
| `DIVVY_FORECAST_QUEUE_DIR` | `<DATA_DIR>/forecast_queue`        |
| `DIVVY_FORECAST_QUEUE_DRAIN_LIMIT` | `200` queued requests per tick |
| `DIVVY_MODEL_OUTCOME_RESOLVE_SECONDS` | `60`                    |
| `DIVVY_MODEL_METRICS_SNAPSHOT_SECONDS` | `600`                 |
| `DIVVY_SELF_EVAL_INTERVAL_SECONDS` | `300` (0 disables; collector emits forecasts for a rotating shard of stations on this cadence so model selection keeps learning without API traffic) |
| `DIVVY_SELF_EVAL_STATION_SAMPLE` | `25` (stations scored per self-eval tick) |

## Data model

Core tables in a single DuckDB file:

- **`stations`** — one row per station; metadata refreshed periodically. PK `station_id`.
- **`station_status`** — time series, one row per *state change*. PK `(station_id, last_reported)` deduplicates: when a station hasn't updated since the last poll, the `INSERT OR IGNORE` is a no-op.
- **`free_bike_status`** — deduped free-floating eBike position events from GBFS.
- **`divvy_trips`**, **`station_trip_flows`**, **`station_trip_routes`** — optional monthly trip history, station arrival/departure pressure, and origin-destination travel-time aggregates.
- **`weather_hourly`** — optional hourly Chicago weather observations/forecasts used by experimental models.
- **`model_forecasts`**, **`model_outcomes`**, **`model_metrics`** — forecast logs, delayed outcome resolution, and rolling evaluation snapshots.

`num_classic_bikes = num_bikes_available - num_ebikes_available` is computed on read.

## Health checks

```sql
-- minutes covered in the last 24h (should be near 1440 if the poller runs uninterrupted)
SELECT COUNT(DISTINCT DATE_TRUNC('minute', fetched_at))
FROM station_status
WHERE fetched_at > now() - INTERVAL 1 DAY;

-- busiest stations (highest row count = most state churn)
SELECT s.name, COUNT(*) AS rows
FROM station_status ss JOIN stations s USING (station_id)
GROUP BY s.name ORDER BY rows DESC LIMIT 10;
```

`uv run duckdb data/divvy.duckdb` opens an interactive SQL shell if you have the `duckdb` CLI installed (`brew install duckdb`).
