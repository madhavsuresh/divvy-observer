# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Long-running observatory for Chicago Divvy bike-share. A collector polls the GBFS feed every 60s and owns all DuckDB writes; a FastAPI prediction service and a Streamlit dashboard read a 60s-refreshed replica and queue forecasts for the collector to resolve. Roughly ten coexisting models score `p_has_ebike` / `p_zero` etc. across eight horizons (5–90 min) and are ranked by rolling Brier + 0.05·log-loss. Python only; orchestrated by `uv`.

The README (`README.md`) is the canonical source for env vars, the data model, and SQL health checks — read it for those. This file documents what spans multiple modules.

## Commands

```bash
uv sync                                  # install deps into .venv
./run.sh                                 # collector + automation + API (:8000) + dashboard (:8501), foreground
./run.sh stop                            # kill workers + unload any divvy LaunchAgent plists
./run.sh status                          # show running services, port bindings, plist files
uv run divvy {status,health,stop,launchd-status}   # LaunchAgent path is opt-in via `uv run divvy start`
uv run pytest                            # full suite
uv run pytest -m 'not slow'              # skip the slow marker
uv run pytest tests/test_predictor.py::test_name   # single test
uv run python -m divvy.tripdata sync --months 6    # optional trip-history enrichment
uv run python -m divvy.weather sync-recent --days 365
```

Split foreground scripts (`./run_collector.sh`, `./run_prediction.sh`) bind to `0.0.0.0` and use ports 8001 / 8503 so a phone on the LAN can hit the dashboard — different from `./run.sh`'s 8000/8501.

## Architecture

Four layers, all in `src/divvy/`:

1. **Collection** — `poller.py` (main loop), `collector.py` (entry). Polls GBFS, writes `station_status` / `free_bike_status`, and on each tick also drains the forecast queue, resolves due outcomes, snapshots metrics, and refreshes the read replica.
2. **State** — `db.py` (schema, connections), `live_cache.py`, `service_state.py`. Single DuckDB writer = the collector; everyone else reads `data/divvy_readonly.duckdb`. Don't add a second writer.
3. **Prediction** — `predictor.py` is the shared feature builder for *all* models (`_history_rates_for_candidates`, `_live_neighbor_features`, etc.). Model modules call into it; they don't re-implement features. `model_registry.py` / `model_selection.py` rank and promote.
4. **Serving** — `api.py` (FastAPI), `dashboard.py` (Streamlit), `recommendations.py` (scoring logic for API responses).

### The forecast queue pattern (important)

API and dashboard never write to DuckDB. When they make a prediction, they emit a JSON log to `data/forecast_queue/pending/`. The collector drains up to `DIVVY_FORECAST_QUEUE_DRAIN_LIMIT` (200) per tick, persists them as `model_forecasts`, resolves their outcomes once enough time has passed, and rolls metrics. If you add a new prediction call site, follow this — never open the main DB for writes from API/dashboard code.

### Adding a new model

1. New module `src/divvy/<name>.py` exporting `predict(conn, candidates_df, now, horizons) -> dict`.
2. Reuse feature builders from `predictor.py`; don't redo flow/weather/graph features.
3. Register in `predictor.MODEL_SPECS` with a stable label + version.
4. Return keys shaped like `p_has_ebike_{horizon}m_<label>`, `p_zero_{horizon}m_<label>`.
5. `model_selection.py` ranks it automatically once it logs forecasts.

If torch isn't available, fall back gracefully — see `disabled_predictor.py` for the pattern. STG-NCDE is governed by `DIVVY_STG_NCDE_*` env vars and falls back to a deterministic Euler/logistic path.

## Gotchas

- **External drive**: data lives on `/Volumes/GameDrive`. If unmounted, the collector errors and LaunchAgent restarts it forever. To pin to internal disk for dev, set `DIVVY_DB_PATH` and `DIVVY_FORECAST_QUEUE_DIR`.
- **Don't run `divvy.model_eval loop` next to the collector** — it wants the writer lock. Self-evaluation already happens in the poller via `DIVVY_SELF_EVAL_INTERVAL_SECONDS`.
- **Read replica lag**: API/dashboard see state up to 60s stale (`DIVVY_READ_REPLICA_REFRESH_SECONDS`). Tests that need fresh data should connect with `db.connect(read_only=False)` or bypass.
- **`station_status` is event-sourced**, PK `(station_id, last_reported)`. An unchanged station doesn't produce a row each tick — `INSERT OR IGNORE` is a no-op. `num_classic_bikes` is computed (`num_bikes_available − num_ebikes_available`), not stored.
- **Open TODO: `p_has_open_dock`** — the parking dual of `p_has_ebike` (probability ≥1 open dock for a return trip) is missing. Search for the TODO in the cdg/inventory modules before adding new return-side features so you don't fork the work.

## Entry points to read first

- `src/divvy/poller.py` — the collector loop, where all the periodic jobs are wired together
- `src/divvy/predictor.py` — the shared feature surface every model uses
- `src/divvy/api.py` + `src/divvy/recommendations.py` — request → score → queue
- `src/divvy/automation.py` — `WRITE_JOBS` dict + the `divvy` CLI
