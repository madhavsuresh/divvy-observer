from __future__ import annotations

import time
import shutil
from contextlib import contextmanager

import duckdb

from . import config

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS stations (
  station_id     TEXT PRIMARY KEY,
  legacy_id      TEXT,
  short_name     TEXT,
  name           TEXT,
  lat            DOUBLE,
  lon            DOUBLE,
  capacity       INTEGER,
  station_type   TEXT,
  first_seen_at  TIMESTAMP,
  last_seen_at   TIMESTAMP
);

CREATE TABLE IF NOT EXISTS station_status (
  station_id            TEXT      NOT NULL,
  last_reported         TIMESTAMP NOT NULL,
  fetched_at            TIMESTAMP NOT NULL,
  num_bikes_available   INTEGER,
  num_ebikes_available  INTEGER,
  num_bikes_disabled    INTEGER,
  num_docks_available   INTEGER,
  num_docks_disabled    INTEGER,
  is_installed          BOOLEAN,
  is_renting            BOOLEAN,
  is_returning          BOOLEAN,
  PRIMARY KEY (station_id, last_reported)
);

CREATE INDEX IF NOT EXISTS idx_status_time ON station_status(last_reported);

CREATE TABLE IF NOT EXISTS free_bike_status (
  bike_id        TEXT      NOT NULL,
  fetched_at     TIMESTAMP NOT NULL,
  name           TEXT,
  lat            DOUBLE,
  lon            DOUBLE,
  is_reserved    BOOLEAN,
  is_disabled    BOOLEAN,
  PRIMARY KEY (bike_id, fetched_at)
);

CREATE INDEX IF NOT EXISTS idx_free_bike_fetched ON free_bike_status(fetched_at);
CREATE INDEX IF NOT EXISTS idx_free_bike_bike_id ON free_bike_status(bike_id);

ALTER TABLE free_bike_status ADD COLUMN IF NOT EXISTS tile_id TEXT;
CREATE INDEX IF NOT EXISTS idx_free_bike_tile_time ON free_bike_status(tile_id, fetched_at);

CREATE TABLE IF NOT EXISTS divvy_trips (
  ride_id             TEXT PRIMARY KEY,
  rideable_type       TEXT,
  started_at          TIMESTAMP NOT NULL,
  ended_at            TIMESTAMP NOT NULL,
  start_station_id    TEXT,
  start_station_name  TEXT,
  end_station_id      TEXT,
  end_station_name    TEXT,
  start_lat           DOUBLE,
  start_lon           DOUBLE,
  end_lat             DOUBLE,
  end_lon             DOUBLE,
  member_casual       TEXT,
  duration_minutes    DOUBLE
);

CREATE INDEX IF NOT EXISTS idx_divvy_trips_started ON divvy_trips(started_at);
CREATE INDEX IF NOT EXISTS idx_divvy_trips_ended ON divvy_trips(ended_at);
CREATE INDEX IF NOT EXISTS idx_divvy_trips_start_station ON divvy_trips(start_station_id);
CREATE INDEX IF NOT EXISTS idx_divvy_trips_end_station ON divvy_trips(end_station_id);

CREATE TABLE IF NOT EXISTS station_trip_flows (
  station_id          TEXT NOT NULL,
  bucket_start        TIMESTAMP NOT NULL,
  departures          INTEGER DEFAULT 0,
  arrivals            INTEGER DEFAULT 0,
  ebike_departures    INTEGER DEFAULT 0,
  ebike_arrivals      INTEGER DEFAULT 0,
  PRIMARY KEY (station_id, bucket_start)
);

CREATE INDEX IF NOT EXISTS idx_station_trip_flows_station ON station_trip_flows(station_id);
CREATE INDEX IF NOT EXISTS idx_station_trip_flows_bucket ON station_trip_flows(bucket_start);

CREATE TABLE IF NOT EXISTS station_inferred_flows (
  station_id          TEXT NOT NULL,
  bucket_start        TIMESTAMP NOT NULL,
  departures          INTEGER DEFAULT 0,
  arrivals            INTEGER DEFAULT 0,
  ebike_departures    INTEGER DEFAULT 0,
  ebike_arrivals      INTEGER DEFAULT 0,
  classic_departures  INTEGER DEFAULT 0,
  classic_arrivals    INTEGER DEFAULT 0,
  observations        INTEGER DEFAULT 0,
  rebalancing_events  INTEGER DEFAULT 0,
  computed_at         TIMESTAMP NOT NULL,
  PRIMARY KEY (station_id, bucket_start)
);

CREATE INDEX IF NOT EXISTS idx_station_inferred_flows_station ON station_inferred_flows(station_id);
CREATE INDEX IF NOT EXISTS idx_station_inferred_flows_bucket ON station_inferred_flows(bucket_start);

CREATE TABLE IF NOT EXISTS flow_processing_state (
  key         TEXT PRIMARY KEY,
  value       TEXT,
  updated_at  TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS station_trip_routes (
  start_station_id          TEXT NOT NULL,
  end_station_id            TEXT NOT NULL,
  local_hour                INTEGER NOT NULL,
  dow                       INTEGER NOT NULL,
  trips                     INTEGER NOT NULL,
  ebike_trips               INTEGER NOT NULL,
  avg_duration_minutes      DOUBLE,
  median_duration_minutes   DOUBLE,
  PRIMARY KEY (start_station_id, end_station_id, local_hour, dow)
);

CREATE INDEX IF NOT EXISTS idx_station_trip_routes_end ON station_trip_routes(end_station_id);
CREATE INDEX IF NOT EXISTS idx_station_trip_routes_start ON station_trip_routes(start_station_id);

CREATE TABLE IF NOT EXISTS weather_hourly (
  observed_at               TIMESTAMP PRIMARY KEY,
  source                    TEXT DEFAULT 'open-meteo',
  temperature_2m            DOUBLE,
  relative_humidity_2m      DOUBLE,
  apparent_temperature      DOUBLE,
  precipitation             DOUBLE,
  rain                      DOUBLE,
  snowfall                  DOUBLE,
  snow_depth                DOUBLE,
  cloud_cover               DOUBLE,
  wind_speed_10m            DOUBLE,
  wind_gusts_10m            DOUBLE,
  weather_code              INTEGER,
  fetched_at                TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_weather_hourly_observed ON weather_hourly(observed_at);

CREATE TABLE IF NOT EXISTS collector_ticks (
  tick_id                         TEXT PRIMARY KEY,
  ticked_at                       TIMESTAMP NOT NULL,
  station_payload_count           INTEGER,
  free_bike_payload_count         INTEGER,
  station_rows_inserted           INTEGER,
  free_bike_events_inserted       INTEGER,
  forecast_queue_files_processed  INTEGER DEFAULT 0,
  forecast_rows_logged            INTEGER DEFAULT 0,
  forecast_queue_files_failed     INTEGER DEFAULT 0,
  outcomes_resolved               INTEGER DEFAULT 0,
  metrics_rows_snapshotted        INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_collector_ticks_ticked ON collector_ticks(ticked_at);

CREATE TABLE IF NOT EXISTS model_forecasts (
  forecast_id       TEXT PRIMARY KEY,
  request_id        TEXT,
  source            TEXT,
  model_key         TEXT DEFAULT 'logistic',
  model_label       TEXT,
  model_version     TEXT,
  baseline_version  TEXT,
  query_place_key   TEXT,
  query_label       TEXT,
  station_id        TEXT NOT NULL,
  station_name      TEXT,
  forecasted_at     TIMESTAMP NOT NULL,
  target_at         TIMESTAMP NOT NULL,
  horizon_minutes   INTEGER NOT NULL,
  user_lat          DOUBLE,
  user_lon          DOUBLE,
  station_lat       DOUBLE,
  station_lon       DOUBLE,
  distance_km       DOUBLE,
  current_ebikes    INTEGER,
  p_has_ebike       DOUBLE NOT NULL,
  p_zero            DOUBLE,
  p_appears         DOUBLE,
  p_survives        DOUBLE,
  confidence        TEXT,
  is_recommended    BOOLEAN DEFAULT false
);

ALTER TABLE model_forecasts ADD COLUMN IF NOT EXISTS model_key TEXT DEFAULT 'logistic';
ALTER TABLE model_forecasts ADD COLUMN IF NOT EXISTS model_label TEXT;
ALTER TABLE model_forecasts ADD COLUMN IF NOT EXISTS query_place_key TEXT;
ALTER TABLE model_forecasts ADD COLUMN IF NOT EXISTS query_label TEXT;
ALTER TABLE model_forecasts ADD COLUMN IF NOT EXISTS expected_ebikes DOUBLE;
ALTER TABLE model_forecasts ADD COLUMN IF NOT EXISTS expected_total_bikes DOUBLE;
ALTER TABLE model_forecasts ADD COLUMN IF NOT EXISTS p_count_ebikes_json JSON;
ALTER TABLE model_forecasts ADD COLUMN IF NOT EXISTS p_count_total_json JSON;
ALTER TABLE model_forecasts ADD COLUMN IF NOT EXISTS p_capacity_violation DOUBLE;
ALTER TABLE model_forecasts ADD COLUMN IF NOT EXISTS p_dock_constrained_arrival DOUBLE;
ALTER TABLE model_forecasts ADD COLUMN IF NOT EXISTS expected_ebike_departures DOUBLE;
ALTER TABLE model_forecasts ADD COLUMN IF NOT EXISTS expected_classic_departures DOUBLE;
ALTER TABLE model_forecasts ADD COLUMN IF NOT EXISTS expected_ebike_arrivals DOUBLE;
ALTER TABLE model_forecasts ADD COLUMN IF NOT EXISTS expected_classic_arrivals DOUBLE;
ALTER TABLE model_forecasts ADD COLUMN IF NOT EXISTS walk_adjusted_score DOUBLE;
ALTER TABLE model_forecasts ADD COLUMN IF NOT EXISTS arrival_time_minutes DOUBLE;
ALTER TABLE model_forecasts ADD COLUMN IF NOT EXISTS p_arrival DOUBLE;
ALTER TABLE model_forecasts ADD COLUMN IF NOT EXISTS reliable_probability_lcb DOUBLE;
ALTER TABLE model_forecasts ADD COLUMN IF NOT EXISTS feature_snapshot_id TEXT;
ALTER TABLE model_forecasts ADD COLUMN IF NOT EXISTS decision_role TEXT;
ALTER TABLE model_forecasts ADD COLUMN IF NOT EXISTS recommended_rank INTEGER;
ALTER TABLE model_forecasts ADD COLUMN IF NOT EXISTS active_model_key TEXT;
ALTER TABLE model_forecasts ADD COLUMN IF NOT EXISTS active_model_source TEXT;
ALTER TABLE model_forecasts ADD COLUMN IF NOT EXISTS best_evaluated_model_key TEXT;

CREATE INDEX IF NOT EXISTS idx_model_forecasts_target ON model_forecasts(target_at);
CREATE INDEX IF NOT EXISTS idx_model_forecasts_station ON model_forecasts(station_id);
CREATE INDEX IF NOT EXISTS idx_model_forecasts_request ON model_forecasts(request_id);
CREATE INDEX IF NOT EXISTS idx_model_forecasts_model ON model_forecasts(model_key);
CREATE INDEX IF NOT EXISTS idx_model_forecasts_place ON model_forecasts(query_place_key);

CREATE TABLE IF NOT EXISTS prediction_queries (
  request_id        TEXT PRIMARY KEY,
  source            TEXT,
  queried_at        TIMESTAMP NOT NULL,
  query_label       TEXT,
  query_place_key   TEXT NOT NULL,
  lat               DOUBLE NOT NULL,
  lon               DOUBLE NOT NULL,
  near_radius_km    DOUBLE,
  search_radius_km  DOUBLE,
  candidate_count   INTEGER,
  best_station_id   TEXT
);

CREATE INDEX IF NOT EXISTS idx_prediction_queries_place ON prediction_queries(query_place_key);
CREATE INDEX IF NOT EXISTS idx_prediction_queries_time ON prediction_queries(queried_at);

CREATE TABLE IF NOT EXISTS model_outcomes (
  forecast_id          TEXT PRIMARY KEY,
  station_id           TEXT NOT NULL,
  horizon_minutes      INTEGER NOT NULL,
  target_at            TIMESTAMP NOT NULL,
  observed_at          TIMESTAMP NOT NULL,
  observed_ebikes      INTEGER,
  observed_has_ebike   BOOLEAN,
  resolved_at          TIMESTAMP NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_model_outcomes_station ON model_outcomes(station_id);
CREATE INDEX IF NOT EXISTS idx_model_outcomes_target ON model_outcomes(target_at);

CREATE TABLE IF NOT EXISTS model_metrics (
  metric_id        TEXT PRIMARY KEY,
  model_version    TEXT,
  computed_at      TIMESTAMP NOT NULL,
  window_hours     INTEGER NOT NULL,
  horizon_minutes  INTEGER,
  group_key        TEXT NOT NULL,
  group_value      TEXT NOT NULL,
  n                INTEGER NOT NULL,
  brier_score      DOUBLE,
  log_loss         DOUBLE,
  rank_loss        DOUBLE,
  observed_rate    DOUBLE,
  mean_prediction  DOUBLE
);

CREATE INDEX IF NOT EXISTS idx_model_metrics_computed ON model_metrics(computed_at);
ALTER TABLE model_metrics ADD COLUMN IF NOT EXISTS rank_loss DOUBLE;
ALTER TABLE model_metrics ADD COLUMN IF NOT EXISTS count_log_loss DOUBLE;
ALTER TABLE model_metrics ADD COLUMN IF NOT EXISTS crps DOUBLE;
ALTER TABLE model_metrics ADD COLUMN IF NOT EXISTS ece DOUBLE;
ALTER TABLE model_metrics ADD COLUMN IF NOT EXISTS recommended_hit_rate DOUBLE;
ALTER TABLE model_metrics ADD COLUMN IF NOT EXISTS distance_adjusted_regret DOUBLE;
ALTER TABLE model_metrics ADD COLUMN IF NOT EXISTS decision_rank_loss DOUBLE;
ALTER TABLE model_metrics ADD COLUMN IF NOT EXISTS is_active_model BOOLEAN;
ALTER TABLE model_metrics ADD COLUMN IF NOT EXISTS is_best_model BOOLEAN;
ALTER TABLE model_outcomes ADD COLUMN IF NOT EXISTS observed_total_bikes INTEGER;
ALTER TABLE model_outcomes ADD COLUMN IF NOT EXISTS observed_docks INTEGER;
ALTER TABLE model_outcomes ADD COLUMN IF NOT EXISTS status_age_minutes DOUBLE;
ALTER TABLE model_outcomes ADD COLUMN IF NOT EXISTS count_log_prob DOUBLE;
ALTER TABLE model_outcomes ADD COLUMN IF NOT EXISTS crps DOUBLE;

CREATE TABLE IF NOT EXISTS live_station_predictions (
  as_of TIMESTAMP NOT NULL,
  model_key TEXT NOT NULL,
  model_version TEXT,
  artifact_id TEXT,
  active_model_key TEXT,
  station_id TEXT NOT NULL,
  horizon_minutes INTEGER NOT NULL,
  p_has_ebike DOUBLE,
  p_zero DOUBLE,
  p_appears DOUBLE,
  p_survives DOUBLE,
  expected_ebikes DOUBLE,
  expected_total_bikes DOUBLE,
  p_count_ebikes_json JSON,
  p_count_total_json JSON,
  p_capacity_violation DOUBLE,
  p_dock_constrained_arrival DOUBLE,
  reliable_probability_lcb DOUBLE,
  calibration_group TEXT,
  feature_snapshot_id TEXT,
  data_age_minutes DOUBLE,
  created_at TIMESTAMP DEFAULT now(),
  PRIMARY KEY (as_of, model_key, station_id, horizon_minutes)
);

CREATE INDEX IF NOT EXISTS idx_live_station_predictions_model_station_horizon_asof
  ON live_station_predictions(model_key, station_id, horizon_minutes, as_of);
CREATE INDEX IF NOT EXISTS idx_live_station_predictions_as_of ON live_station_predictions(as_of);
CREATE INDEX IF NOT EXISTS idx_live_station_predictions_station ON live_station_predictions(station_id);

CREATE TABLE IF NOT EXISTS live_inflight_arrivals (
  updated_at TIMESTAMP,
  source_station_id TEXT,
  dst_station_id TEXT,
  horizon_minutes INTEGER,
  ebike_mass DOUBLE,
  classic_mass DOUBLE,
  expires_at TIMESTAMP,
  source_status_at TIMESTAMP,
  created_at TIMESTAMP DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_live_inflight_arrivals_dst ON live_inflight_arrivals(dst_station_id);
CREATE INDEX IF NOT EXISTS idx_live_inflight_arrivals_expires ON live_inflight_arrivals(expires_at);

CREATE TABLE IF NOT EXISTS background_job_runs (
  run_id TEXT PRIMARY KEY,
  job_name TEXT NOT NULL,
  service_name TEXT,
  status TEXT NOT NULL,
  started_at TIMESTAMP NOT NULL,
  finished_at TIMESTAMP,
  duration_seconds DOUBLE,
  message TEXT,
  metadata_json JSON,
  error TEXT,
  created_at TIMESTAMP DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_background_job_runs_job_started
  ON background_job_runs(job_name, started_at);
CREATE INDEX IF NOT EXISTS idx_background_job_runs_status ON background_job_runs(status);

CREATE TABLE IF NOT EXISTS background_job_locks (
  job_name TEXT PRIMARY KEY,
  run_id TEXT,
  owner_pid INTEGER,
  acquired_at TIMESTAMP,
  expires_at TIMESTAMP,
  heartbeat_at TIMESTAMP,
  lock_path TEXT,
  stale BOOLEAN DEFAULT false,
  metadata_json JSON
);

CREATE TABLE IF NOT EXISTS service_heartbeats (
  service_name TEXT PRIMARY KEY,
  heartbeat_at TIMESTAMP NOT NULL,
  pid INTEGER,
  metadata_json JSON
);

CREATE TABLE IF NOT EXISTS model_selection_state (
  computed_at TIMESTAMP,
  active_model_key TEXT,
  active_artifact_id TEXT,
  active_model_source TEXT,
  best_evaluated_model_key TEXT,
  best_sota_model_key TEXT,
  best_baseline_model_key TEXT,
  active_equals_best BOOLEAN,
  selection_metric TEXT,
  selection_window_hours INTEGER,
  min_resolved INTEGER,
  reason TEXT,
  metrics_json JSON,
  PRIMARY KEY(computed_at)
);

CREATE INDEX IF NOT EXISTS idx_model_selection_state_computed ON model_selection_state(computed_at);

CREATE TABLE IF NOT EXISTS calibration_state (
  computed_at TIMESTAMP,
  model_key TEXT,
  horizon_minutes INTEGER,
  calibration_group TEXT,
  n INTEGER,
  mean_prediction DOUBLE,
  observed_rate DOUBLE,
  ece DOUBLE,
  lcb_offset DOUBLE,
  metadata_json JSON,
  PRIMARY KEY(computed_at, model_key, horizon_minutes, calibration_group)
);

CREATE INDEX IF NOT EXISTS idx_calibration_state_model ON calibration_state(model_key, horizon_minutes);

CREATE TABLE IF NOT EXISTS model_artifacts (
  artifact_id TEXT PRIMARY KEY,
  model_key TEXT NOT NULL,
  model_family TEXT NOT NULL,
  model_version TEXT NOT NULL,
  trained_at TIMESTAMP NOT NULL,
  train_start TIMESTAMP,
  train_end TIMESTAMP,
  valid_start TIMESTAMP,
  valid_end TIMESTAMP,
  horizons INTEGER[],
  feature_columns TEXT[],
  artifact_path TEXT NOT NULL,
  metrics_json JSON,
  calibration_json JSON,
  is_primary_eligible BOOLEAN DEFAULT true,
  is_active BOOLEAN DEFAULT false,
  created_at TIMESTAMP DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_model_artifacts_key ON model_artifacts(model_key);
CREATE INDEX IF NOT EXISTS idx_model_artifacts_active ON model_artifacts(is_active);
CREATE INDEX IF NOT EXISTS idx_model_artifacts_trained ON model_artifacts(trained_at);

CREATE TABLE IF NOT EXISTS model_feature_snapshots (
  feature_snapshot_id TEXT PRIMARY KEY,
  request_id TEXT,
  model_key TEXT NOT NULL,
  station_id TEXT NOT NULL,
  anchor_ts TIMESTAMP NOT NULL,
  horizon_minutes INTEGER NOT NULL,
  feature_json JSON,
  status_reported_at TIMESTAMP,
  created_at TIMESTAMP DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_feature_snapshots_request ON model_feature_snapshots(request_id);
CREATE INDEX IF NOT EXISTS idx_feature_snapshots_station ON model_feature_snapshots(station_id);

CREATE TABLE IF NOT EXISTS dynamic_graph_edges (
  graph_key TEXT NOT NULL,
  anchor_ts TIMESTAMP NOT NULL,
  relation TEXT NOT NULL,
  src_station_id TEXT NOT NULL,
  dst_station_id TEXT NOT NULL,
  horizon_minutes INTEGER,
  weight DOUBLE NOT NULL,
  edge_rank INTEGER NOT NULL,
  distance_km DOUBLE,
  median_duration_minutes DOUBLE,
  lookback_start TIMESTAMP,
  lookback_end TIMESTAMP,
  created_at TIMESTAMP DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_dynamic_graph_edges_anchor ON dynamic_graph_edges(anchor_ts);
CREATE INDEX IF NOT EXISTS idx_dynamic_graph_edges_dst ON dynamic_graph_edges(dst_station_id);

CREATE TABLE IF NOT EXISTS external_events (
  event_id TEXT PRIMARY KEY,
  event_source TEXT,
  event_name TEXT,
  starts_at TIMESTAMP,
  ends_at TIMESTAMP,
  lat DOUBLE,
  lon DOUBLE,
  radius_km DOUBLE,
  metadata_json JSON,
  created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS recommendation_outcomes (
  request_id TEXT,
  model_key TEXT,
  station_id TEXT,
  decision_role TEXT,
  recommended_rank INTEGER,
  distance_km DOUBLE,
  arrival_time_minutes INTEGER,
  p_arrival DOUBLE,
  reliable_probability_lcb DOUBLE,
  walk_adjusted_score DOUBLE,
  target_at TIMESTAMP,
  observed_at TIMESTAMP,
  observed_ebikes INTEGER,
  observed_has_ebike BOOLEAN,
  realized_utility DOUBLE,
  oracle_utility DOUBLE,
  distance_adjusted_regret DOUBLE,
  created_at TIMESTAMP DEFAULT now(),
  PRIMARY KEY (request_id, model_key, station_id, decision_role)
);

CREATE INDEX IF NOT EXISTS idx_recommendation_outcomes_model ON recommendation_outcomes(model_key);
CREATE INDEX IF NOT EXISTS idx_recommendation_outcomes_target ON recommendation_outcomes(target_at);
"""


def _connection_path(read_only: bool) -> str:
    if read_only and config.ENABLE_READ_REPLICA and config.READ_DB_PATH.exists():
        return str(config.READ_DB_PATH)
    return str(config.DB_PATH)


def connect(read_only: bool = False, retries: int = 60, retry_sleep: float = 0.5) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection, retrying briefly if another process holds the lock.

    DuckDB doesn't allow a read-only reader process while another process has the
    file open read-write (and vice versa). The poller and dashboard both use
    short-lived connections to keep contention windows small; this retry covers
    the occasional collision.
    """
    config.ensure_dirs()
    last_exc: Exception | None = None
    for _ in range(retries):
        try:
            return duckdb.connect(_connection_path(read_only), read_only=read_only)
        except duckdb.IOException as exc:
            last_exc = exc
            time.sleep(retry_sleep)
    assert last_exc is not None
    raise last_exc


@contextmanager
def session(read_only: bool = False, retries: int = 60, retry_sleep: float = 0.5):
    conn = connect(read_only=read_only, retries=retries, retry_sleep=retry_sleep)
    try:
        yield conn
    finally:
        conn.close()


def init_schema(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(SCHEMA_SQL)


def refresh_read_replica() -> bool:
    """Atomically publish a read-only DuckDB snapshot for API/dashboard reads.

    Holds a write connection across CHECKPOINT + file copy so the snapshot
    captures a consistent state with no pending WAL writes. Without the
    CHECKPOINT, a plain file copy would capture metadata pointing at pages
    still pending in the WAL, producing a SerializationException on open.
    """
    if not config.ENABLE_READ_REPLICA or not config.DB_PATH.exists():
        return False
    config.ensure_dirs()
    tmp_path = config.READ_DB_PATH.with_name(f".{config.READ_DB_PATH.name}.tmp")

    with session(read_only=False) as conn:
        conn.execute("CHECKPOINT")
        shutil.copy2(config.DB_PATH, tmp_path)

    tmp_path.replace(config.READ_DB_PATH)
    stale_wal = config.READ_DB_PATH.with_name(config.READ_DB_PATH.name + ".wal")
    if stale_wal.exists():
        stale_wal.unlink()
    return True
