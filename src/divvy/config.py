from __future__ import annotations

import os
from pathlib import Path

STATION_INFO_URL = "https://gbfs.divvybikes.com/gbfs/en/station_information.json"
STATION_STATUS_URL = "https://gbfs.divvybikes.com/gbfs/en/station_status.json"
FREE_BIKE_STATUS_URL = "https://gbfs.divvybikes.com/gbfs/en/free_bike_status.json"

POLL_INTERVAL_SECONDS = int(os.environ.get("DIVVY_POLL_INTERVAL", "60"))
STATION_INFO_REFRESH_SECONDS = int(os.environ.get("DIVVY_INFO_REFRESH", str(6 * 60 * 60)))

REQUEST_TIMEOUT = 15
USER_AGENT = "divvy-observer/0.1 (personal research)"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"

DATA_DIR = Path(os.environ.get("DIVVY_DATA_DIR", DEFAULT_DATA_DIR))
DB_PATH = Path(os.environ.get("DIVVY_DB_PATH", DATA_DIR / "divvy.duckdb"))
READ_DB_PATH = Path(os.environ.get("DIVVY_READ_DB_PATH", DATA_DIR / "divvy_readonly.duckdb"))
ENABLE_READ_REPLICA = os.environ.get("DIVVY_ENABLE_READ_REPLICA", "1") != "0"
READ_REPLICA_REFRESH_SECONDS = int(os.environ.get("DIVVY_READ_REPLICA_REFRESH_SECONDS", "60"))
LOG_PATH = Path(os.environ.get("DIVVY_LOG_PATH", DATA_DIR / "poller.log"))
FORECAST_QUEUE_DIR = Path(os.environ.get("DIVVY_FORECAST_QUEUE_DIR", DATA_DIR / "forecast_queue"))
FORECAST_QUEUE_DRAIN_LIMIT = int(os.environ.get("DIVVY_FORECAST_QUEUE_DRAIN_LIMIT", "200"))
MODEL_OUTCOME_RESOLVE_SECONDS = int(os.environ.get("DIVVY_MODEL_OUTCOME_RESOLVE_SECONDS", "60"))
MODEL_METRICS_SNAPSHOT_SECONDS = int(os.environ.get("DIVVY_MODEL_METRICS_SNAPSHOT_SECONDS", "600"))
SELF_EVAL_INTERVAL_SECONDS = int(os.environ.get("DIVVY_SELF_EVAL_INTERVAL_SECONDS", "300"))
SELF_EVAL_STATION_SAMPLE = int(os.environ.get("DIVVY_SELF_EVAL_STATION_SAMPLE", "25"))

DISABLE_REQUEST_TRAINING = os.environ.get("DIVVY_DISABLE_REQUEST_TRAINING", "1") != "0"
ACTIVE_MODEL_POLICY = os.environ.get("DIVVY_ACTIVE_MODEL_POLICY", "best_sota")
ACTIVE_MODEL_KEY = os.environ.get("DIVVY_ACTIVE_MODEL_KEY", "")
PREDICTION_CACHE_INTERVAL_SECONDS = int(os.environ.get("DIVVY_PREDICTION_CACHE_INTERVAL_SECONDS", "120"))
COMPARISON_CACHE_INTERVAL_SECONDS = int(os.environ.get("DIVVY_COMPARISON_CACHE_INTERVAL_SECONDS", "900"))
OUTCOME_RESOLVE_INTERVAL_SECONDS = int(os.environ.get("DIVVY_OUTCOME_RESOLVE_INTERVAL_SECONDS", "60"))
METRIC_SNAPSHOT_INTERVAL_SECONDS = int(os.environ.get("DIVVY_METRIC_SNAPSHOT_INTERVAL_SECONDS", "3600"))
NIGHTLY_TRAIN_LOCAL_TIME = os.environ.get("DIVVY_NIGHTLY_TRAIN_LOCAL_TIME", "02:30")
WEEKLY_TRAIN_DAY = os.environ.get("DIVVY_WEEKLY_TRAIN_DAY", "Sunday")
WEEKLY_TRAIN_LOCAL_TIME = os.environ.get("DIVVY_WEEKLY_TRAIN_LOCAL_TIME", "03:30")
ACTIVE_SWITCH_WINDOW_HOURS = int(os.environ.get("DIVVY_ACTIVE_SWITCH_WINDOW_HOURS", "168"))
ACTIVE_SWITCH_MIN_RESOLVED = int(os.environ.get("DIVVY_ACTIVE_SWITCH_MIN_RESOLVED", "100"))
ACTIVE_SWITCH_MARGIN = float(os.environ.get("DIVVY_ACTIVE_SWITCH_MARGIN", "0.01"))
CACHE_MAX_AGE_MINUTES = float(os.environ.get("DIVVY_CACHE_MAX_AGE_MINUTES", "5"))
COMPARISON_CACHE_MAX_AGE_MINUTES = float(os.environ.get("DIVVY_COMPARISON_CACHE_MAX_AGE_MINUTES", "15"))
LAUNCHD_ENABLE_DASHBOARD = os.environ.get("DIVVY_LAUNCHD_ENABLE_DASHBOARD", "1") != "0"
API_HOST = os.environ.get("DIVVY_API_HOST", "127.0.0.1")
API_PORT = int(os.environ.get("DIVVY_API_PORT", "8000"))
DASHBOARD_PORT = int(os.environ.get("DIVVY_DASHBOARD_PORT", "8501"))
SERVICE_LOCK_DIR = Path(os.environ.get("DIVVY_SERVICE_LOCK_DIR", DATA_DIR / "locks"))
LOG_DIR = Path(os.environ.get("DIVVY_LOG_DIR", PROJECT_ROOT / "logs"))
LIVE_PREDICTION_RETENTION_HOURS = int(os.environ.get("DIVVY_LIVE_PREDICTION_RETENTION_HOURS", "24"))
JOB_LOCK_TTL_SECONDS = int(os.environ.get("DIVVY_JOB_LOCK_TTL_SECONDS", "1800"))
TRAIN_ANCHOR_EVERY_MIN = int(os.environ.get("DIVVY_TRAIN_ANCHOR_EVERY_MIN", "30"))


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    READ_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    FORECAST_QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    SERVICE_LOCK_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
