from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo

from . import config


LOCAL_TZ = ZoneInfo("America/Chicago")


@dataclass
class ScheduledJob:
    name: str
    interval_seconds: int | None = None
    local_time: time | None = None
    weekly_day: int | None = None
    last_started_monotonic: float = 0.0
    last_calendar_key: str | None = None

    def due(self, now_monotonic: float, now_utc: datetime | None = None) -> bool:
        if self.interval_seconds is not None:
            return (now_monotonic - self.last_started_monotonic) >= self.interval_seconds
        if self.local_time is None:
            return False
        now_utc = now_utc or datetime.now(timezone.utc)
        local = now_utc.astimezone(LOCAL_TZ)
        if self.weekly_day is not None and local.weekday() != self.weekly_day:
            return False
        if local.time().hour != self.local_time.hour or local.time().minute != self.local_time.minute:
            return False
        key = local.strftime("%Y-%m-%d-%H-%M")
        return key != self.last_calendar_key

    def mark_started(self, now_monotonic: float, now_utc: datetime | None = None) -> None:
        self.last_started_monotonic = now_monotonic
        if self.local_time is not None:
            now_utc = now_utc or datetime.now(timezone.utc)
            self.last_calendar_key = now_utc.astimezone(LOCAL_TZ).strftime("%Y-%m-%d-%H-%M")


@dataclass
class Scheduler:
    jobs: list[ScheduledJob] = field(default_factory=list)

    def due_jobs(self, now_monotonic: float, now_utc: datetime | None = None) -> list[ScheduledJob]:
        return [job for job in self.jobs if job.due(now_monotonic, now_utc)]


def _parse_local_time(value: str) -> time:
    hour, minute = value.split(":", 1)
    return time(hour=int(hour), minute=int(minute))


def _weekday(value: str) -> int:
    names = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
    }
    return names.get(value.strip().lower(), 6)


def default_scheduler() -> Scheduler:
    return Scheduler(
        jobs=[
            ScheduledJob("drain-forecast-queue", interval_seconds=60),
            ScheduledJob("resolve-outcomes", interval_seconds=config.OUTCOME_RESOLVE_INTERVAL_SECONDS),
            ScheduledJob("refresh-live-predictions", interval_seconds=config.PREDICTION_CACHE_INTERVAL_SECONDS),
            ScheduledJob("refresh-inflight", interval_seconds=300),
            ScheduledJob("refresh-inferred-flows", interval_seconds=300),
            ScheduledJob("refresh-comparison-predictions", interval_seconds=config.COMPARISON_CACHE_INTERVAL_SECONDS),
            ScheduledJob("snapshot-metrics", interval_seconds=config.METRIC_SNAPSHOT_INTERVAL_SECONDS),
            ScheduledJob("select-model", interval_seconds=3600),
            ScheduledJob("cleanup", interval_seconds=3600),
            # Pull the previous month's Divvy historical trip dump once a day
            # (Divvy publishes around the 10th — running daily catches it
            # within ~24h without re-downloading anything we already have).
            ScheduledJob("sync-tripdata", interval_seconds=24 * 3600),
            ScheduledJob("train-nightly", local_time=_parse_local_time(config.NIGHTLY_TRAIN_LOCAL_TIME)),
            ScheduledJob(
                "train-weekly",
                local_time=_parse_local_time(config.WEEKLY_TRAIN_LOCAL_TIME),
                weekly_day=_weekday(config.WEEKLY_TRAIN_DAY),
            ),
        ]
    )
