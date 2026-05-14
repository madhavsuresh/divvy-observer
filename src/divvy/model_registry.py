from __future__ import annotations

import json
import pickle
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb

from . import config, db


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _as_datetime(value) -> datetime:
    if isinstance(value, datetime):
        return value.replace(tzinfo=None) if value.tzinfo is not None else value
    try:
        ts = datetime.fromisoformat(str(value))
        return ts.replace(tzinfo=None) if ts.tzinfo is not None else ts
    except Exception:
        return _utc_now()


def _artifacts_dir(trained_at: datetime | None = None) -> Path:
    trained_at = trained_at or _utc_now()
    path = config.DATA_DIR / "artifacts" / "models" / trained_at.strftime("%Y%m%d")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _json_text(value: Any) -> str:
    if value is None:
        return "{}"
    if isinstance(value, str):
        return value
    return json.dumps(value, separators=(",", ":"), sort_keys=True, default=str)


def _write_model(path: Path, model_obj: object) -> None:
    if hasattr(model_obj, "save"):
        model_obj.save(path)
        return
    with path.open("wb") as handle:
        pickle.dump(model_obj, handle)


def _read_model(path: Path) -> object:
    with path.open("rb") as handle:
        return pickle.load(handle)


def artifact_method(artifact: dict | None) -> str:
    if not artifact:
        return ""
    model = artifact.get("model")
    metrics = artifact.get("metrics_json") or {}
    if isinstance(metrics, str):
        try:
            metrics = json.loads(metrics)
        except json.JSONDecodeError:
            metrics = {}
    return str(
        getattr(model, "method", None)
        or artifact.get("method")
        or metrics.get("method")
        or f"{artifact.get('model_key') or 'model'}_artifact"
    )


def is_trained_artifact(artifact: dict | None) -> bool:
    if not artifact or artifact.get("artifact_id") is None:
        return False
    method = artifact_method(artifact).lower()
    return (
        bool(artifact.get("artifact_path"))
        and bool(artifact.get("path_exists", True))
        and artifact.get("model") is not None
        and "bootstrap" not in method
        and "fallback" not in method
    )


def save_artifact(
    conn: duckdb.DuckDBPyConnection,
    model_key: str,
    model_obj: object,
    metadata: dict | None = None,
    metrics: dict | None = None,
    *,
    model_family: str | None = None,
    model_version: str | None = None,
    feature_columns: list[str] | None = None,
    train_start=None,
    train_end=None,
    valid_start=None,
    valid_end=None,
    horizons: list[int] | None = None,
    calibration: dict | None = None,
    is_primary_eligible: bool | None = None,
    is_active: bool | None = None,
) -> dict:
    db.init_schema(conn)
    metadata = dict(metadata or {})
    metrics = dict(metrics or {})
    for key, value in {
        "model_family": model_family,
        "model_version": model_version,
        "feature_columns": feature_columns,
        "train_start": train_start,
        "train_end": train_end,
        "valid_start": valid_start,
        "valid_end": valid_end,
        "horizons": horizons,
        "calibration": calibration,
        "is_primary_eligible": is_primary_eligible,
        "is_active": is_active,
    }.items():
        if value is not None:
            metadata[key] = value
    artifact_id = metadata.get("artifact_id") or f"{model_key}-{uuid.uuid4().hex}"
    model_version_value = metadata.get("model_version") or getattr(model_obj, "model_version", None) or getattr(model_obj, "version", None) or "v1"
    suffix = ".pkl"
    trained_at = metadata.get("trained_at") or getattr(model_obj, "trained_at", None) or _utc_now()
    path = Path(metadata.get("artifact_path") or (_artifacts_dir(_as_datetime(trained_at)) / f"{artifact_id}{suffix}"))
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_model(path, model_obj)
    metrics.setdefault("method", getattr(model_obj, "method", f"{model_key}_artifact"))
    metrics.setdefault("model_version", model_version_value)
    conn.execute(
        """
        INSERT OR REPLACE INTO model_artifacts (
          artifact_id, model_key, model_family, model_version, trained_at,
          train_start, train_end, valid_start, valid_end, horizons,
          feature_columns, artifact_path, metrics_json, calibration_json,
          is_primary_eligible, is_active
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            artifact_id,
            model_key,
            metadata.get("model_family") or model_key,
            model_version_value,
            trained_at,
            metadata.get("train_start"),
            metadata.get("train_end"),
            metadata.get("valid_start"),
            metadata.get("valid_end"),
            list(metadata.get("horizons") or []),
            list(metadata.get("feature_columns") or []),
            str(path),
            _json_text(metrics),
            _json_text(metadata.get("calibration") or {}),
            bool(metadata.get("is_primary_eligible", True)),
            bool(metadata.get("is_active", False)),
        ],
    )
    if metadata.get("is_active"):
        set_active_artifact(conn, artifact_id)
    return {
        "artifact_id": artifact_id,
        "model_key": model_key,
        "model_version": model_version_value,
        "artifact_path": str(path),
    }


def _row_to_artifact(row: dict | None) -> dict | None:
    if not row:
        return None
    path = Path(row["artifact_path"])
    if not path.exists():
        return None
    try:
        model = _read_model(path)
    except Exception:
        return None
    out = dict(row)
    out["model"] = model
    out["path_exists"] = True
    for key in ["metrics_json", "calibration_json"]:
        value = out.get(key)
        if isinstance(value, str):
            try:
                out[key] = json.loads(value)
            except json.JSONDecodeError:
                out[key] = {}
    out["method"] = artifact_method(out)
    out["trained_artifact"] = is_trained_artifact(out)
    out["usable"] = out.get("model") is not None
    return out


def load_latest_artifact(conn: duckdb.DuckDBPyConnection, model_key: str) -> dict | None:
    try:
        db.init_schema(conn)
    except Exception:
        pass
    try:
        row = conn.execute(
            """
            SELECT *
            FROM model_artifacts
            WHERE model_key = ?
            ORDER BY is_active DESC, trained_at DESC, created_at DESC
            LIMIT 1
            """,
            [model_key],
        ).df()
    except Exception:
        return None
    if row.empty:
        return None
    return _row_to_artifact(row.iloc[0].to_dict())


def load_latest_trained_artifact(conn: duckdb.DuckDBPyConnection, model_key: str) -> dict | None:
    try:
        db.init_schema(conn)
    except Exception:
        pass
    try:
        rows = conn.execute(
            """
            SELECT *
            FROM model_artifacts
            WHERE model_key = ?
            ORDER BY is_active DESC, trained_at DESC, created_at DESC
            """,
            [model_key],
        ).df()
    except Exception:
        return None
    for row in rows.to_dict(orient="records"):
        artifact = _row_to_artifact(row)
        if is_trained_artifact(artifact):
            return artifact
    return None


def load_active_artifact(conn: duckdb.DuckDBPyConnection) -> dict | None:
    try:
        db.init_schema(conn)
    except Exception:
        pass
    try:
        row = conn.execute(
            """
            SELECT *
            FROM model_artifacts
            WHERE is_active = true
            ORDER BY trained_at DESC, created_at DESC
            LIMIT 1
            """
        ).df()
    except Exception:
        return None
    if row.empty:
        return None
    return _row_to_artifact(row.iloc[0].to_dict())


def list_artifacts(conn: duckdb.DuckDBPyConnection) -> list[dict]:
    try:
        db.init_schema(conn)
    except Exception:
        pass
    try:
        rows = conn.execute(
            """
            SELECT *
            FROM model_artifacts
            ORDER BY trained_at DESC, created_at DESC
            """
        ).df()
    except Exception:
        return []
    out = []
    for row in rows.to_dict(orient="records"):
        path = Path(row.get("artifact_path") or "")
        row["path_exists"] = path.exists()
        loaded = _row_to_artifact(row) if path.exists() else None
        row["method"] = artifact_method(loaded or row)
        row["trained_artifact"] = is_trained_artifact(loaded)
        row["usable"] = bool(loaded and loaded.get("model") is not None)
        out.append(row)
    return out


def set_active_artifact(conn: duckdb.DuckDBPyConnection, artifact_id: str) -> None:
    db.init_schema(conn)
    rows = conn.execute(
        "SELECT * FROM model_artifacts WHERE artifact_id = ?",
        [artifact_id],
    ).df()
    if rows.empty:
        raise KeyError(f"Unknown artifact_id: {artifact_id}")
    artifact = _row_to_artifact(rows.iloc[0].to_dict())
    if not is_trained_artifact(artifact):
        method = artifact_method(artifact)
        raise ValueError(f"Artifact is not activatable trained model: {artifact_id} method={method}")
    conn.execute("UPDATE model_artifacts SET is_active = false")
    conn.execute("UPDATE model_artifacts SET is_active = true WHERE artifact_id = ?", [artifact_id])


def load_primary_eligible_models(conn: duckdb.DuckDBPyConnection) -> dict[str, dict]:
    db.init_schema(conn)
    rows = conn.execute(
        """
        SELECT model_key, MAX(trained_at) AS trained_at
        FROM model_artifacts
        WHERE COALESCE(is_primary_eligible, true) = true
        GROUP BY model_key
        """
    ).fetchall()
    out: dict[str, dict] = {}
    for model_key, _trained_at in rows:
        row = conn.execute(
            """
            SELECT *
            FROM model_artifacts
            WHERE model_key = ?
              AND COALESCE(is_primary_eligible, true) = true
            ORDER BY is_active DESC, trained_at DESC, created_at DESC
            LIMIT 1
            """,
            [str(model_key)],
        ).df()
        artifact = None if row.empty else _row_to_artifact(row.iloc[0].to_dict())
        if artifact is not None:
            out[str(model_key)] = artifact
    return out


def artifact_health(conn: duckdb.DuckDBPyConnection, artifact_id: str | None = None) -> dict:
    try:
        db.init_schema(conn)
    except Exception:
        pass
    if artifact_id:
        rows = conn.execute(
            "SELECT * FROM model_artifacts WHERE artifact_id = ?",
            [artifact_id],
        ).df()
    else:
        rows = conn.execute(
            """
            SELECT *
            FROM model_artifacts
            WHERE is_active = true
            ORDER BY trained_at DESC, created_at DESC
            LIMIT 1
            """
        ).df()
    if rows.empty:
        return {"ok": False, "reason": "artifact_not_found", "artifact_id": artifact_id}
    row = rows.iloc[0].to_dict()
    path = Path(row["artifact_path"])
    load_error = None
    loadable = False
    loaded = None
    if path.exists():
        try:
            loaded = _read_model(path)
            loadable = True
        except Exception as exc:
            load_error = str(exc)
    artifact = dict(row)
    artifact["model"] = loaded
    artifact["path_exists"] = path.exists()
    method = artifact_method(artifact)
    return {
        "ok": bool(path.exists() and loadable),
        "artifact_id": row.get("artifact_id"),
        "model_key": row.get("model_key"),
        "model_version": row.get("model_version"),
        "method": method,
        "artifact_path": str(path),
        "path_exists": path.exists(),
        "loadable": loadable,
        "load_error": load_error,
        "trained_artifact": is_trained_artifact(artifact),
        "is_active": bool(row.get("is_active")),
        "trained_at": row.get("trained_at"),
    }
