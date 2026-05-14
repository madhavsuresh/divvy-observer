from __future__ import annotations

import os
import plistlib
import shutil
import subprocess
import sys
from pathlib import Path

from . import config


SERVICE_LABELS = {
    "collector": "divvy.collector",
    "automation": "divvy.automation",
    "api": "divvy.api",
    "dashboard": "divvy.dashboard",
}

NOT_LOADED_CODES = {113}


def _pythonpath(project_root: Path) -> str:
    return str(project_root / "src")


def _base_plist(label: str, args: list[str], project_root: Path) -> dict:
    config.ensure_dirs()
    return {
        "Label": label,
        "ProgramArguments": args,
        "WorkingDirectory": str(project_root),
        "RunAtLoad": True,
        "KeepAlive": True,
        "StandardOutPath": str(config.LOG_DIR / f"{label}.out.log"),
        "StandardErrorPath": str(config.LOG_DIR / f"{label}.err.log"),
        "EnvironmentVariables": {
            "PYTHONPATH": _pythonpath(project_root),
            "DIVVY_DB_PATH": str(config.DB_PATH),
            "DIVVY_DATA_DIR": str(config.DATA_DIR),
            "DIVVY_LOG_DIR": str(config.LOG_DIR),
            "UV_CACHE_DIR": str(config.DATA_DIR / "uv-cache"),
            "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
        },
    }


def _service_args(module_or_command: list[str], project_root: Path, python_executable: str | None) -> list[str]:
    if python_executable:
        return [python_executable, *module_or_command]
    uv = shutil.which("uv")
    if uv:
        return [uv, "run", "--project", str(project_root), "python", *module_or_command]
    return [sys.executable, *module_or_command]


def build_plists(
    *,
    python_executable: str | None = None,
    project_root: str | Path | None = None,
    enable_dashboard: bool | None = None,
) -> dict[str, dict]:
    root = Path(project_root or config.PROJECT_ROOT).resolve()
    enable_dashboard = config.LAUNCHD_ENABLE_DASHBOARD if enable_dashboard is None else bool(enable_dashboard)
    services = {
        "collector": _base_plist(
            SERVICE_LABELS["collector"],
            _service_args(["-m", "divvy.collector"], root, python_executable),
            root,
        ),
        "automation": _base_plist(
            SERVICE_LABELS["automation"],
            _service_args(["-m", "divvy.automation", "run"], root, python_executable),
            root,
        ),
        "api": _base_plist(
            SERVICE_LABELS["api"],
            _service_args(
                [
                    "-m",
                    "uvicorn",
                    "divvy.api:app",
                    "--host",
                    config.API_HOST,
                    "--port",
                    str(config.API_PORT),
                ],
                root,
                python_executable,
            ),
            root,
        ),
    }
    if enable_dashboard:
        services["dashboard"] = _base_plist(
            SERVICE_LABELS["dashboard"],
            _service_args(
                [
                    "-m",
                    "streamlit",
                    "run",
                    str(root / "src" / "divvy" / "dashboard.py"),
                    "--server.address",
                    "127.0.0.1",
                    "--server.port",
                    str(config.DASHBOARD_PORT),
                ],
                root,
                python_executable,
            ),
            root,
        )
    return services


def write_plists(
    output_dir: str | Path,
    *,
    python_executable: str | None = None,
    project_root: str | Path | None = None,
    enable_dashboard: bool | None = None,
) -> list[Path]:
    out = Path(output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for plist in build_plists(
        python_executable=python_executable,
        project_root=project_root,
        enable_dashboard=enable_dashboard,
    ).values():
        path = out / f"{plist['Label']}.plist"
        with path.open("wb") as handle:
            plistlib.dump(plist, handle, sort_keys=True)
        written.append(path)
    return written


def install_launchd(*, enable_dashboard: bool | None = None) -> dict:
    target = Path.home() / "Library" / "LaunchAgents"
    try:
        written = write_plists(target, enable_dashboard=enable_dashboard)
    except PermissionError as exc:
        return {
            "status": "permission_denied",
            "error": str(exc),
            "manual_commands": [
                "mkdir -p ~/Library/LaunchAgents",
                f"cd {config.PROJECT_ROOT}",
                "uv run divvy install-launchd",
            ],
        }
    uid = os.getuid()
    commands = []
    for path in written:
        commands.append(f"launchctl bootstrap gui/{uid} {path}")
        commands.append(f"launchctl enable gui/{uid}/{path.stem}")
        commands.append(f"launchctl kickstart -k gui/{uid}/{path.stem}")
    return {
        "status": "installed",
        "plist_paths": [str(path) for path in written],
        "load_commands": commands,
    }


def _launchctl(args: list[str], *, timeout: float = 10.0) -> dict:
    command = ["launchctl", *args]
    try:
        proc = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "command": " ".join(command),
            "returncode": proc.returncode,
            "ok": proc.returncode == 0,
            "stdout": proc.stdout[-4000:],
            "stderr": proc.stderr[-4000:],
        }
    except Exception as exc:
        return {
            "command": " ".join(command),
            "returncode": None,
            "ok": False,
            "error": str(exc),
        }


def _label_loaded(label: str, uid: int) -> bool:
    result = _launchctl(["print", f"gui/{uid}/{label}"], timeout=5.0)
    return bool(result.get("ok"))


def _action_ok(action: dict, *, allow_not_loaded: bool = False) -> bool:
    if action.get("ok"):
        return True
    if allow_not_loaded and action.get("returncode") in NOT_LOADED_CODES:
        return True
    text = f"{action.get('stdout') or ''}\n{action.get('stderr') or ''}".lower()
    if "service is already loaded" in text or "already bootstrapped" in text:
        return True
    if allow_not_loaded and ("service cannot be found" in text or "could not find service" in text):
        return True
    return False


def _compact_status() -> dict:
    return {
        label: {
            "loaded": info.get("loaded"),
            "returncode": info.get("returncode"),
            "error": info.get("error"),
        }
        for label, info in launchd_status().items()
    }


def start_launchd(*, enable_dashboard: bool | None = None) -> dict:
    """Write LaunchAgent plists and load/kickstart all enabled services."""
    installed = install_launchd(enable_dashboard=enable_dashboard)
    if installed.get("status") != "installed":
        return installed
    uid = os.getuid()
    actions = []
    for raw_path in installed.get("plist_paths") or []:
        path = Path(raw_path)
        label = path.stem
        if _label_loaded(label, uid):
            action = _launchctl(["bootout", f"gui/{uid}/{label}"])
            action.update({"service": label, "action": "bootout"})
            actions.append(action)
        action = _launchctl(["bootstrap", f"gui/{uid}", str(path)])
        action.update({"service": label, "action": "bootstrap"})
        actions.append(action)
        action = _launchctl(["enable", f"gui/{uid}/{label}"])
        action.update({"service": label, "action": "enable"})
        actions.append(action)
    ok = all(_action_ok(action) for action in actions)
    return {
        "status": "started" if ok else "partial_failure",
        "plist_paths": installed.get("plist_paths", []),
        "actions": actions,
        "status_after": _compact_status(),
    }


def stop_launchd() -> dict:
    """Boot out all Divvy LaunchAgent services without deleting plist files."""
    uid = os.getuid()
    actions = []
    for label in SERVICE_LABELS.values():
        action = _launchctl(["bootout", f"gui/{uid}/{label}"])
        action.update({"service": label, "action": "bootout"})
        actions.append(action)
    ok = all(_action_ok(action, allow_not_loaded=True) for action in actions)
    return {
        "status": "stopped" if ok else "partial_failure",
        "actions": actions,
        "status_after": _compact_status(),
    }


def restart_launchd(*, enable_dashboard: bool | None = None) -> dict:
    stopped = stop_launchd()
    started = start_launchd(enable_dashboard=enable_dashboard)
    ok = stopped.get("status") in {"stopped", "partial_failure"} and started.get("status") == "started"
    return {
        "status": "restarted" if ok else "partial_failure",
        "stop": stopped,
        "start": started,
    }


def uninstall_launchd() -> dict:
    target = Path.home() / "Library" / "LaunchAgents"
    stopped = stop_launchd()
    removed = []
    for label in SERVICE_LABELS.values():
        path = target / f"{label}.plist"
        if path.exists():
            try:
                path.unlink()
                removed.append(str(path))
            except PermissionError:
                pass
    return {"status": "uninstalled", "removed": removed, "stop": stopped}


def launchd_status() -> dict:
    uid = os.getuid()
    out = {}
    for label in SERVICE_LABELS.values():
        try:
            proc = subprocess.run(
                ["launchctl", "print", f"gui/{uid}/{label}"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            out[label] = {
                "returncode": proc.returncode,
                "loaded": proc.returncode == 0,
                "stdout": proc.stdout[-4000:],
                "stderr": proc.stderr[-2000:],
            }
        except Exception as exc:
            out[label] = {"loaded": False, "error": str(exc)}
    return out
