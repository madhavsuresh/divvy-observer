from __future__ import annotations

import plistlib
from types import SimpleNamespace

from divvy import launchd


def test_launchd_plist_generation_is_idempotent(tmp_path) -> None:
    first = launchd.write_plists(
        tmp_path,
        python_executable="/usr/bin/python3",
        project_root=tmp_path,
        enable_dashboard=True,
    )
    second = launchd.write_plists(
        tmp_path,
        python_executable="/usr/bin/python3",
        project_root=tmp_path,
        enable_dashboard=True,
    )

    assert [p.name for p in first] == [p.name for p in second]
    api = plistlib.loads((tmp_path / "divvy.api.plist").read_bytes())
    assert api["Label"] == "divvy.api"
    assert "uvicorn" in api["ProgramArguments"]
    assert api["WorkingDirectory"] == str(tmp_path)


def test_start_launchd_bootstraps_enabled_services(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    calls = []

    def fake_run(command, check=False, capture_output=True, text=True, timeout=10):
        calls.append(command)
        if command[:2] == ["launchctl", "print"]:
            return SimpleNamespace(returncode=113, stdout="", stderr="service not found")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(launchd.subprocess, "run", fake_run)

    result = launchd.start_launchd(enable_dashboard=False)

    assert result["status"] == "started"
    assert (tmp_path / "Library" / "LaunchAgents" / "divvy.collector.plist").exists()
    assert (tmp_path / "Library" / "LaunchAgents" / "divvy.dashboard.plist").exists() is False
    joined = [" ".join(call) for call in calls]
    assert any("bootstrap" in call and "divvy.collector.plist" in call for call in joined)
    assert any("enable gui/" in call and "divvy.api" in call for call in joined)


def test_stop_launchd_treats_missing_services_as_stopped(monkeypatch) -> None:
    def fake_run(command, check=False, capture_output=True, text=True, timeout=10):
        return SimpleNamespace(returncode=113, stdout="", stderr="service not found")

    monkeypatch.setattr(launchd.subprocess, "run", fake_run)

    result = launchd.stop_launchd()

    assert result["status"] == "stopped"
    assert {action["action"] for action in result["actions"]} == {"bootout"}
