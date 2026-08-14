"""Snapshot of the Mac's system health — CPU/RAM/disk/battery plus the
foreground-visible running apps (via System Events)."""

from __future__ import annotations

import subprocess

import psutil


def _running_apps(limit: int = 8) -> list[str]:
    script = 'tell application "System Events" to get name of (processes where background only is false)'
    try:
        result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=10, check=True)
        apps = [a.strip() for a in result.stdout.split(",") if a.strip()]
        return apps[:limit]
    except Exception:  # noqa: BLE001
        return []


def get_system_info() -> dict:
    battery = psutil.sensors_battery()
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.5),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage("/").percent,
        "battery_percent": battery.percent if battery else None,
        "battery_plugged": battery.power_plugged if battery else None,
        "top_apps": _running_apps(),
    }
