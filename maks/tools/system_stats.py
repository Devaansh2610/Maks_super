"""Live system telemetry for the dashboard's HUD panels — CPU/RAM/disk/
network/battery/uptime for the local (Windows) box Maks itself runs on.
Same psutil-based approach as mac_companion/actions/system_info.py uses for
the Mac's own stats, just for a different machine.

Plain function, not an MCP tool (same reasoning as maks/tools/weather_tools.py):
this is for maks/main.py's own internal stats-publishing loop, not something
an LLM agent calls.
"""

from __future__ import annotations

import os
import time

import psutil

_DISK_ROOT = os.path.abspath(os.sep)  # "C:\\" on Windows, "/" on Linux/macOS
_GB = 1024**3

# Network counters are cumulative since boot; the dashboard wants a live
# rate (bytes/sec), so track the previous reading and compute a delta each
# call. Module-level since get_system_stats() is meant to be polled
# repeatedly from one long-lived loop (maks/main.py's stats thread), not
# called once.
_prev_net: tuple[float, int, int] | None = None  # (timestamp, bytes_sent, bytes_recv)


def _format_uptime(boot_time: float) -> str:
    seconds = max(0, int(time.time() - boot_time))
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, _ = divmod(seconds, 60)
    if days:
        return f"{days}d {hours}h {minutes}m"
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def _network_rates() -> tuple[float, float]:
    global _prev_net
    now = time.monotonic()
    counters = psutil.net_io_counters()

    if _prev_net is None:
        _prev_net = (now, counters.bytes_sent, counters.bytes_recv)
        return 0.0, 0.0

    prev_time, prev_sent, prev_recv = _prev_net
    elapsed = max(now - prev_time, 0.001)  # guard against a near-zero interval
    sent_rate = max(0.0, (counters.bytes_sent - prev_sent) / elapsed)
    recv_rate = max(0.0, (counters.bytes_recv - prev_recv) / elapsed)
    _prev_net = (now, counters.bytes_sent, counters.bytes_recv)
    return sent_rate, recv_rate


def get_system_stats() -> dict:
    """Returns a flat dict of current stats — safe to call repeatedly
    (network rates are computed from the delta since the previous call;
    the very first call reports 0 for those since there's no prior reading
    to diff against). Never raises — a metric psutil can't read on this
    platform/hardware (e.g. no battery) comes back as None, not an error.
    """
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage(_DISK_ROOT)
    battery = psutil.sensors_battery()
    net_sent_rate, net_recv_rate = _network_rates()

    return {
        "cpu_percent": psutil.cpu_percent(interval=None),
        "ram_percent": memory.percent,
        "ram_used_gb": round(memory.used / _GB, 1),
        "ram_total_gb": round(memory.total / _GB, 1),
        "disk_percent": disk.percent,
        "disk_used_gb": round(disk.used / _GB, 1),
        "disk_total_gb": round(disk.total / _GB, 1),
        "net_sent_kbps": round(net_sent_rate / 1024, 1),
        "net_recv_kbps": round(net_recv_rate / 1024, 1),
        "battery_percent": battery.percent if battery else None,
        "battery_plugged": battery.power_plugged if battery else None,
        "uptime": _format_uptime(psutil.boot_time()),
    }
