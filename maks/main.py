"""Entrypoint: runs the dashboard server, the wake-word/hotkey voice loop,
and a live system-stats publisher side by side. Run with `python -m maks.main`
(see scripts/maks.service for running this as a systemd unit on boot).
"""

from __future__ import annotations

import threading
import time

import uvicorn

from maks import runtime
from maks.events import bus
from maks.pipeline import get_graph, handle_command
from maks.server.app import app
from maks.settings import settings
from maks.tools.system_stats import get_system_stats
from maks.tools.weather_tools import get_current_weather
from maks.voice.hotkey import wait_for_hotkey
from maks.voice.stt import listen_and_transcribe
from maks.voice.tts import speak
from maks.voice.wake_word import wait_for_wake_word

_stop_event = threading.Event()

_STATS_INTERVAL_SECONDS = 1.5
_WEATHER_REFRESH_SECONDS = 600  # 10 minutes -- stays fresh without hammering Open-Meteo


def _greeting() -> str:
    hour = time.localtime().tm_hour
    time_of_day = "morning" if hour < 12 else "afternoon" if hour < 17 else "evening"

    try:
        weather = get_current_weather()
    except Exception:  # noqa: BLE001
        weather = "I couldn't reach the weather service just now"

    return f"Hello sir, good {time_of_day}. {weather}. What can I do for you?"


def _voice_loop() -> None:
    """First activation each run needs the wake phrase, same as always. Every
    activation after that stays lit (no dimming back to `idle`) and waits on
    the global "hold Ctrl" hotkey instead — see maks/voice/hotkey.py.
    """
    print("[maks] warming up the local model, agents, and MCP servers…")
    runtime.run_coro(get_graph())  # build once up front so the first wake doesn't stall
    print(f"[maks] ready — listening for the wake phrase: '{settings.wake_phrase}'")

    woke_once = False
    while not _stop_event.is_set():
        if not woke_once:
            bus.publish("idle")
            if not wait_for_wake_word(stop_event=_stop_event):
                break
            woke_once = True

            bus.publish("wake")
            greeting = _greeting()
            bus.publish("speaking", text=greeting)
            speak(greeting)
        else:
            bus.publish("ready")
            if not wait_for_hotkey(stop_event=_stop_event):
                break
            bus.publish("wake")

        bus.publish("listening")
        transcript = listen_and_transcribe()
        if not transcript:
            continue

        reply = runtime.run_coro(handle_command(transcript))
        bus.publish("speaking", text=reply)
        speak(reply)


def _stats_loop() -> None:
    """Publishes live system telemetry (+ an occasionally-refreshed weather
    string) for the dashboard's HUD panels. Independent of the voice loop's
    state — panels stay live even before the first wake, while dim.
    """
    last_weather_refresh = 0.0
    weather_text = ""

    while not _stop_event.is_set():
        try:
            stats = get_system_stats()
        except Exception:  # noqa: BLE001
            stats = {}

        now = time.monotonic()
        if now - last_weather_refresh >= _WEATHER_REFRESH_SECONDS:
            try:
                weather_text = get_current_weather()
            except Exception:  # noqa: BLE001
                pass  # keep the last-known reading rather than blanking the panel
            last_weather_refresh = now

        bus.publish("stats", weather=weather_text, **stats)
        _stop_event.wait(_STATS_INTERVAL_SECONDS)


def main() -> None:
    runtime.start()

    threading.Thread(target=_voice_loop, daemon=True).start()
    threading.Thread(target=_stats_loop, daemon=True).start()

    try:
        uvicorn.run(app, host=settings.dashboard_host, port=settings.dashboard_port, log_level="info")
    finally:
        _stop_event.set()


if __name__ == "__main__":
    main()
