"""System-wide "hold Ctrl for N seconds" shortcut that re-triggers listening
after the first wake-phrase activation each run, so you don't have to keep
saying the wake phrase — see maks/main.py's _voice_loop() for how the two
triggers (wake word vs. this hotkey) fit together.

Global, not dashboard-only: uses pynput's low-level OS keyboard hook, so it
fires no matter which window has focus.
"""

from __future__ import annotations

import threading
import time

from pynput import keyboard

from maks.settings import settings

_CTRL_KEYS = (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r)
_POLL_SECONDS = 0.05


def wait_for_hotkey(stop_event: threading.Event | None = None) -> bool:
    """Blocks until Ctrl has been held continuously for
    settings.hotkey_hold_seconds; returns True in that case, or False if
    `stop_event` was set first (used for clean shutdown). A quick Ctrl+C /
    Ctrl+Tab / etc. won't trigger this — only a deliberate multi-second hold
    will, since the hold timer is cancelled the instant Ctrl is released.
    """
    hold_seconds = settings.hotkey_hold_seconds
    press_time: list[float | None] = [None]  # mutable cell, written from the listener thread

    def on_press(key):
        if key in _CTRL_KEYS and press_time[0] is None:
            press_time[0] = time.monotonic()

    def on_release(key):
        if key in _CTRL_KEYS:
            press_time[0] = None

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    try:
        while True:
            if stop_event is not None and stop_event.is_set():
                return False

            started = press_time[0]
            if started is not None and time.monotonic() - started >= hold_seconds:
                return True

            time.sleep(_POLL_SECONDS)
    finally:
        listener.stop()
