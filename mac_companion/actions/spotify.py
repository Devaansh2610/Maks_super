"""Controls the Spotify desktop app on the Mac via AppleScript (osascript).

Requires the Spotify app to be installed. Playing a `spotify:search:` URI is
supported by Spotify's own AppleScript dictionary and plays the top result.
"""

from __future__ import annotations

import subprocess


def _run_applescript(script: str) -> None:
    subprocess.run(["osascript", "-e", script], check=True, capture_output=True, timeout=15)


def play(query: str) -> str:
    escaped = query.replace('"', '\\"')
    script = f'''
    tell application "Spotify"
        activate
        play track "spotify:search:{escaped}"
    end tell
    '''
    try:
        _run_applescript(script)
        return f"Playing '{query}' on Spotify."
    except subprocess.CalledProcessError as exc:
        return f"Spotify control failed: {exc.stderr.decode(errors='ignore')}"
    except Exception as exc:  # noqa: BLE001
        return f"Spotify control failed: {exc}"


_ACTIONS = {
    "pause": "pause",
    "play": "play",
    "next": "next track",
    "previous": "previous track",
}


def control(action: str) -> str:
    applescript_command = _ACTIONS.get(action)
    if not applescript_command:
        return f"Unknown Spotify action '{action}'."

    script = f'tell application "Spotify" to {applescript_command}'
    try:
        _run_applescript(script)
        return f"Spotify: {action}."
    except subprocess.CalledProcessError as exc:
        return f"Spotify control failed: {exc.stderr.decode(errors='ignore')}"
    except Exception as exc:  # noqa: BLE001
        return f"Spotify control failed: {exc}"
