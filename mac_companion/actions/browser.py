"""Opens things in the Mac's default browser: YouTube videos (finds the top
result for a query and jumps straight to the watch page) or arbitrary URLs.
"""

from __future__ import annotations

import re
import subprocess
import urllib.parse

import requests


def _open(url: str) -> None:
    subprocess.run(["open", url], check=True, timeout=10)


def open_url(url: str) -> str:
    try:
        _open(url)
        return f"Opened {url}."
    except Exception as exc:  # noqa: BLE001
        return f"Failed to open {url}: {exc}"


def play_youtube(query: str) -> str:
    """Find the top YouTube result for `query` and open its watch page
    directly (so it starts playing), rather than just a search results page.
    """
    search_url = "https://www.youtube.com/results?search_query=" + urllib.parse.quote(query)
    try:
        resp = requests.get(search_url, timeout=10, headers={"User-Agent": "Mozilla/5.0 (Maks)"})
        resp.raise_for_status()
        match = re.search(r'"videoId":"([a-zA-Z0-9_-]{11})"', resp.text)
    except Exception as exc:  # noqa: BLE001
        return f"Couldn't search YouTube: {exc}"

    if not match:
        # Fall back to just opening the search results page.
        try:
            _open(search_url)
            return f"Couldn't find a direct video, opened YouTube search results for '{query}'."
        except Exception as exc:  # noqa: BLE001
            return f"Failed to open YouTube: {exc}"

    video_id = match.group(1)
    watch_url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        _open(watch_url)
        return f"Playing the top YouTube result for '{query}'."
    except Exception as exc:  # noqa: BLE001
        return f"Failed to open {watch_url}: {exc}"
