"""Fast file search on the Mac using Spotlight's `mdfind` CLI — no indexing
work needed, it's already there."""

from __future__ import annotations

import os
import subprocess


def search(query: str, root: str | None = None, limit: int = 25) -> list[str]:
    cmd = ["mdfind"]
    if root:
        cmd += ["-onlyin", os.path.expanduser(root)]
    cmd.append(query)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15, check=True)
    except Exception:  # noqa: BLE001
        return []

    paths = [line for line in result.stdout.splitlines() if line.strip()]
    return paths[:limit]
