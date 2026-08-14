"""Companion MCP server that runs on the Mac and exposes a small,
authenticated set of tools that Maks (on the Ubuntu box) calls to
remote-control this machine, over MCP's streamable-HTTP transport.

Run directly:   python mac_agent.py
Run on login:   see install/install.sh (sets up a launchd job that does this)
"""

from __future__ import annotations

import os

import uvicorn
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.routing import Route

from actions import browser, files, spotify
from actions import system_info as system_info_module

load_dotenv()

TOKEN = os.environ.get("MAC_COMPANION_TOKEN", "")
HOST = os.environ.get("MAC_COMPANION_HOST", "0.0.0.0")
PORT = int(os.environ.get("MAC_COMPANION_PORT", "8765"))

mcp = FastMCP("Maks Mac Companion", host=HOST, port=PORT)


@mcp.tool()
def play_youtube(query: str) -> str:
    """Play a video on YouTube on this Mac (opens/searches YouTube in the
    default browser for the given query, e.g. a song or video title).
    """
    return browser.play_youtube(query)


@mcp.tool()
def play_spotify(query: str) -> str:
    """Play a song/artist/album on Spotify on this Mac (Spotify app must be
    installed; searches and starts playing the best match).
    """
    return spotify.play(query)


@mcp.tool()
def spotify_control(action: str) -> str:
    """Control Spotify playback on this Mac. action must be one of:
    'pause', 'play', 'next', 'previous'.
    """
    if action not in {"pause", "play", "next", "previous"}:
        return "Invalid action. Use one of: pause, play, next, previous."
    return spotify.control(action)


@mcp.tool()
def search_files(query: str, root: str = "") -> str:
    """Search for files by name on this Mac using Spotlight (mdfind).
    Optionally restrict the search to a root directory (e.g. '~/Documents').
    """
    results = files.search(query, root or None)
    if not results:
        return "No matching files found."
    preview = "\n".join(results[:10])
    return f"Found {len(results)} file(s):\n{preview}"


@mcp.tool()
def system_info() -> str:
    """Get a snapshot of this Mac: CPU/RAM/disk usage, battery, and top
    running applications. Use this when asked to 'check'/'analyze' the Mac.
    """
    data = system_info_module.get_system_info()
    return (
        f"CPU: {data.get('cpu_percent')}% | "
        f"RAM: {data.get('memory_percent')}% used | "
        f"Disk: {data.get('disk_percent')}% used | "
        f"Battery: {data.get('battery_percent', 'n/a')}% | "
        f"Top apps: {', '.join(data.get('top_apps', []))}"
    )


async def _health(request: Request) -> PlainTextResponse:
    return PlainTextResponse("ok")


class TokenAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/health":
            return await call_next(request)
        if not TOKEN:
            return PlainTextResponse("MAC_COMPANION_TOKEN is not configured on this Mac", status_code=500)
        if request.headers.get("X-Maks-Token") != TOKEN:
            return PlainTextResponse("unauthorized", status_code=401)
        return await call_next(request)


app = mcp.streamable_http_app()
app.router.routes.insert(0, Route("/health", _health))
app.add_middleware(TokenAuthMiddleware)


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
