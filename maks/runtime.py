"""Owns a single persistent background asyncio event loop for the whole app.

MCP client connections (stdio subprocesses, HTTP sessions) are async-only and
must stay open for the life of the process — reconnecting per call costs
~1.3s (measured), a reused session costs ~10ms. This loop is where those
connections live (see mcp_client.py); sync code (the voice loop thread,
FastAPI's request handlers) calls into it via run_coro().
"""

from __future__ import annotations

import asyncio
import threading
from typing import Coroutine, TypeVar

T = TypeVar("T")

_loop: asyncio.AbstractEventLoop | None = None
_thread: threading.Thread | None = None


def start() -> None:
    """Start the background loop thread. Safe to call more than once."""
    global _loop, _thread
    if _loop is not None:
        return

    ready = threading.Event()

    def _run() -> None:
        global _loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        _loop = loop
        ready.set()
        loop.run_forever()

    _thread = threading.Thread(target=_run, daemon=True, name="maks-async-runtime")
    _thread.start()
    ready.wait()


def run_coro(coro: "Coroutine[object, object, T]", timeout: float | None = None) -> T:
    """Schedule `coro` on the persistent loop from any (sync) thread and
    block the calling thread until it completes. Raises whatever `coro` raised.
    """
    if _loop is None:
        raise RuntimeError("runtime.start() must be called before run_coro()")
    future = asyncio.run_coroutine_threadsafe(coro, _loop)
    return future.result(timeout=timeout)
