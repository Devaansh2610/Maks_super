"""Thread-safe pub/sub bus bridging the (synchronous) voice loop thread and
the (async) FastAPI WebSocket handlers that drive the dashboard.

The voice loop runs in a plain background thread and calls `bus.publish(...)`
synchronously. The FastAPI app calls `bus.bind_loop(asyncio.get_running_loop())`
on startup, then each WebSocket connection calls `bus.subscribe()` and awaits
its own queue.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class Event:
    type: str
    data: dict[str, Any] = field(default_factory=dict)
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "data": self.data, "ts": self.ts}


class EventBus:
    def __init__(self) -> None:
        self._subscribers: list[asyncio.Queue] = []
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self.last_status: dict[str, Any] = {"state": "idle", "agent": None}

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def publish(self, event_type: str, **data: Any) -> None:
        """Safe to call from any thread, including the voice loop thread."""
        if event_type in ("state", "listening", "speaking", "thinking", "idle", "ready", "wake"):
            self.last_status["state"] = event_type
        if "agent" in data:
            self.last_status["agent"] = data["agent"]

        event = Event(type=event_type, data=data)
        with self._lock:
            subs = list(self._subscribers)

        if self._loop is None:
            return
        for q in subs:
            self._loop.call_soon_threadsafe(q.put_nowait, event)


bus = EventBus()
