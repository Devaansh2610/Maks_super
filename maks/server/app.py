"""FastAPI dashboard: serves the web UI, streams live voice/agent activity
over a WebSocket, and offers a text-input fallback that goes through the
exact same pipeline as spoken commands.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from maks import runtime
from maks.events import bus
from maks.settings import SYSTEM_PROMPT_PATH, settings

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Maks")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class ChatRequest(BaseModel):
    text: str


class SystemPromptRequest(BaseModel):
    content: str


class NarrateRequest(BaseModel):
    agent: str
    message: str


@app.on_event("startup")
async def _bind_event_loop() -> None:
    bus.bind_loop(asyncio.get_running_loop())


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/status")
async def status() -> JSONResponse:
    return JSONResponse(bus.last_status)


@app.get("/jobs")
async def list_jobs() -> JSONResponse:
    """Background jobs dispatched this session (see maks/jobs.py) — what's
    running, what finished, results, and errors. The supervisor reads the
    same registry through its check_background_jobs tool; this is the
    dashboard's (and a human debugger's) direct view of it, without going
    through the model.
    """
    from maks import jobs

    return JSONResponse(
        [
            {
                "id": j.id,
                "agent": j.agent,
                "task": j.task,
                "status": j.status,
                "result": j.result,
                "error": j.error,
                "elapsed_seconds": round(j.elapsed_seconds, 1),
            }
            for j in jobs.all_jobs()
        ]
    )


@app.get("/system-prompt")
async def get_system_prompt() -> JSONResponse:
    return JSONResponse({"content": settings.read_system_prompt()})


@app.post("/system-prompt")
async def save_system_prompt(req: SystemPromptRequest) -> JSONResponse:
    SYSTEM_PROMPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SYSTEM_PROMPT_PATH.write_text(req.content, encoding="utf-8")
    bus.publish("system_prompt_updated")
    return JSONResponse({"ok": True})


@app.post("/internal/narrate")
async def internal_narrate(req: NarrateRequest) -> JSONResponse:
    """Live-progress bridge for run_claude_code (maks/mcp_servers/
    api_connectors.py), which runs in a separate OS process and so can't
    reach this process's event bus/speaker directly. Not a general-purpose
    endpoint -- only ever called from that one tool, over the loopback
    address it hardcodes. No auth, same as every other endpoint on this
    dashboard: it's a single-user LAN app, not something exposed to
    untrusted networks.

    Reuses announce_delegation exactly like every other in-process
    "handing off" narration (see maks/agents/_common.py) -- publishes a bus
    event for the dashboard and speaks the message as a background task,
    without blocking this request.
    """
    from maks.agents._common import announce_delegation

    announce_delegation(req.agent, req.message)
    return JSONResponse({"ok": True})


@app.post("/chat")
async def chat(req: ChatRequest) -> JSONResponse:
    from maks.pipeline import handle_command

    # handle_command's graph lives on maks.runtime's persistent event loop
    # (that's where the MCP server connections are held open), not this
    # request's own loop — dispatch there and wait for the result off-thread
    # so this endpoint doesn't block FastAPI's loop while it runs.
    reply = await asyncio.to_thread(runtime.run_coro, handle_command(req.text))
    return JSONResponse({"reply": reply})


@app.websocket("/ws")
async def ws_events(websocket: WebSocket) -> None:
    await websocket.accept()
    queue = bus.subscribe()
    try:
        await websocket.send_json({"type": "status", "data": bus.last_status, "ts": None})
        while True:
            event = await queue.get()
            await websocket.send_json(event.to_dict())
    except WebSocketDisconnect:
        pass
    finally:
        bus.unsubscribe(queue)
