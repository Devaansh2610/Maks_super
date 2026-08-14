"""Builds and holds persistent MCP sessions for every Maks integration, and
groups the resulting tools by which agent should get them.

Must be initialized (via `init()`) from inside maks.runtime's persistent
event loop — the AsyncExitStack it opens has to stay alive for the whole
process lifetime, exactly like the stdio subprocesses / HTTP session it
wraps (see maks/runtime.py for why; verified empirically that a session
opened once and reused is ~100x faster than reconnecting per call).
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

from maks.settings import settings

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_exit_stack: contextlib.AsyncExitStack | None = None
_tools_by_agent: dict[str, list[BaseTool]] = {}


def _stdio(module_name: str) -> dict:
    # Launched as `-m maks.mcp_servers.<module_name>` (not a bare script path)
    # so the subprocess can resolve `from maks... import ...` — a script path
    # alone doesn't put the project root on that subprocess's sys.path.
    return {
        "command": sys.executable,
        "args": ["-m", f"maks.mcp_servers.{module_name}"],
        "cwd": str(PROJECT_ROOT),
        "transport": "stdio",
    }


def _server_configs() -> dict:
    return {
        "connectors": _stdio("api_connectors"),
        "mac": {
            "url": f"{settings.mac_companion_url.rstrip('/')}/mcp",
            "transport": "streamable_http",
            "headers": {"X-Maks-Token": settings.mac_companion_token},
            # Default is 30s (mcp.client.streamable_http.streamablehttp_client's
            # own default) -- far too slow for an unreachable companion, which
            # is the common case (see settings.mac_companion_connect_timeout).
            "timeout": settings.mac_companion_connect_timeout,
        },
    }


_CODER_TOOL_NAMES = {"run_claude_code"}
_OFFICE_TOOL_NAMES = {
    "web_search", "fetch_page",
    "gmail_search", "gmail_send", "calendar_list_events", "calendar_create_event",
    "whatsapp_send", "slack_send", "outlook_send",
    "notion_search", "notion_create_page", "notion_append_text",
    "notion_get_page_content", "notion_query_database",
}


def _bucket_tools(connector_tools: list[BaseTool], mac_tools: list[BaseTool]) -> dict[str, list[BaseTool]]:
    """Shared by init() and load_tools_stateless() below — the only
    difference between them is *how* the tools were fetched (persistent
    session vs. one-shot), not how they get grouped by agent.
    """
    return {
        # weather_lookup goes straight to the supervisor (cheap, frequently
        # needed for casual conversation); office_agent gets communications
        # + research tools; coder_agent gets the Claude handoff.
        "supervisor": [t for t in connector_tools if t.name == "weather_lookup"],
        "office_agent": [t for t in connector_tools if t.name in _OFFICE_TOOL_NAMES],
        "coder_agent": [t for t in connector_tools if t.name in _CODER_TOOL_NAMES],
        # chit_chat_agent is deliberately tool-less ("just the LLM").
        "chit_chat_agent": [],
        "mac_control_agent": mac_tools,
    }


async def init() -> None:
    """Open a persistent session to every configured MCP server and load
    their tools, grouped by which agent uses them. A server that's
    unreachable (e.g. Mac companion not running, or Google/Notion not
    configured yet) just leaves that agent with an empty tool list rather
    than crashing startup — the agent will explain plainly that it can't help
    with that yet. Safe to call more than once (no-ops after the first).

    Only for the real app (maks/pipeline.py, via maks/runtime.py's one
    persistent background loop). Do NOT call this from
    maks/graph/supervisor.py's make_graph() — see load_tools_stateless()
    below for why.
    """
    global _exit_stack, _tools_by_agent
    if _exit_stack is not None:
        return

    _exit_stack = contextlib.AsyncExitStack()
    client = MultiServerMCPClient(_server_configs())

    async def load(server_name: str) -> list[BaseTool]:
        try:
            session = await _exit_stack.enter_async_context(client.session(server_name))
            return await load_mcp_tools(session)
        except Exception as exc:  # noqa: BLE001
            print(f"[mcp_client] Couldn't connect to the '{server_name}' MCP server: {exc}")
            return []

    connector_tools = await load("connectors")
    mac_tools = await load("mac")
    _tools_by_agent = _bucket_tools(connector_tools, mac_tools)


def get_tools(agent_name: str) -> list[BaseTool]:
    if _exit_stack is None:
        raise RuntimeError("mcp_client.init() must be called before get_tools()")
    return _tools_by_agent.get(agent_name, [])


async def load_tools_stateless() -> dict[str, list[BaseTool]]:
    """One-shot alternative to init()/get_tools(), for contexts that can't
    guarantee a single persistent task/cancel-scope to hang a long-lived
    AsyncExitStack off of — specifically `langgraph dev`'s in-memory API
    server, which calls maks/graph/supervisor.py's make_graph() factory from
    a new short-lived per-request task on every API call (confirmed: its own
    "Slow graph load" warnings fire on every schema/graph/history request,
    not just once at startup). init()'s persistent sessions were getting
    torn down when the particular request task that opened them ended,
    leaving _tools_by_agent pointing at closed streams — the next real tool
    call then raised anyio.ClosedResourceError.

    Uses MultiServerMCPClient.get_tools() instead, which opens (and closes)
    a fresh session per tool *call*, not per tool *list* — slower per call
    (no persistent-connection speedup, same ~1.3s-per-call cost init()
    exists to avoid) but immune to this failure mode, which matters far more
    for a dev/test tool than raw latency does.
    """
    client = MultiServerMCPClient(_server_configs())
    connector_tools = await client.get_tools(server_name="connectors")
    try:
        mac_tools = await client.get_tools(server_name="mac")
    except Exception as exc:  # noqa: BLE001
        print(f"[mcp_client] Couldn't connect to the 'mac' MCP server: {exc}")
        mac_tools = []
    return _bucket_tools(connector_tools, mac_tools)


async def shutdown() -> None:
    global _exit_stack
    if _exit_stack is not None:
        await _exit_stack.aclose()
        _exit_stack = None
