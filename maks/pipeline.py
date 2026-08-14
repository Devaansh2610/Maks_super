"""Runs a single user utterance through Maks' routing and returns the reply,
publishing progress events along the way. Shared by the voice loop
(main.py) and the dashboard's text-input fallback (server/app.py) so both
paths behave identically.

Two routing paths, tried in order:
1. Embedding fast path (maks/graph/router.py): a confident, single-intent
   utterance goes straight to the matching specialist — no LLM call at all
   for the routing decision itself. Fast-pathed turns are NOT added to the
   supervisor's checkpointed conversation history (specialists are already
   stateless/isolated by design — see maks/agents/_common.py's
   invoke_specialist — so this only affects whether a *later*
   supervisor-routed turn can see that a fast-pathed turn happened; a
   deliberate simplicity trade-off, not an oversight).
2. Full supervisor graph (maks/graph/supervisor.py): anything ambiguous or
   compound, so it can actually reason about routing and self-correct.

Async because the underlying tools are MCP-backed (see maks/mcp_client.py,
maks/runtime.py) — callers on plain sync threads should go through
maks.runtime.run_coro(handle_command(text)) rather than awaiting this
directly, unless they're already running inside the persistent loop.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from maks.agents._common import invoke_specialist
from maks.events import bus
from maks.graph.router import route
from maks.graph.state import DEFAULT_THREAD_ID

_graph = None
_specialists: dict[str, object] = {}


async def get_graph():
    global _graph, _specialists
    if _graph is None:
        from maks.graph.supervisor import build_supervisor_graph

        _graph, _specialists = await build_supervisor_graph()
    return _graph


def _extract_agent_name(messages: list) -> str | None:
    """Best-effort "who actually handled this" for the dashboard's agent
    label. The supervisor is a single agent now (supervisor-as-tools), so
    there's no per-message `.name` tag to read — instead, scan this turn's
    messages (from the most recent HumanMessage onward) for an `ask_*`
    delegate tool call and report the last one used. None means the
    supervisor answered directly with no delegation.
    """
    start = 0
    for i in range(len(messages) - 1, -1, -1):
        if getattr(messages[i], "type", None) == "human":
            start = i
            break

    agent_name = None
    for msg in messages[start:]:
        for call in getattr(msg, "tool_calls", None) or []:
            name = call.get("name", "")
            if name.startswith("ask_"):
                agent_name = name[len("ask_") :]
    return agent_name


async def handle_command(text: str) -> str:
    """Send `text` through the fast-path router (or, on a miss, the full
    supervisor graph), return Maks' final reply.
    """
    text = text.strip()
    if not text:
        return ""

    bus.publish("thinking", text=text)

    # Make sure specialists exist before trying the fast path — they're
    # built once as part of the supervisor graph either way.
    await get_graph()

    decision = await route(text)
    if decision.agent is not None and decision.agent in _specialists:
        reply, _needs_handoff = await invoke_specialist(_specialists[decision.agent], text)
        bus.publish("reply", text=reply, agent=decision.agent)
        return reply

    graph = await get_graph()
    config = {"configurable": {"thread_id": DEFAULT_THREAD_ID}}

    result = await graph.ainvoke({"messages": [HumanMessage(content=text)]}, config=config)
    messages = result.get("messages", [])
    reply = messages[-1].content if messages else "Sorry, I didn't get a response."
    agent_name = _extract_agent_name(messages)

    bus.publish("reply", text=reply, agent=agent_name)
    return reply
