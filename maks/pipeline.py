"""Runs a single user utterance through Maks' routing and returns the reply,
publishing progress events along the way. Shared by the voice loop
(main.py) and the dashboard's text-input fallback (server/app.py) so both
paths behave identically.

Two routing paths, tried in order:
1. Embedding fast path (maks/graph/router.py): a confident, single-intent
   utterance goes straight to the matching specialist — no LLM call at all
   for the routing decision itself. This is also the "sticky" path: every
   confident match calls invoke_specialist with that agent's stable
   SPECIALIST_THREAD_IDS thread id (see maks/graph/state.py), so a
   specialist that's been talked to before actually remembers it via its
   own checkpointer (threaded in from build_supervisor_graph) — there's no
   separate "stay with the same agent" mechanism; a cheap re-check of the
   embedding router on every turn (see _active_agent below) is what decides
   whether continuity applies, not a sticky flag that has to be explicitly
   broken out of. Fast/sticky-routed turns are NOT added to the router's
   own top-level checkpointed conversation (a deliberate, known trade-off —
   see the module docstring in maks/graph/supervisor.py's plan notes): the
   specialist's own persisted thread is what carries continuity for that
   agent's conversation, not the router's.
2. Full supervisor graph (maks/graph/supervisor.py): anything ambiguous or
   compound, so it can actually reason about routing and self-correct.

Async because the underlying tools are MCP-backed (see maks/mcp_client.py,
maks/runtime.py) — callers on plain sync threads should go through
maks.runtime.run_coro(handle_command(text)) rather than awaiting this
directly, unless they're already running inside the persistent loop.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from maks import jobs
from maks.agents._common import invoke_specialist
from maks.events import bus
from maks.graph.router import route
from maks.graph.state import DEFAULT_THREAD_ID, SPECIALIST_THREAD_IDS

_graph = None
_specialists: dict[str, object] = {}

# Best-effort record of which specialist last answered — not persisted
# across restarts (each specialist's own checkpointed thread already is,
# see SPECIALIST_THREAD_IDS, so continuity itself survives a restart; this
# variable only affects dashboard/debugging visibility of "who's active"
# and isn't currently read by any routing decision — route() re-checks the
# embedding router fresh every turn rather than trusting a sticky flag).
_active_agent: str | None = None


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
    global _active_agent

    text = text.strip()
    if not text:
        return ""

    # Tells background-job heartbeats to stay quiet for a bit — see
    # maks/jobs.py's seconds_since_user_activity. Marked on the way *in*, so
    # a progress update can't land in the gap between the user speaking and
    # Maks answering.
    jobs.mark_user_active()
    bus.publish("thinking", text=text)

    # Make sure specialists exist before trying the fast path — they're
    # built once as part of the supervisor graph either way.
    await get_graph()

    decision = await route(text)
    if decision.agent is not None and decision.agent in _specialists:
        thread_id = SPECIALIST_THREAD_IDS.get(decision.agent)
        reply, _needs_handoff = await invoke_specialist(_specialists[decision.agent], text, thread_id=thread_id)
        _active_agent = decision.agent
        bus.publish("reply", text=reply, agent=decision.agent)
        return reply

    graph = await get_graph()
    config = {"configurable": {"thread_id": DEFAULT_THREAD_ID}}

    result = await graph.ainvoke({"messages": [HumanMessage(content=text)]}, config=config)
    messages = result.get("messages", [])
    reply = messages[-1].content if messages else "Sorry, I didn't get a response."
    agent_name = _extract_agent_name(messages)
    _active_agent = agent_name

    bus.publish("reply", text=reply, agent=agent_name)
    return reply
