"""Shared helpers used by every specialist agent and by whatever calls them
(the supervisor's delegate tools in maks/graph/supervisor.py, and the
embedding fast-path in maks/graph/router.py) so there's exactly one place
that knows how to invoke a specialist in isolation and read its handoff
signal.
"""

from __future__ import annotations

import asyncio
from typing import Callable

from groq import APIError as GroqAPIError
from langchain_core.messages import HumanMessage, SystemMessage

from maks.events import bus
from maks.settings import settings
from maks.voice.tts import speak

# gpt-oss-20b (see GROQ_MODEL) occasionally emits a tool call in a way that
# trips Groq's own server-side request validation ("Tool choice is none, but
# model called a tool") — confirmed this isn't caused by anything in this
# codebase (create_react_agent never sets tool_choice anywhere here, and the
# exact same failing task succeeded immediately on a plain direct retry
# outside any special context) — it's an intermittent quirk of gpt-oss's
# tool-call format on Groq, not a deterministic bug.
_TRANSIENT_TOOL_CHOICE_MARKER = "tool choice is none"
_MAX_SPECIALIST_ATTEMPTS = 2

# Holds references to fire-and-forget announcement tasks (see
# announce_delegation below) so asyncio doesn't garbage-collect a task while
# it's still running — a task with no other reference is only weakly held.
_background_tasks: set[asyncio.Task] = set()

# A specialist ends its final answer with this exact line only when
# completing the user's full request needs a different agent it can't reach
# itself. Absence of the marker means "this answer is complete" — see
# invoke_specialist() below and maks/graph/supervisor.py's _delegate_tool,
# which uses that to skip an extra supervisor LLM call and end the turn
# immediately instead of always bouncing back for a wrap-up it isn't
# changing anyway.
NEEDS_HANDOFF_PREFIX = "NEEDS_HANDOFF:"

# Appended to every specialist's ROLE so they all describe the sentinel the
# same way. `other_agents` should name the other delegate agents this one
# can plausibly hand off to (e.g. "office_agent or coder_agent").
HANDOFF_INSTRUCTION = (
    "\n\nIf, and only if, fully answering the user's request needs a "
    "different specialist you cannot reach yourself ({other_agents}), end "
    "your reply with exactly one line: 'NEEDS_HANDOFF: <agent name>: <short "
    "reason>'. Otherwise never write that line — most answers are complete "
    "and should end normally."
)


def announce_delegation(agent_name: str, message: str) -> None:
    """Fire off a spoken "working on it" line the moment the supervisor
    hands work to a slower agent, instead of staying silent until the whole
    delegated task finishes (which, for the coder agent's Claude Code
    handoff, can be minutes). Non-blocking: publishes a dashboard event and
    schedules the actual speech as a background task on the current event
    loop, so the caller (a delegate tool in maks/graph/supervisor.py, or a
    post_model_hook like coder_agent's) continues immediately rather than
    waiting for the announcement to finish being spoken.

    speak() itself is synchronous (blocking HTTP call + blocking audio
    playback), so it always runs via asyncio.to_thread here — calling it
    directly would stall the persistent event loop that every MCP-backed
    tool call also depends on (see maks/runtime.py).
    """
    bus.publish("handoff", target=agent_name, task=message)

    task = asyncio.create_task(asyncio.to_thread(speak, message))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


def dynamic_prompt(role_instructions: str) -> Callable[[dict], list]:
    def _prompt(state: dict) -> list:
        persona = settings.read_system_prompt()
        system = f"{persona}\n\n---\nYour specific role right now:\n{role_instructions}"
        return [SystemMessage(content=system)] + state["messages"]

    return _prompt


async def invoke_specialist(agent, task: str) -> tuple[str, bool]:
    """Runs `agent` on a single isolated task (no shared conversation
    history — the specialist only ever sees this one HumanMessage) and
    returns (answer_text, needs_handoff).

    `agent` just needs to expose `.ainvoke({"messages": [...]}) ->
    {"messages": [...]}` — true for both a create_react_agent-built
    specialist and a hand-built compiled StateGraph (e.g. the Mac
    DynamicWorker in maks/graph/dynamic_worker.py), so this one helper works
    for all of them.

    Retries once (see _MAX_SPECIALIST_ATTEMPTS) on the specific transient
    Groq tool-choice error described above. Known, accepted trade-off: a
    retry re-runs the *whole* specialist from scratch, so if the failure
    happens after a non-idempotent tool already succeeded (gmail_send,
    calendar_create_event, notion_create_page, run_claude_code, ...), that
    action could run twice. This is deliberately narrow (exact error message
    match, not a broad catch-and-retry) and bounded (one retry) to keep that
    risk small — a bare failure on every occurrence of a rare model quirk
    was judged worse for now. Revisit if this turns out to fire often enough
    for the duplicate-side-effect risk to matter in practice.
    """
    last_exc: GroqAPIError | None = None
    for attempt in range(_MAX_SPECIALIST_ATTEMPTS):
        try:
            result = await agent.ainvoke({"messages": [HumanMessage(content=task)]})
            break
        except GroqAPIError as exc:
            if _TRANSIENT_TOOL_CHOICE_MARKER not in str(exc).lower():
                raise
            last_exc = exc
            continue
    else:
        raise last_exc  # exhausted retries, still failing

    answer = result["messages"][-1].content

    needs_handoff = False
    lines = answer.rstrip().splitlines()
    if lines and lines[-1].strip().startswith(NEEDS_HANDOFF_PREFIX):
        needs_handoff = True
        answer = "\n".join(lines[:-1]).rstrip()

    return answer, needs_handoff
