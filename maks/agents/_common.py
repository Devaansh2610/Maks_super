"""Shared helpers used by every specialist agent and by whatever calls them
(the supervisor's delegate tools in maks/graph/supervisor.py, and the
embedding fast-path in maks/graph/router.py) so there's exactly one place
that knows how to invoke a specialist in isolation and read its handoff
signal.
"""

from __future__ import annotations

import asyncio
import re
from typing import Callable

from groq import APIError as GroqAPIError
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage, SystemMessage, trim_messages

from maks import jobs
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
_TRANSIENT_TOOL_CHOICE_MARKERS = (
    "tool choice is none",
    # gpt-oss was trained with its own built-in browser/web_search tool, and
    # intermittently calls *that* one from memory instead of the one it was
    # actually given — with the remembered signature (top_n, recency_days,
    # source) rather than ours. Groq rejects it as "not in request.tools".
    # Verified probabilistic, not structural: the identical agent, tools and
    # model succeeded on retry and in isolation, so retrying is the right
    # response rather than trying to out-engineer it.
    "tool call validation failed",
    "was not in request.tools",
)
# 3, not 2: these model quirks are independent per attempt, so a second
# retry meaningfully improves the odds of a long background job surviving —
# and a background job failing is far more annoying than a foreground one,
# since the user walked away expecting an answer to arrive.
_MAX_SPECIALIST_ATTEMPTS = 3

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


def _say_in_background(message: str) -> None:
    """Schedule `message` to be spoken without blocking the caller.

    speak() itself is synchronous (blocking HTTP call + blocking audio
    playback), so it always runs via asyncio.to_thread here — calling it
    directly would stall the persistent event loop that every MCP-backed
    tool call also depends on (see maks/runtime.py). Overlapping calls are
    safe: speak() serializes them behind its own lock (maks/voice/tts.py),
    so two announcements landing at once queue instead of garbling.
    """
    task = asyncio.create_task(asyncio.to_thread(speak, message))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


def announce_delegation(agent_name: str, message: str) -> None:
    """Fire off a spoken "working on it" line the moment the supervisor
    hands work to a slower agent, instead of staying silent until the whole
    delegated task finishes (which, for the coder agent's Claude Code
    handoff, can be minutes). Non-blocking: publishes a dashboard event and
    schedules the speech, so the caller (a delegate tool in
    maks/graph/supervisor.py, or a post_model_hook like coder_agent's)
    continues immediately rather than waiting for it to finish being spoken.
    """
    bus.publish("handoff", target=agent_name, task=message)
    _say_in_background(message)


def announce_unprompted(agent_name: str, message: str) -> None:
    """Speak something the user did *not* just ask for — specifically, a
    background job finishing on its own schedule (see maks/jobs.py). Same
    non-blocking push as announce_delegation, but a distinct event type so
    the dashboard can style "here's news, unprompted" differently from
    "working on what you just asked".

    This is the piece that makes dispatch-and-move-on actually usable: the
    user is told when work they walked away from lands, instead of having to
    remember to ask.
    """
    bus.publish("notice", target=agent_name, task=message)
    _say_in_background(message)


def dynamic_prompt(role_instructions: str) -> Callable[[dict], list]:
    def _prompt(state: dict) -> list:
        persona = settings.read_system_prompt()
        system = f"{persona}\n\n---\nYour specific role right now:\n{role_instructions}"
        return [SystemMessage(content=system)] + state["messages"]

    return _prompt


# Markers of a tool call the model *described in prose* instead of actually
# emitting as a structured call — a known gpt-oss failure mode (see
# GROQ_MODEL). Harmless as a one-off; poisonous once checkpointed, because
# the model then reads it back as an example of its own past output and
# copies the format, and every repeat makes the next turn likelier to repeat
# it too. Observed in practice: a specialist that had stopped calling tools
# entirely and only narrated their names.
_TEXT_TOOL_CALL_MARKERS = ("<function=", "<tool_call>", "</tool_call>", "<|tool")
_TEXT_TOOL_CALL_RE = re.compile(r'^\s*[\w.-]+\s*\{\s*"', re.MULTILINE)


def looks_like_text_tool_call(content: object) -> bool:
    """True if `content` looks like a tool call rendered as text rather than
    issued properly. Kept deliberately narrow — these markers effectively
    never occur in real spoken replies, so false positives are unlikely.
    """
    if not isinstance(content, str):
        return False
    lowered = content.lower()
    if any(marker in lowered for marker in _TEXT_TOOL_CALL_MARKERS):
        return True
    return bool(_TEXT_TOOL_CALL_RE.search(content))


def sanitize_history(messages: list) -> list:
    """Drop assistant turns that are really malformed tool calls in disguise.

    This runs on the way *into* the model (via the pre_model_hook below), not
    on the stored history — the checkpoint keeps the honest record, the model
    just never sees the bad examples. Doing it here rather than at write time
    is what makes it heal threads that are *already* poisoned, without
    rewriting or deleting anyone's saved conversation.
    """
    kept = []
    for m in messages:
        is_ai = getattr(m, "type", None) == "ai"
        # Never drop an AIMessage that carries real tool_calls, even if its
        # prose also trips the detector: its ToolMessage replies reference it
        # by id, and removing it would leave them orphaned — which most
        # providers reject outright. A genuinely malformed turn has the call
        # only in the text, so it has no tool_calls to orphan.
        has_real_calls = bool(getattr(m, "tool_calls", None))
        if is_ai and not has_real_calls and looks_like_text_tool_call(getattr(m, "content", "")):
            continue
        kept.append(m)
    return kept


def make_trim_hook(max_tokens: int) -> Callable[[dict], dict]:
    """pre_model_hook factory for a specialist's own checkpointed history —
    same trim_messages pattern maks/graph/supervisor.py's router uses, just
    with a caller-chosen budget. Only matters once a specialist has a
    checkpointer + stable thread_id (see maks/graph/state.py's
    SPECIALIST_THREAD_IDS) and can actually accumulate turns; harmless
    no-op otherwise (a short history is never trimmed).

    Also sanitizes (see sanitize_history) — a sticky thread is exactly where
    a single malformed turn would otherwise compound.
    """

    def _trim(state: dict) -> dict:
        trimmed = trim_messages(
            sanitize_history(state["messages"]),
            max_tokens=max_tokens,
            strategy="last",
            token_counter="approximate",
            start_on="human",
            include_system=True,
        )
        return {"llm_input_messages": trimmed}

    return _trim


# Human-readable phrasing for the tools a background job commonly fires, so
# a progress update says "reading a web page" rather than "read_web_page".
# deepagents' own tools are in here too — `task` is the sub-agent spawn, and
# is what makes "3 sub-agents running" reportable at all.
_ACTIVITY_PHRASES = {
    "search_the_web": "searching the web",
    "web_search": "searching the web",
    "read_web_page": "reading a web page",
    "fetch_page": "reading a web page",
    "write_todos": "planning out the work",
    "task": "delegating to a sub-agent",
    "write_file": "saving findings",
    "read_file": "reviewing its notes",
    "edit_file": "updating its notes",
    "ls": "checking its notes",
    "glob": "looking through its notes",
    "grep": "searching its notes",
    "run_claude_code": "working in Claude Code",
}


class JobActivityHandler(BaseCallbackHandler):
    """Reports a background job's tool calls into the job registry as they
    happen (see maks/jobs.py's note_activity).

    A callback handler rather than switching the call to `astream`, because
    callbacks propagate automatically into nested runs — which is exactly
    what's needed to see what a deep researcher's *sub-agents* are doing,
    not just the top-level agent. It also leaves invoke_specialist's retry
    logic untouched, which streaming would have complicated.
    """

    def __init__(self, job_id: str) -> None:
        self.job_id = job_id

    def on_tool_start(self, serialized: dict, input_str: str, **kwargs: object) -> None:
        try:
            name = (serialized or {}).get("name") or "a tool"
            phrase = _ACTIVITY_PHRASES.get(name, f"using {name}")
            jobs.note_activity(self.job_id, phrase, is_subagent=(name == "task"))
        except Exception:  # noqa: BLE001 — progress reporting must never break the job it reports on
            pass


async def invoke_specialist(
    agent, task: str, thread_id: str | None = None, job_id: str | None = None
) -> tuple[str, bool]:
    """Runs `agent` on a single task (one new HumanMessage) and returns
    (answer_text, needs_handoff).

    When `thread_id` is None (the default), this is fully isolated — the
    specialist only ever sees this one HumanMessage, no shared conversation
    history, exactly as before sticky sessions existed. When given a
    thread_id (see maks/graph/state.py's SPECIALIST_THREAD_IDS), the
    specialist's own checkpointer (threaded into build_*_agent — see
    maks/graph/supervisor.py's _assemble_graph) resumes that thread's prior
    turns automatically; this call still only appends the new HumanMessage,
    the checkpointer supplies everything before it.

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
    config: dict = {}
    if thread_id:
        config["configurable"] = {"thread_id": thread_id}
    if job_id:
        # Callbacks propagate into nested runs, so this sees sub-agent tool
        # calls too — see JobActivityHandler.
        config["callbacks"] = [JobActivityHandler(job_id)]
    config = config or None

    last_exc: GroqAPIError | None = None
    answer: str | None = None
    for attempt in range(_MAX_SPECIALIST_ATTEMPTS):
        try:
            result = await agent.ainvoke({"messages": [HumanMessage(content=task)]}, config=config)
        except GroqAPIError as exc:
            lowered = str(exc).lower()
            if not any(marker in lowered for marker in _TRANSIENT_TOOL_CHOICE_MARKERS):
                raise
            last_exc = exc
            continue

        candidate = result["messages"][-1].content
        # A "reply" that's really a tool call the model typed out as prose is
        # not an answer — relaying it would hand the user raw
        # <function=...> noise, and returning it as a completed background
        # job's result would be worse still. Retry once; sanitize_history
        # keeps the bad turn from teaching the next attempt to repeat it.
        if looks_like_text_tool_call(candidate) and attempt < _MAX_SPECIALIST_ATTEMPTS - 1:
            continue
        answer = candidate
        break
    else:
        if last_exc is not None:
            raise last_exc  # exhausted retries, still failing

    if answer is None or looks_like_text_tool_call(answer):
        return (
            "I couldn't complete that one — my tools didn't fire correctly. "
            "Worth trying again.",
            False,
        )

    needs_handoff = False
    lines = answer.rstrip().splitlines()
    if lines and lines[-1].strip().startswith(NEEDS_HANDOFF_PREFIX):
        needs_handoff = True
        answer = "\n".join(lines[:-1]).rstrip()

    return answer, needs_handoff
