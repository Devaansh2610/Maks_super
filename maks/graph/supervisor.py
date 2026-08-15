"""Builds the subgraph-visible supervisor graph: a router node that delegates
to real specialist *nodes* (via `Command(graph=Command.PARENT, goto=...)`)
instead of calling them from inside a tool's coroutine. Chosen specifically
so LangGraph Studio can render every specialist's own internal nodes as an
expandable subgraph — verified by reading the installed LangGraph package's
actual subgraph-detection code: Studio walks each node's *closure* via
static AST analysis, finding any bare-name-referenced compiled graph a node
function calls (e.g. `agent.ainvoke(...)` inside `_specialist_node` below).
This means every specialist file (`maks/agents/chit_chat_agent.py`,
`office_agent.py`, `coder_agent.py`, `maks/graph/dynamic_worker.py`) stays
completely unchanged — only this file's wiring changed, from tool-calling to
real graph nodes/edges.

Still not `langgraph_supervisor.create_supervisor` (its supervisor node
replays the *entire* shared conversation into every call and injects
synthetic "transferring to X"/"transferred back" messages into the
checkpointed history on every handoff — verified by reading its source) and
still not `langgraph-swarm` (peer-to-peer handoffs would re-share a growing
conversation thread across agents, undoing the isolated-task-string design
`invoke_specialist` gives every specialist). `Command(graph=Command.PARENT,
...)` is the same handoff primitive swarm's own tools use internally,
applied here directly without the dependency.

Classic supervisor control flow, by deliberate choice: a specialist's result
routes back through the router (`after_specialist -> router`) so the router
reviews it before answering — one extra LLM call per delegation, in exchange
for the router usually getting a chance to catch a wrong/incomplete answer
before it reaches the user, rather than specialists silently ending the turn
themselves. Bounded, not unconditional: `MAX_DELEGATIONS_PER_TURN` caps how
many specialist calls one turn can rack up before `_finalize` forces a reply
straight from the last specialist's own answer — otherwise the router's own
"is this good enough?" judgment call is the only thing standing between one
delegation and an unbounded chain of them, which in practice combined badly
with Groq's free-tier rate limit (a re-delegation right as the limit was hit
turned into the whole turn silently retrying with backoff for the better
part of a minute). A specialist's result comes back to the router labeled as
a finding to review (see `_after_specialist`), not injected as if the router
already said it — otherwise the model reading its own history back would get
confused about why it's being asked to speak again immediately after an
apparently-already-complete answer.

Each specialist still only ever sees its own isolated task string via
`invoke_specialist` (maks/agents/_common.py) — nothing about routing through
real graph nodes instead of tool coroutines changes that; `invoke_specialist`
is called exactly the same way either way, just from a plain node function
instead of from inside a `StructuredTool`'s coroutine.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage, trim_messages
from langchain_core.tools import BaseTool, InjectedToolCallId, StructuredTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command

from maks import jobs, mcp_client
from maks.agents._common import (
    announce_delegation,
    announce_unprompted,
    dynamic_prompt,
    invoke_specialist,
    sanitize_history,
)
from maks.agents.chit_chat_agent import build_chit_chat_agent
from maks.agents.coder_agent import build_coder_agent
from maks.agents.deep_research_agent import build_deep_research_agent
from maks.agents.office_agent import build_office_agent
from maks.graph.dynamic_worker import build_mac_dynamic_worker_graph
from maks.graph.state import SPECIALIST_THREAD_IDS
from maks.llm import get_llm
from maks.settings import PROJECT_ROOT, settings

_SPECIALIST_NAMES = ("chit_chat_agent", "office_agent", "coder_agent", "mac_control_agent")

# Caps how many times a single turn can bounce specialist -> router -> a
# (possibly different) specialist before the graph forces a final answer.
# Without this, the router's own judgment call ("is this finding good
# enough?") is the only thing standing between one delegation and an
# unbounded chain of them — observed in practice: the router re-delegated a
# perfectly reasonable office_agent answer because it wasn't phrased exactly
# like the request, and the second specialist call queued up right as the
# free-tier Groq rate limit was hit, so the whole turn sat retrying with
# exponential backoff for the better part of a minute — indistinguishable
# from a hang. 2 preserves the one-self-correction value of "always return
# to router" (see the module docstring) while guaranteeing every turn
# terminates in a bounded number of specialist calls.
MAX_DELEGATIONS_PER_TURN = 2

# Opened once, lazily, and kept open for the process's life — same lifecycle
# pattern maks/mcp_client.py uses for its persistent MCP sessions. Only
# build_supervisor_graph() (the real app's entry point) touches this;
# make_graph() (langgraph dev/Studio) stays on a fresh in-memory MemorySaver
# every call, on purpose — see make_graph()'s docstring.
_checkpointer_stack: contextlib.AsyncExitStack | None = None
_checkpointer: AsyncSqliteSaver | None = None


async def _get_checkpointer() -> AsyncSqliteSaver:
    global _checkpointer_stack, _checkpointer
    if _checkpointer is None:
        _checkpointer_stack = contextlib.AsyncExitStack()
        _checkpointer = await _checkpointer_stack.enter_async_context(
            AsyncSqliteSaver.from_conn_string(str(PROJECT_ROOT / "maks_memory.sqlite"))
        )
        await _checkpointer.setup()
    return _checkpointer


SUPERVISOR_ROLE = """You are Maks, routing every request to the right
specialist. Use weather_lookup yourself for weather questions — that's the
one thing you handle directly. For everything else, delegate:

- ask_chit_chat_agent: greetings, small talk, opinions, jokes, general
  knowledge — plain conversation with no tools involved.
- ask_office_agent: web research, Gmail, Google Calendar, WhatsApp/Slack/
  Outlook messaging, or the user's Notion workspace.
- ask_mac_control_agent: playing YouTube/Spotify on the user's Mac,
  searching files on the Mac, checking/analyzing the Mac's system health.
- ask_coder_agent: writing, editing, debugging, reviewing, or explaining
  code in a real project — always goes to Claude via this tool.

Two special tools change *when* you answer, not who does the work:
- dispatch_background_task: for long-horizon work the user shouldn't have to
  sit and wait through — a deep research sweep (deep_research_agent), or a
  substantial coding job. It starts the work and returns immediately. When
  you use it, say you've kicked it off, mention the job number, and ask what
  else you can help with. Never wait for it, and never invent a result — the
  user is told automatically the moment it finishes. Prefer this whenever a
  request sounds like it'll take minutes rather than seconds. Anything the
  user asks you to "research", "dig into", or "do a deep dive on" belongs
  here, with agent='deep_research_agent'.
- check_background_jobs: whenever the user asks what you're working on, how
  something is going, whether a job is finished, or wants to hear the result
  of earlier background work. Read the answer from this tool, never from
  memory.

Everything quick still goes through the normal ask_ tools above — don't
background a request the user would rather just have answered now.

When you delegate, write `task` as a clear, standalone instruction — the
specialist does not see the rest of this conversation, only what you put in
`task`. After a specialist runs, its finding comes back to you labeled
"[<agent> specialist reported back — review before replying]" — that is NOT
your final reply, it's a draft for you to check. Read it, then either:
- produce your own final reply to the user (relay/confirm the finding in
  your own words — never show the user the raw labeled note itself), or
- delegate again — to the same specialist with a clearer task, or to a
  different one — if it's wrong, incomplete, or the specialist flagged it
  couldn't fully complete the request on its own.

A finding that substantively answers what the user asked is complete, even
if it isn't phrased exactly like the request or doesn't cover every possible
angle — relay it and stop. Only re-delegate for a real problem (wrong,
empty, or an explicit NEEDS_HANDOFF flag), never just to get a more literal
match. Re-delegation costs real time and a scarce rate-limit budget.

Your final reply is spoken out loud, not read — if a finding is long,
technical, or full of raw detail (a big list, a full code diff, a long
email thread), summarize the key point(s) concisely instead of reading it
all back; give the full detail only if the user actually asked for it.

One exception to being transparent about delegation: for chit_chat_agent
specifically, never mention that you routed or delegated anything ("handing
this to chit-chat", "let me check with...") — just answer directly, as if
you'd answered it yourself. For every other specialist (office_agent,
mac_control_agent, coder_agent), it's fine, and expected, to be clear
you're getting that specialist's help."""


class SupervisorState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    task: str
    result: str
    needs_handoff: bool
    agent_key: str
    # Reset to 0 at the start of every turn (see _reset_turn), incremented by
    # each specialist call within that turn (see _specialist_node) — read by
    # _route_after_specialist to enforce MAX_DELEGATIONS_PER_TURN.
    delegation_count: int


def _delegate_tool(name: str, description: str, agent_key: str) -> StructuredTool:
    async def _run(task: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> list[Command]:
        # Two Commands: the first (graph=None, the default) satisfies the
        # pending tool call on *this* graph (the router's own ToolNode
        # requires a matching ToolMessage) — the second jumps to the
        # specialist node on the *parent* graph with `task` set. Both are
        # required; a single combined Command can't target two different
        # graphs at once.
        return [
            Command(update={"messages": [ToolMessage(content=f"Routing to {agent_key}.", tool_call_id=tool_call_id)]}),
            Command(graph=Command.PARENT, goto=agent_key, update={"task": task}),
        ]

    return StructuredTool.from_function(coroutine=_run, name=name, description=description)


# Which specialists are worth dispatching in the background. chit_chat_agent
# and mac_control_agent are excluded deliberately: both answer in seconds
# (small talk; a single device command), so backgrounding them would add a
# "I'll get back to you" round trip to work that would already be done.
# deep_research_agent is background-*only* — see its module docstring.
_BACKGROUNDABLE = ("deep_research_agent", "office_agent", "coder_agent")

# The tools the deep researcher gets, out of the office tool set: finding
# sources and reading them. Deliberately not the whole office kit — a
# research sweep has no business sending mail or writing to Notion, and a
# smaller tool list keeps its many LLM calls cheaper on a tight token budget.
_RESEARCH_TOOL_NAMES = {"web_search", "fetch_page"}

# Strong references to in-flight dispatch tasks, so asyncio doesn't garbage
# collect one mid-run — a task held only weakly can be collected while still
# running. Same reason maks/agents/_common.py keeps its own set.
_dispatched_tasks: set[asyncio.Task] = set()

# Background jobs run one at a time. Not for correctness — they're fully
# independent — but for the token budget: every job is a multi-step agent
# loop, the deep researcher especially so, and two running at once reliably
# blew Groq's free-tier tokens-per-minute ceiling and failed *both* plus
# whatever the user was asking in the foreground. Queueing means a second
# job starts a little later instead of poisoning the first. The user is
# unblocked either way, which is the whole point — they never wait on this.
_job_slot = asyncio.Semaphore(1)


def _friendly_error(exc: Exception) -> str:
    """Turn a provider exception into something worth saying out loud.

    Raw Groq errors are a wall of JSON with org ids and billing links in
    them — fine in a log, useless spoken aloud and alarming on a dashboard.
    Rate limiting in particular isn't really a failure, it's "too much at
    once", and the honest advice differs from a genuine crash.
    """
    text = str(exc)
    lowered = text.lower()
    if "rate_limit" in lowered or "429" in text or "tokens per minute" in lowered:
        return "I hit my hourly usage limit with the language model. Worth trying again in a minute."
    if "tool_use_failed" in lowered or "tool choice is none" in lowered:
        return "My tools didn't fire correctly on that one — worth trying again."
    if "timeout" in lowered or "timed out" in lowered:
        return "It took too long and timed out."
    return text[:200]


def _dispatch_tool(specialists: dict) -> StructuredTool:
    """The "hand it off and stay free" tool.

    Unlike the ask_<agent> delegate tools above — which redirect the graph to
    a specialist node and make the whole turn wait for the answer — this one
    starts the specialist as a detached asyncio task and returns *instantly*.
    The router therefore finishes its turn normally, in seconds, and the user
    can immediately ask for something else while the work continues.

    Implemented as a tool rather than a graph node on purpose: a node would
    have to either block the turn (defeating the point) or fork execution out
    of the graph, which LangGraph's checkpointer has no way to represent. A
    tool that returns a receipt keeps the graph's own control flow completely
    ordinary — the "background" part lives entirely on maks/runtime.py's
    persistent event loop, outside the graph, where it can safely outlive the
    turn that started it.
    """

    async def _run(agent: str, task: str) -> str:
        if agent not in _BACKGROUNDABLE:
            return (
                f"'{agent}' can't be run in the background. Only "
                f"{' and '.join(_BACKGROUNDABLE)} support it; everything else "
                "is fast enough to just do now with its normal ask_ tool."
            )
        if agent not in specialists:
            return f"Unknown agent '{agent}'."

        job = jobs.create(agent, task)

        async def _heartbeat() -> None:
            """Speaks a short progress line while the job runs, but only into
            silence — if the user has said anything recently they're mid
            conversation, and interrupting that to report that a search is
            still running is worse than saying nothing. Cancelled as soon as
            the job finishes, so it can never talk over the completion
            announcement.
            """
            while True:
                await asyncio.sleep(settings.job_progress_interval_seconds)
                if jobs.seconds_since_user_activity() < settings.job_progress_quiet_seconds:
                    continue
                current = jobs.get(job.id)
                if current is None or current.status != jobs.RUNNING:
                    return
                announce_unprompted(agent, current.progress_line())

        async def _work() -> None:
            async with _job_slot:  # one background job at a time — see _job_slot
                beat = asyncio.create_task(_heartbeat())
                try:
                    answer, _needs_handoff = await invoke_specialist(
                        specialists[agent],
                        task,
                        thread_id=SPECIALIST_THREAD_IDS.get(agent),
                        job_id=job.id,
                    )
                except Exception as exc:  # noqa: BLE001 — a crashed background job must report, never vanish
                    # Log the raw exception before reducing it to something
                    # speakable: nobody is watching this task, so if the
                    # friendly version is all that survives, a real bug has
                    # nowhere to show up.
                    print(f"[jobs] background job {job.id} ({agent}) failed: {exc!r}")
                    jobs.fail(job.id, _friendly_error(exc))
                    announce_unprompted(
                        agent, f"Sorry sir, job {job.id} didn't make it. {_friendly_error(exc)}"
                    )
                    return
                finally:
                    # Always, on both paths: a surviving heartbeat would keep
                    # narrating a job that has already finished, and could
                    # talk over the completion announcement.
                    beat.cancel()
            jobs.complete(job.id, answer)
            # Deliberately does NOT read the result out. A research sweep's
            # answer runs to paragraphs, and having minutes of prose suddenly
            # narrated at you — while you're doing something else — is worse
            # than useless. Announce that it landed, then let the user pull
            # the detail on their own terms via check_background_jobs.
            announce_unprompted(agent, f"Job {job.id} is done, sir — {task}. Say the word and I'll go through it.")

        task_handle = asyncio.create_task(_work())
        _dispatched_tasks.add(task_handle)
        task_handle.add_done_callback(_dispatched_tasks.discard)

        return (
            f"Dispatched to {agent} as background job {job.id}. It is running now. "
            "Tell the user you've kicked it off, mention the job number, and ask what "
            "else you can help with — do NOT wait for or invent a result; they'll be "
            "told automatically when it finishes."
        )

    return StructuredTool.from_function(
        coroutine=_run,
        name="dispatch_background_task",
        description=(
            "Start a long-running task in the background and return immediately, so you "
            "stay free for the user's next request. Use this for work that would take a "
            "while and that the user doesn't need to sit and wait for.\n"
            "`agent` must be exactly one of:\n"
            "  - 'deep_research_agent' for a genuine deep/thorough research sweep across "
            "multiple sources — the user asking you to 'research', 'dig into', 'do a deep "
            "dive on', or 'find everything about' a topic. It plans, reads many sources, "
            "and synthesizes. Slow but thorough.\n"
            "  - 'office_agent' for ordinary look-it-up web searches, email, calendar, or "
            "Notion work that just happens to be slow.\n"
            "  - 'coder_agent' ONLY for writing/editing/debugging actual code in a project.\n"
            "A research or 'find out about X' request is deep_research_agent or "
            "office_agent, never coder_agent.\n"
            "`task` must be a clear, standalone instruction. Do NOT use this tool for quick "
            "questions — use the normal ask_<agent> tools for those, since the user would "
            "rather just have the answer now."
        ),
    )


def _job_status_tool() -> StructuredTool:
    async def _run() -> str:
        return jobs.summary()

    return StructuredTool.from_function(
        coroutine=_run,
        name="check_background_jobs",
        description=(
            "List background jobs dispatched this session — what's still running, what "
            "finished, and the results of the finished ones. Use this whenever the user "
            "asks how something is going, what you're working on, whether a job is done, "
            "or asks to hear the result of earlier background work."
        ),
    )


def _trim_history(state: dict) -> dict:
    """pre_model_hook: caps how much history is fed to the router's own
    model call without touching the actual checkpointed state — a long
    session shouldn't make every turn slower or blow the token budget.
    Token-aware (not a flat message-count slice) so a handful of long
    messages gets trimmed the same as many short ones.
    """
    trimmed = trim_messages(
        # sanitize_history drops past turns where the model described a tool
        # call in prose instead of making one — see its docstring. The router
        # is the worst place to let that compound, since it's the node that
        # has to pick a tool on every single turn.
        sanitize_history(state["messages"]),
        max_tokens=settings.router_max_context_tokens,
        strategy="last",
        token_counter="approximate",
        start_on="human",
        include_system=True,
    )
    return {"llm_input_messages": trimmed}


def _specialist_node(agent, agent_key: str, announce_lead_in: str | None = None):
    """`announce_lead_in`, if given, names the specialist in a short spoken
    lead-in (e.g. "Handing this to the office agent") — the actual task text
    (already a clear, standalone instruction, per SUPERVISOR_ROLE) gets
    appended so the announcement says what's actually being done, not just
    that *something* is happening.

    Passes SPECIALIST_THREAD_IDS.get(agent_key) to invoke_specialist — for
    the three specialists that have one (everything but mac_control_agent),
    this is what makes delegation "sticky": each router-mediated delegation
    lands on that same persisted thread, so a specialist that's been asked
    for something before actually remembers it, via its own checkpointer
    (threaded in from _assemble_graph) rather than anything tracked here.
    """
    thread_id = SPECIALIST_THREAD_IDS.get(agent_key)

    async def _run(state: SupervisorState) -> dict:
        task = state["task"]
        if announce_lead_in is not None:
            announce_delegation(agent_key, f"{announce_lead_in} — {task}")
        answer, needs_handoff = await invoke_specialist(agent, task, thread_id=thread_id)
        return {
            "result": answer,
            "needs_handoff": needs_handoff,
            "agent_key": agent_key,
            "delegation_count": state.get("delegation_count", 0) + 1,
        }

    return _run


def _reset_turn(state: SupervisorState) -> dict:
    """Runs once, between START and the router, on every fresh top-level
    invocation — the loop-back edge (after_specialist -> router) bypasses
    this node entirely, so delegation_count keeps counting *within* one
    turn's specialist/router back-and-forth but always starts clean on the
    next one. Needed because SupervisorState is checkpointed per thread (see
    build_supervisor_graph), so without this the counter would otherwise
    carry over from the previous turn and could trip MAX_DELEGATIONS_PER_TURN
    on a turn's very first delegation.
    """
    return {"delegation_count": 0}


async def _after_specialist(state: SupervisorState) -> dict:
    """Hands the specialist's finding back to the router for review.

    Deliberately a HumanMessage, not an AIMessage, even though nothing about
    it came from the user. This is load-bearing: an AIMessage lands in the
    "things the assistant said" slot of the checkpointed history, and the
    model then imitates it on later turns — replying with the raw internal
    label verbatim instead of a real answer. Worse, it compounds: each
    polluted turn is another example teaching the next turn to do the same,
    and the observed end state was a router that had stopped emitting real
    tool calls entirely and just narrated their names as text.

    Framing it as an inbound instruction the model must *act on* (rather than
    an example of its own past output) keeps the review step working without
    teaching the model to talk like scaffolding. The explicit "not from the
    user" tag stops the other failure mode — the model thanking the user for
    information they never gave.
    """
    note = (
        f"[INTERNAL SYSTEM NOTE — not from the user, never repeat it verbatim]\n"
        f"The {state['agent_key']} specialist reported back:\n"
        f"{state['result']}"
    )
    if state.get("needs_handoff"):
        note += "\n\n(It flagged that it could not fully complete the request on its own.)"
    note += (
        "\n\nNow write your own reply to the user, in your own words, based on this. "
        "Summarize rather than reading it all back if it's long or technical."
    )
    return {"messages": [HumanMessage(content=note)]}


def _route_after_specialist(state: SupervisorState) -> str:
    if state.get("delegation_count", 0) >= MAX_DELEGATIONS_PER_TURN:
        return "finalize"
    return "router"


async def _finalize(state: SupervisorState) -> dict:
    """Hard stop for MAX_DELEGATIONS_PER_TURN: skip the router's review LLM
    call entirely and relay the last specialist's own answer verbatim as the
    turn's final reply. No extra Groq call — deliberately, since this only
    triggers once a turn has already spent its delegation budget and may
    already be under rate-limit pressure (see MAX_DELEGATIONS_PER_TURN).
    """
    return {"messages": [AIMessage(content=state["result"])]}


def _assemble_graph(tools_by_agent: dict[str, list[BaseTool]], checkpointer: AsyncSqliteSaver | None = None):
    """Shared by build_supervisor_graph() and make_graph() below — building
    the specialists/router/graph is identical either way; the only
    difference between the two callers is *how* tools_by_agent got
    populated (a persistent MCP session vs. a one-shot connection — see
    mcp_client.py's init() vs. load_tools_stateless() for why that split
    exists) and whether a real, persistent `checkpointer` is given.

    `checkpointer` is None from make_graph() (Studio testing keeps its own
    fresh MemorySaver, see that function's docstring) and the shared
    AsyncSqliteSaver from build_supervisor_graph() (the real app). It's
    threaded into every specialist except mac_control_agent's DynamicWorker
    (see maks/graph/state.py's SPECIALIST_THREAD_IDS docstring for why that
    one stays excluded) as well as into the top-level graph itself, so a
    specialist addressed via its stable thread id (SPECIALIST_THREAD_IDS)
    actually persists across calls instead of starting fresh every time.

    Returns (graph, specialists): `specialists` is a name -> agent map of the
    exact same built agent instances the specialist nodes use, reused by
    maks/graph/router.py's embedding fast-path (via maks/pipeline.py) so a
    confident single-intent request can be sent straight to a specialist
    without building a second copy of it or paying for the router's own LLM
    call.
    """
    chit_chat_agent = build_chit_chat_agent(tools_by_agent.get("chit_chat_agent", []), checkpointer=checkpointer)
    office_agent = build_office_agent(tools_by_agent.get("office_agent", []), checkpointer=checkpointer)
    coder_agent = build_coder_agent(tools_by_agent.get("coder_agent", []), checkpointer=checkpointer)
    mac_dynamic_worker = build_mac_dynamic_worker_graph(tools_by_agent.get("mac_control_agent", []))

    specialists = {
        "chit_chat_agent": chit_chat_agent,
        "office_agent": office_agent,
        "coder_agent": coder_agent,
        "mac_control_agent": mac_dynamic_worker,
    }

    # Not a graph node and not in _SPECIALIST_NAMES: the deep researcher is
    # only ever reachable via dispatch_background_task, so it needs to exist
    # in this dict (that's where dispatch looks agents up) but has no inline
    # delegate tool or node. None when `deepagents` isn't installed, in which
    # case it simply isn't offered.
    office_tools = tools_by_agent.get("office_agent", [])
    deep_research_agent = build_deep_research_agent(
        [t for t in office_tools if t.name in _RESEARCH_TOOL_NAMES]
    )
    if deep_research_agent is not None:
        specialists["deep_research_agent"] = deep_research_agent

    router_tools: list[BaseTool] = list(tools_by_agent.get("supervisor", []))  # weather_lookup
    router_tools += [
        _delegate_tool(
            "ask_chit_chat_agent",
            "Delegate plain conversation (greetings, small talk, opinions, general "
            "knowledge) to the chit-chat specialist. `task` should be a clear, "
            "standalone instruction — it does not see the rest of this conversation.",
            "chit_chat_agent",
        ),
        _delegate_tool(
            "ask_office_agent",
            "Delegate a web-research, Gmail/Calendar/messaging, or Notion task to the "
            "office specialist. `task` should be a clear, standalone instruction — it "
            "does not see the rest of this conversation.",
            "office_agent",
        ),
        _delegate_tool(
            "ask_mac_control_agent",
            "Delegate a Mac remote-control task (YouTube, Spotify, file search, system "
            "health) to the Mac specialist. `task` should be a clear, standalone "
            "instruction — it does not see the rest of this conversation.",
            "mac_control_agent",
        ),
        _delegate_tool(
            "ask_coder_agent",
            "Delegate a coding task to Claude via the coder specialist. `task` should be "
            "a clear, standalone instruction — it does not see the rest of this conversation.",
            "coder_agent",
        ),
        _dispatch_tool(specialists),
        _job_status_tool(),
    ]

    router_agent = create_react_agent(
        model=get_llm(temperature=0.2),
        tools=router_tools,
        prompt=dynamic_prompt(SUPERVISOR_ROLE),
        pre_model_hook=_trim_history,
        name="router",
    )

    builder = StateGraph(SupervisorState)
    # destinations= doesn't affect execution (the router's delegate tools
    # already redirect via Command(graph=PARENT, goto=...)) — it's purely so
    # Studio can draw the dashed router -> specialist edges; without it,
    # Studio has no static way to know a tool buried inside the router's own
    # ToolNode can jump to these nodes.
    builder.add_node("router", router_agent, destinations=_SPECIALIST_NAMES)
    builder.add_node("chit_chat_agent", _specialist_node(chit_chat_agent, "chit_chat_agent"))
    builder.add_node(
        "office_agent",
        _specialist_node(office_agent, "office_agent", announce_lead_in="Handing this to the office agent"),
    )
    builder.add_node(
        "mac_control_agent",
        _specialist_node(
            mac_dynamic_worker, "mac_control_agent", announce_lead_in="Handing this to the Mac control agent"
        ),
    )
    builder.add_node(
        "coder_agent",
        # No announcement here: coder_agent's own post_model_hook
        # (maks/agents/coder_agent.py) announces the instant it actually
        # decides to call run_claude_code — more precise than announcing
        # the moment the router delegates, before it's confirmed that's
        # what the specialist will do.
        _specialist_node(coder_agent, "coder_agent"),
    )
    builder.add_node("after_specialist", _after_specialist)
    builder.add_node("start_turn", _reset_turn)
    builder.add_node("finalize", _finalize)

    builder.add_edge(START, "start_turn")
    builder.add_edge("start_turn", "router")
    builder.add_edge("router", END)
    for name in _SPECIALIST_NAMES:
        builder.add_edge(name, "after_specialist")
    # Back to the router so it can review every specialist result before
    # answering (see the module docstring for why this trades one extra LLM
    # call per delegation for the router usually getting a chance to catch a
    # bad answer) — but only up to MAX_DELEGATIONS_PER_TURN; past that,
    # _finalize short-circuits straight to a reply so a turn can never
    # bounce back and forth unboundedly (see that constant's docstring).
    builder.add_conditional_edges("after_specialist", _route_after_specialist, ["router", "finalize"])
    builder.add_edge("finalize", END)

    graph = builder.compile(checkpointer=checkpointer or MemorySaver())
    return graph, specialists


async def build_supervisor_graph():
    """Connects to every MCP server (once — the connections stay open for
    the life of the process, via maks/mcp_client.py's init()) and compiles
    the runnable supervisor, backed by a single persistent AsyncSqliteSaver
    (see _get_checkpointer()) so both the router's own conversation and
    every sticky specialist's own thread survive a process restart. Used by
    the real app (maks/pipeline.py).
    """
    await mcp_client.init()
    tools_by_agent = {
        name: mcp_client.get_tools(name)
        for name in ("supervisor", "chit_chat_agent", "office_agent", "coder_agent", "mac_control_agent")
    }
    checkpointer = await _get_checkpointer()
    return _assemble_graph(tools_by_agent, checkpointer=checkpointer)


async def make_graph(config=None):
    """Entry point for `langgraph dev` (see langgraph.json at the project
    root) — the CLI expects a factory returning just the compiled graph, not
    the (graph, specialists) tuple build_supervisor_graph() returns for
    maks/pipeline.py's own use (specialists are only needed there, to back
    the embedding fast-path router). `config` is accepted but unused — it's
    part of the factory signature the CLI supports, for cases that vary the
    graph per-config; this one doesn't.

    Deliberately does NOT call build_supervisor_graph()/mcp_client.init():
    `langgraph dev`'s in-memory server calls this factory from a new
    short-lived task on every API request (confirmed via its own "Slow graph
    load" warnings firing on every schema/graph/history call, not just
    startup), so a persistent AsyncExitStack opened here gets torn down when
    that particular request's task ends — the next real tool call then hits
    a closed session (anyio.ClosedResourceError). mcp_client.load_tools_stateless()
    opens a fresh connection per tool call instead, which is slower per call
    but immune to that failure mode — see its docstring for the full story.

    Also note: this bypasses maks/pipeline.py's embedding fast-path router
    entirely — `langgraph dev` talks to this graph directly, so every
    request goes through the full router's own LLM-based routing. That's
    the point for testing the graph in isolation from the voice/dashboard
    app, not a bug.
    """
    tools_by_agent = await mcp_client.load_tools_stateless()
    graph, _specialists = _assemble_graph(tools_by_agent)
    return graph
