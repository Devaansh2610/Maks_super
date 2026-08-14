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

Classic supervisor control flow, by deliberate choice: every specialist's
result always routes back through the router (`after_specialist -> router`,
unconditional) so the router reviews it before answering — one extra LLM
call per delegation, in exchange for the router always getting a chance to
catch a wrong/incomplete answer before it reaches the user, rather than
specialists silently ending the turn themselves. A specialist's result comes
back to the router labeled as a finding to review (see `_after_specialist`),
not injected as if the router already said it — otherwise the model reading
its own history back would get confused about why it's being asked to speak
again immediately after an apparently-already-complete answer.

Each specialist still only ever sees its own isolated task string via
`invoke_specialist` (maks/agents/_common.py) — nothing about routing through
real graph nodes instead of tool coroutines changes that; `invoke_specialist`
is called exactly the same way either way, just from a plain node function
instead of from inside a `StructuredTool`'s coroutine.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
from langchain_core.tools import BaseTool, InjectedToolCallId, StructuredTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command

from maks import mcp_client
from maks.agents._common import announce_delegation, dynamic_prompt, invoke_specialist
from maks.agents.chit_chat_agent import build_chit_chat_agent
from maks.agents.coder_agent import build_coder_agent
from maks.agents.office_agent import build_office_agent
from maks.graph.dynamic_worker import build_mac_dynamic_worker_graph
from maks.llm import get_llm

MAX_HISTORY_MESSAGES = 12
_SPECIALIST_NAMES = ("chit_chat_agent", "office_agent", "coder_agent", "mac_control_agent")

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

When you delegate, write `task` as a clear, standalone instruction — the
specialist does not see the rest of this conversation, only what you put in
`task`. After a specialist runs, its finding comes back to you labeled
"[<agent> specialist reported back — review before replying]" — that is NOT
your final reply, it's a draft for you to check. Read it, then either:
- produce your own final reply to the user (relay/confirm the finding in
  your own words — never show the user the raw labeled note itself), or
- delegate again — to the same specialist with a clearer task, or to a
  different one — if it's wrong, incomplete, or the specialist flagged it
  couldn't fully complete the request on its own."""


class SupervisorState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    task: str
    result: str
    needs_handoff: bool
    agent_key: str


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


def _trim_history(state: dict) -> dict:
    """pre_model_hook: caps how much history is fed to the router's own
    model call without touching the actual checkpointed state — a long
    session shouldn't make every turn slower.
    """
    messages = state["messages"]
    if len(messages) <= MAX_HISTORY_MESSAGES:
        return {}
    return {"llm_input_messages": messages[-MAX_HISTORY_MESSAGES:]}


def _specialist_node(agent, agent_key: str, announce_lead_in: str | None = None):
    """`announce_lead_in`, if given, names the specialist in a short spoken
    lead-in (e.g. "Handing this to the office agent") — the actual task text
    (already a clear, standalone instruction, per SUPERVISOR_ROLE) gets
    appended so the announcement says what's actually being done, not just
    that *something* is happening.
    """

    async def _run(state: SupervisorState) -> dict:
        task = state["task"]
        if announce_lead_in is not None:
            announce_delegation(agent_key, f"{announce_lead_in} — {task}")
        answer, needs_handoff = await invoke_specialist(agent, task)
        return {"result": answer, "needs_handoff": needs_handoff, "agent_key": agent_key}

    return _run


async def _after_specialist(state: SupervisorState) -> dict:
    # Labeled as a finding to review, not injected as a bare AIMessage that
    # would look like the router already said it — a model reading its own
    # history back would otherwise get confused about why it's being asked
    # to speak again right after an apparently-already-complete answer. Only
    # this clean text ever enters the shared conversation — the specialist's
    # internal tool calls/back-and-forth stay isolated inside
    # invoke_specialist, same as before.
    note = f"[{state['agent_key']} specialist reported back — review before replying]: {state['result']}"
    if state.get("needs_handoff"):
        note += "\n(This specialist flagged that it could not fully complete the request on its own.)"
    return {"messages": [AIMessage(content=note)]}


def _assemble_graph(tools_by_agent: dict[str, list[BaseTool]]):
    """Shared by build_supervisor_graph() and make_graph() below — building
    the specialists/router/graph is identical either way; the only
    difference between the two callers is *how* tools_by_agent got
    populated (a persistent MCP session vs. a one-shot connection — see
    mcp_client.py's init() vs. load_tools_stateless() for why that split
    exists).

    Returns (graph, specialists): `specialists` is a name -> agent map of the
    exact same built agent instances the specialist nodes use, reused by
    maks/graph/router.py's embedding fast-path (via maks/pipeline.py) so a
    confident single-intent request can be sent straight to a specialist
    without building a second copy of it or paying for the router's own LLM
    call.
    """
    chit_chat_agent = build_chit_chat_agent(tools_by_agent.get("chit_chat_agent", []))
    office_agent = build_office_agent(tools_by_agent.get("office_agent", []))
    coder_agent = build_coder_agent(tools_by_agent.get("coder_agent", []))
    mac_dynamic_worker = build_mac_dynamic_worker_graph(tools_by_agent.get("mac_control_agent", []))

    specialists = {
        "chit_chat_agent": chit_chat_agent,
        "office_agent": office_agent,
        "coder_agent": coder_agent,
        "mac_control_agent": mac_dynamic_worker,
    }

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

    builder.add_edge(START, "router")
    builder.add_edge("router", END)
    for name in _SPECIALIST_NAMES:
        builder.add_edge(name, "after_specialist")
    # Always back to the router — it reviews every specialist result before
    # answering (see the module docstring for why this trades one extra LLM
    # call per delegation for the router always getting a chance to catch a
    # bad answer).
    builder.add_edge("after_specialist", "router")

    graph = builder.compile(checkpointer=MemorySaver())
    return graph, specialists


async def build_supervisor_graph():
    """Connects to every MCP server (once — the connections stay open for
    the life of the process, via maks/mcp_client.py's init()) and compiles
    the runnable supervisor (with an in-memory checkpointer so a session
    remembers context turn to turn). Used by the real app (maks/pipeline.py).
    """
    await mcp_client.init()
    tools_by_agent = {
        name: mcp_client.get_tools(name)
        for name in ("supervisor", "chit_chat_agent", "office_agent", "coder_agent", "mac_control_agent")
    }
    return _assemble_graph(tools_by_agent)


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
