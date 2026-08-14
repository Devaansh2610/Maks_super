"""The Mac-control DynamicWorker: a small hand-built StateGraph (LangGraph's
`Send` API — the standard orchestrator/worker map-reduce primitive) that
breaks a Mac-control request into 1+ independent subtasks at runtime and
runs them concurrently, instead of one sequential ReAct loop over every Mac
tool. Replaces the old, simpler mac_control_agent.

Not langgraph-swarm, not a fixed pipeline: the number of workers is decided
per-request by `plan_subtasks` and fanned out via `Send`, which is exactly
what "dynamic" means here — most requests are a single subtask (one worker),
but a compound one ("open the browser and play a song") splits into two
workers that run at the same time instead of forcing a round trip back up to
the top-level supervisor between them.

The compiled graph exposes the same `.ainvoke({"messages": [...]}) ->
{"messages": [...]}` shape every specialist agent does, so
maks/graph/supervisor.py's _delegate_tool can call it exactly like any other
specialist — see build_mac_dynamic_worker_graph() below, which is what gets
passed there instead of the old build_mac_control_agent(...).
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.types import Send
from pydantic import BaseModel, Field

from maks.agents._common import dynamic_prompt
from maks.llm import get_llm

# Static, never interpolated with per-request text — every worker call's
# system prompt is byte-identical across requests, so consecutive Groq calls
# share an identical prefix and benefit from Groq's automatic prompt caching
# (confirmed: automatic, no code changes required, cache hit needs an exact
# prefix match). The actual subtask only ever goes in as the trailing
# HumanMessage, never baked into this string.
WORKER_ROLE = (
    "You control the user's Mac remotely (it's on the same network, "
    "reachable via a companion service). Use play_youtube / play_spotify to "
    "start media, spotify_control for play/pause/skip, search_files to find "
    "files on the Mac, and system_info to check/analyze its health (CPU, "
    "RAM, disk, battery, running apps). If you have no tools available, the "
    "companion service isn't reachable — say so plainly rather than "
    "pretending it worked. You'll be given exactly one atomic subtask — "
    "handle only that, report the result concisely."
)

PLANNER_ROLE = (
    "You break a Mac-control request into a short list of independent, "
    "atomic subtasks — most requests are already atomic and should stay as "
    "a single subtask. Only split when the request clearly names more than "
    "one independent action (e.g. 'open the browser and also play a song' "
    "is two; 'play a song by X' is one). Never invent extra subtasks."
)


class _SubtaskList(BaseModel):
    subtasks: list[str] = Field(
        description="1 or more atomic, independent Mac-control instructions, each "
        "phrased as a standalone task."
    )


class MacGraphState(TypedDict):
    messages: list[BaseMessage]
    results: Annotated[list[str], operator.add]
    _subtasks: list[str]


class MacWorkerState(TypedDict):
    subtask: str


def build_mac_dynamic_worker_graph(tools: list[BaseTool]):
    """Compiles the DynamicWorker graph. `tools` is the Mac companion's tool
    list (mcp_client.get_tools("mac_control_agent")) — same source the old
    mac_control_agent used, just consumed differently here.
    """

    async def plan_subtasks(state: MacGraphState) -> dict:
        task_text = state["messages"][-1].content
        if not tools:
            # No point asking the model to plan work it has nothing to
            # execute with — treat it as a single subtask so the worker node
            # below produces the "companion isn't reachable" message.
            return {"_subtasks": [task_text]}

        planner = get_llm(temperature=0.0).with_structured_output(_SubtaskList)
        try:
            plan = await planner.ainvoke(
                [
                    {"role": "system", "content": PLANNER_ROLE},
                    {"role": "user", "content": task_text},
                ]
            )
            subtasks = [s for s in plan.subtasks if s.strip()] or [task_text]
        except Exception:  # noqa: BLE001 — planning is an optimization, never a hard requirement
            subtasks = [task_text]

        return {"_subtasks": subtasks}

    def fan_out(state: dict) -> list[Send]:
        return [Send("mac_worker", {"subtask": s}) for s in state["_subtasks"]]

    async def mac_worker(state: MacWorkerState) -> dict:
        worker_agent = create_react_agent(
            model=get_llm(),
            tools=tools,
            prompt=dynamic_prompt(WORKER_ROLE),
            name="mac_worker",
        )
        result = await worker_agent.ainvoke({"messages": [HumanMessage(content=state["subtask"])]})
        return {"results": [result["messages"][-1].content]}

    async def aggregate(state: MacGraphState) -> dict:
        results = state["results"]
        if len(results) == 1:
            final_text = results[0]
        else:
            synthesis_prompt = (
                "Combine these Mac-control task results into one short, "
                "coherent spoken reply:\n\n" + "\n\n".join(f"- {r}" for r in results)
            )
            response = await get_llm().ainvoke([{"role": "user", "content": synthesis_prompt}])
            final_text = response.content
        return {"messages": [AIMessage(content=final_text)]}

    graph = StateGraph(MacGraphState)
    graph.add_node("plan_subtasks", plan_subtasks)
    graph.add_node("mac_worker", mac_worker)
    graph.add_node("aggregate", aggregate)

    graph.add_edge(START, "plan_subtasks")
    graph.add_conditional_edges("plan_subtasks", fan_out, ["mac_worker"])
    graph.add_edge("mac_worker", "aggregate")
    graph.add_edge("aggregate", END)

    return graph.compile()
