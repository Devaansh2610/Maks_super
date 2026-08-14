"""Coding tasks: always hand off to the Claude Code CLI, never write code itself.

run_claude_code lives in a separate MCP server process (maks/mcp_servers/
api_connectors.py), so it can't reach into this process's event bus/speaker
to announce the handoff before the (possibly slow, up to 15 minutes) subprocess
finishes. Instead, a post_model_hook here fires the moment the model decides
to call that tool — right after the model responds, before the ToolNode
actually executes it — so "handing this to Claude" is spoken and shown on the
dashboard instantly, rather than leaving the user in silence until the whole
handoff completes.
"""

from __future__ import annotations

from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent

from maks.agents._common import HANDOFF_INSTRUCTION, announce_delegation, dynamic_prompt
from maks.llm import get_llm

ROLE = (
    "The user asked for something code-related: writing, editing, debugging, "
    "reviewing, or explaining code in a real project. You never write code "
    "yourself — you always call run_claude_code with a clear task description "
    "built from the user's request. Your final reply must clearly state you "
    "handed this to Claude and briefly relay what it reported back."
    + HANDOFF_INSTRUCTION.format(other_agents="chit_chat_agent, office_agent, or mac_control_agent")
)


async def _announce_claude_handoff(state: dict) -> dict:
    # Deliberately async, not sync: LangGraph runs a sync post_model_hook in
    # a thread-pool executor when the graph is invoked via .ainvoke() (no
    # running event loop there), but announce_delegation needs
    # asyncio.create_task(), which requires being on the loop's own thread.
    # Making this hook itself a coroutine function is what gets LangGraph to
    # await it directly on the loop instead of offloading it.
    messages = state.get("messages", [])
    if not messages:
        return {}

    last = messages[-1]
    for call in getattr(last, "tool_calls", None) or []:
        if call.get("name") == "run_claude_code":
            task = (call.get("args") or {}).get("task", "")
            announce_delegation("claude", f"Handing this off to Claude Code — {task}")
    return {}


def build_coder_agent(tools: list[BaseTool]):
    return create_react_agent(
        model=get_llm(),
        tools=tools,
        prompt=dynamic_prompt(ROLE),
        post_model_hook=_announce_claude_handoff,
        name="coder_agent",
    )
