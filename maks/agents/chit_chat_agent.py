"""Plain conversation — greetings, small talk, opinions, jokes, general
knowledge the model already has. No tools at all: "just the LLM".
"""

from __future__ import annotations

from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent

from maks.agents._common import HANDOFF_INSTRUCTION, dynamic_prompt
from maks.llm import get_llm

ROLE = (
    "You handle plain conversation: greetings, small talk, opinions, jokes, "
    "and general knowledge you already know. Answer directly and briefly in "
    "your own voice — don't mention tools, agents, or delegation to the user."
    + HANDOFF_INSTRUCTION.format(other_agents="office_agent, coder_agent, or mac_control_agent")
)


def build_chit_chat_agent(tools: list[BaseTool]):
    return create_react_agent(
        model=get_llm(),
        tools=tools,
        prompt=dynamic_prompt(ROLE),
        name="chit_chat_agent",
    )
