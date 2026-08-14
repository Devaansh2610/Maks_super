"""Plain conversation — greetings, small talk, opinions, jokes, general
knowledge the model already has. No tools at all: "just the LLM".
"""

from __future__ import annotations

from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent

from maks.agents._common import HANDOFF_INSTRUCTION, dynamic_prompt, make_trim_hook
from maks.llm import get_llm
from maks.settings import settings

ROLE = (
    "You handle plain conversation: greetings, small talk, opinions, jokes, "
    "and general knowledge you already know. Your personality is witty and "
    "a little dry — think JARVIS from Iron Man: clever, warm, quick with a "
    "sharp remark or a light joke, unfailingly helpful underneath the "
    "banter. Never stiff or robotic, but never rambling either — this is "
    "spoken aloud, so land the wit in a line or two, not a monologue. "
    "Answer directly in that voice — don't mention tools, agents, or "
    "delegation to the user."
    + HANDOFF_INSTRUCTION.format(other_agents="office_agent, coder_agent, or mac_control_agent")
)


def build_chit_chat_agent(tools: list[BaseTool], checkpointer=None):
    # checkpointer is only non-None from the real app (build_supervisor_graph)
    # — see maks/graph/state.py's SPECIALIST_THREAD_IDS for why this agent is
    # eligible for sticky, cross-turn memory but mac_control_agent isn't.
    return create_react_agent(
        model=get_llm(),
        tools=tools,
        prompt=dynamic_prompt(ROLE),
        pre_model_hook=make_trim_hook(settings.chit_chat_max_context_tokens),
        checkpointer=checkpointer,
        name="chit_chat_agent",
    )
