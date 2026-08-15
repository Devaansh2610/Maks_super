"""Long-horizon research, via LangChain's `deepagents` harness.

Different in kind from every other specialist here. The others are a single
ReAct loop: think, call a tool, answer. This one is an *agent harness* — it
plans the work into a todo list first, spawns its own ephemeral sub-agents
for independent sub-questions (each with a fresh context window, so one
sub-question's noise doesn't crowd out another's), offloads bulk findings to
a virtual filesystem instead of carrying them in context, and only then
synthesizes. That's what makes it able to sustain a multi-source sweep
rather than a single search-and-summarize.

Reachable ONLY through dispatch_background_task (see
maks/graph/supervisor.py) — deliberately. A real research sweep is many LLM
calls deep and takes minutes; running it in the foreground would block the
whole conversation for the entire time, which is precisely the problem
background dispatch exists to solve. There is no ask_deep_research_agent
delegate tool, so it cannot accidentally be run inline.

Note it is NOT given a checkpointer/sticky thread: each sweep is a
self-contained piece of research, and carrying one topic's accumulated
findings into the next unrelated topic would poison it — the same reasoning
that keeps mac_control_agent isolated (see maks/graph/state.py).
"""

from __future__ import annotations

from langchain_core.tools import BaseTool

from maks.llm import get_llm
from maks.settings import settings

SUB_RESEARCHER_PROMPT = """You research one specific sub-question, on your
own, and report back concisely.

Use search_the_web to find sources and read_web_page to actually read the promising
ones — a search snippet is rarely enough to say anything specific. Ground
every claim in something you read; never invent a source, date, or number.
Report the findings for your sub-question only. Do not try to answer the
wider research question; someone else is assembling that."""

RESEARCH_PROMPT = """You are a research specialist doing a thorough,
multi-source sweep for the user.

Method:
1. Plan first. Break the question into the specific sub-questions that
   actually need answering, and write them to your todo list.
2. Search broadly, then read deeply. Use search_the_web to find sources, then
   read_web_page on the ones that look substantive — a search snippet alone is
   rarely enough to say anything specific.
3. Offload as you go. Write findings to files as you gather them rather than
   trying to hold everything in your head; that's what the filesystem is for.
4. Delegate independent sub-questions to sub-agents so each gets a clean
   context to work in.
5. Synthesize at the end into a single coherent answer.

Ground every factual claim in something you actually read. Never invent a
source, a date, a number, or a quote. If the evidence is thin or sources
disagree, say so plainly — that is a finding too.

Work economically. You have a limited context budget, so keep the number of
sub-questions small (3-5), write findings to files as soon as you have them
instead of carrying them forward in the conversation, and stop searching
once you can actually answer the question rather than gathering more.

Your final message is what gets reported back to the user, and it may be
read aloud. Lead with the headline answer in a sentence or two, then the
supporting detail. Keep it tight: a handful of substantive points beats an
exhaustive dump."""


# gpt-oss ships with its own built-in browser/web_search tool baked into
# training. Handing it a *different* tool with the same name invites it to
# call the remembered one instead — with the remembered arguments (top_n,
# recency_days, source) rather than ours — which the provider then rejects
# outright ("attempted to call tool 'web_search' which was not in
# request.tools"). Renaming ours sidesteps the association entirely. Only the
# name the model sees changes; the underlying MCP tool is untouched.
_RENAMES = {"web_search": "search_the_web", "fetch_page": "read_web_page"}

# Hard ceiling on how much text one tool call can put into the model's
# context. Research is the one job that reads whole web pages, and on Groq's
# free tier the entire *request* must fit in 8000 tokens — so a couple of
# unabridged pages is enough to fail the next call outright with a 413
# (measured: a request reached 9956 tokens and was rejected). Truncating at
# the tool boundary is the reliable place to enforce this: it can't be
# talked out of it, unlike a prompt asking the model to request less.
# ~1200 chars is roughly 300 tokens — enough to judge and quote a source,
# small enough that several fit alongside the conversation.
_MAX_TOOL_OUTPUT_CHARS = 1200


def _prepare_research_tool(tool: BaseTool) -> BaseTool:
    """Rename (see _RENAMES) and cap the output of one research tool."""
    from langchain_core.tools import StructuredTool

    name = _RENAMES.get(tool.name, tool.name)

    async def _run(**kwargs: object) -> str:
        result = await tool.ainvoke(kwargs)
        text = result if isinstance(result, str) else str(result)
        if len(text) > _MAX_TOOL_OUTPUT_CHARS:
            text = (
                text[:_MAX_TOOL_OUTPUT_CHARS]
                + "\n\n[...truncated. Search for a more specific page if you need more detail.]"
            )
        return text

    return StructuredTool.from_function(
        coroutine=_run,
        name=name,
        description=tool.description,
        args_schema=tool.args_schema,
    )


def build_deep_research_agent(tools: list[BaseTool]):
    """`tools` should be the research-capable subset (web_search,
    fetch_page). deepagents adds its own planning/filesystem/sub-agent tools
    on top of whatever is passed here.

    Imported lazily inside the function so the rest of the app still starts
    if `deepagents` isn't installed — it pulls a fairly heavy dependency tree
    (anthropic, google-genai, langchain) that a user who never touches deep
    research shouldn't be forced to have working. Callers handle None by
    simply not offering the agent.
    """
    try:
        from deepagents import create_deep_agent
    except ImportError:
        return None

    model = get_llm(model=settings.deep_research_model)
    research_tools = [_prepare_research_tool(t) for t in tools]

    return create_deep_agent(
        model=model,
        tools=research_tools,
        system_prompt=RESEARCH_PROMPT,
        # Declaring the sub-agent explicitly, with the SAME tools, is
        # load-bearing. Left to the default, a spawned sub-agent doesn't get
        # these web tools — but the research prompt still tells it to search,
        # so it confidently calls a web_search that isn't in its request, and
        # the provider rejects the whole call ("attempted to call tool
        # 'web_search' which was not in request.tools"). Observed: the model
        # even invented a plausible-but-wrong signature (topn/source instead
        # of max_results), which is what an LLM does when it "remembers" a
        # tool rather than reading one.
        subagents=[
            {
                "name": "sub_researcher",
                "description": (
                    "Researches one specific, self-contained sub-question and reports "
                    "back concisely. Use one per independent sub-question."
                ),
                "system_prompt": SUB_RESEARCHER_PROMPT,
                "tools": research_tools,
                "model": model,
            }
        ],
        name="deep_research_agent",
    )
