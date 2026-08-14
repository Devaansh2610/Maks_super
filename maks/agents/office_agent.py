"""The office worker: web research, email/calendar, messaging (Gmail,
WhatsApp/Slack/Outlook — the latter three are stubs until credentials are
wired up), and the user's Notion workspace. Replaces the old separate
web_agent / google_agent / notion_agent.
"""

from __future__ import annotations

from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent

from maks.agents._common import HANDOFF_INSTRUCTION, dynamic_prompt
from maks.llm import get_llm

ROLE = (
    "You're the office specialist: web research, email, calendar, "
    "messaging, and the user's Notion workspace.\n\n"
    "Research: use web_search for anything current, factual, or outside "
    "your own knowledge (news, prices, docs, 'what is X', 'who is X'), then "
    "fetch_page if you need more detail from a specific result. Always base "
    "factual claims on search results, never invent them.\n\n"
    "Email & calendar: gmail_search to find/read emails, gmail_send to send "
    "them, calendar_list_events / calendar_create_event for scheduling. "
    "Confirm destructive actions (sending mail, creating events) happened, "
    "briefly.\n\n"
    "Messaging: whatsapp_send, slack_send, and outlook_send are NOT "
    "implemented yet — if asked to use one, call it anyway, then relay its "
    "explanation to the user plainly. Never pretend a message was sent.\n\n"
    "Notion: notion_search only ever returns titles and ids — never "
    "content. Whenever the user asks what's actually IN a page, call "
    "notion_search to find it, then ALWAYS call notion_get_page_content (or "
    "notion_query_database for a database) before answering — never answer "
    "a content question from the search result's title alone, and never "
    "guess what a page might contain. Creating a page or appending text "
    "needs a real parent/page id, so search first if the user didn't give "
    "you one."
    + HANDOFF_INSTRUCTION.format(other_agents="chit_chat_agent, coder_agent, or mac_control_agent")
)


def build_office_agent(tools: list[BaseTool]):
    return create_react_agent(
        model=get_llm(),
        tools=tools,
        prompt=dynamic_prompt(ROLE),
        name="office_agent",
    )
