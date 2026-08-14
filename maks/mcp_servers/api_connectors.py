"""MCP server (stdio): every non-Mac integration Maks has, in one process --
weather + free web search/fetch, Gmail + Google Calendar, Notion, Slack/
WhatsApp/Outlook (stubs), and the Claude Code handoff. One process to start
for testing everything with MCP Inspector instead of four.

(Mac control stays a separate server -- mac_companion/mac_agent.py -- since
it has to run on the Mac itself over streamable-HTTP, with macOS-only
dependencies like osascript/mdfind. Nothing here changes that.)

Run standalone for debugging: python -m maks.mcp_servers.api_connectors
Normally spawned automatically by maks/mcp_client.py.
"""

from __future__ import annotations

import asyncio
import base64
import datetime as dt
import shutil
import subprocess
from email.mime.text import MIMEText
from pathlib import Path

import httpx
from bs4 import BeautifulSoup
from ddgs import DDGS
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from mcp.server.fastmcp import FastMCP
from notion_client import Client

from maks.settings import settings
from maks.tools.weather_tools import get_current_weather

mcp = FastMCP("Maks Connectors")

# Hard ceilings on every tool's "how many results" parameter, regardless of
# what the model asks for. Real incident this guards against: the model
# asked gmail_search for max_results=1000 to answer "how many unread
# emails" by literally listing all of them — full From/Subject/snippet
# metadata for that many messages ballooned a single follow-up model call to
# ~75,000 tokens and blew straight through Groq's free-tier rate limit
# (8,000 TPM). Capping here bounds the worst case regardless of what a model
# decides to request.
_MAX_SEARCH_RESULTS = 25


@mcp.tool()
def weather_lookup(city: str = "") -> str:
    """Get the current live weather for a city. Leave city empty to use the
    user's configured home city. Always use this instead of guessing weather.
    """
    try:
        return get_current_weather(city or None)
    except Exception as exc:  # noqa: BLE001
        return f"Weather lookup failed: {exc}"


@mcp.tool()
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web via DuckDuckGo and return the top results (title, URL,
    snippet). Use this for anything current, factual, or outside your own
    knowledge — news, prices, docs, "what is", "who is", etc.
    """
    max_results = min(max_results, _MAX_SEARCH_RESULTS)
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except Exception as exc:  # noqa: BLE001
        return f"Search failed: {exc}"

    if not results:
        return "No results found."

    lines = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        href = r.get("href", "")
        body = r.get("body", "")
        lines.append(f"{i}. {title}\n   {href}\n   {body}")
    return "\n".join(lines)


@mcp.tool()
def fetch_page(url: str, max_chars: int = 4000) -> str:
    """Fetch a URL and return its readable text content (truncated). Use this
    after web_search when you need the actual content of a specific page,
    not just the snippet.
    """
    max_chars = min(max_chars, 8000)
    try:
        resp = httpx.get(url, timeout=10, follow_redirects=True, headers={"User-Agent": "Mozilla/5.0 (Maks)"})
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        return f"Failed to fetch {url}: {exc}"

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = " ".join(soup.get_text(separator=" ").split())
    return text[:max_chars]


# =================================================================== google ==

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/calendar",
]

_gmail_service = None
_calendar_service = None


def _get_credentials() -> Credentials:
    token_path = Path(settings.google_token_path)
    creds_path = Path(settings.google_credentials_path)
    creds: Credentials | None = None

    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not creds_path.exists():
                raise FileNotFoundError(
                    f"Google OAuth client file not found at {creds_path}. "
                    "See README.md to create one (it's free)."
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(creds_path), SCOPES)
            creds = flow.run_local_server(port=0)
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(creds.to_json(), encoding="utf-8")

    return creds


def _gmail():
    global _gmail_service
    if _gmail_service is None:
        _gmail_service = build("gmail", "v1", credentials=_get_credentials())
    return _gmail_service


def _calendar():
    global _calendar_service
    if _calendar_service is None:
        _calendar_service = build("calendar", "v3", credentials=_get_credentials())
    return _calendar_service


@mcp.tool()
def gmail_search(query: str = "is:unread", max_results: int = 5) -> str:
    """Search Gmail with a Gmail search query (e.g. 'is:unread',
    'from:boss@company.com', 'subject:invoice') and return a short summary
    of matching messages: sender, subject, and snippet — plus the total
    match count. For "how many" questions, use that count directly; don't
    raise max_results trying to list every match to count them yourself —
    it's capped regardless, and the count is already accurate without that.
    """
    max_results = min(max_results, _MAX_SEARCH_RESULTS)
    try:
        svc = _gmail()
        resp = svc.users().messages().list(userId="me", q=query, maxResults=max_results).execute()
        messages = resp.get("messages", [])
        total_estimate = resp.get("resultSizeEstimate", len(messages))
        if not messages:
            return "No matching emails."

        lines = [f"Total matching '{query}': {total_estimate} (showing {len(messages)})"]
        for m in messages:
            msg = svc.users().messages().get(userId="me", id=m["id"], format="metadata",
                                              metadataHeaders=["From", "Subject"]).execute()
            headers = {h["name"]: h["value"] for h in msg["payload"].get("headers", [])}
            lines.append(
                f"From: {headers.get('From', '?')}\n"
                f"Subject: {headers.get('Subject', '(no subject)')}\n"
                f"Snippet: {msg.get('snippet', '')}"
            )
        return "\n\n".join(lines)
    except Exception as exc:  # noqa: BLE001
        return f"Gmail search failed: {exc}"


@mcp.tool()
def gmail_send(to: str, subject: str, body: str) -> str:
    """Send an email from the user's Gmail account."""
    try:
        message = MIMEText(body)
        message["to"] = to
        message["subject"] = subject
        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
        _gmail().users().messages().send(userId="me", body={"raw": raw}).execute()
        return f"Email sent to {to}."
    except Exception as exc:  # noqa: BLE001
        return f"Failed to send email: {exc}"


@mcp.tool()
def calendar_list_events(days_ahead: int = 7) -> str:
    """List the user's upcoming Google Calendar events for the next N days."""
    try:
        now = dt.datetime.utcnow().isoformat() + "Z"
        end = (dt.datetime.utcnow() + dt.timedelta(days=days_ahead)).isoformat() + "Z"
        resp = (
            _calendar()
            .events()
            .list(calendarId="primary", timeMin=now, timeMax=end, singleEvents=True, orderBy="startTime")
            .execute()
        )
        events = resp.get("items", [])
        if not events:
            return f"No events in the next {days_ahead} day(s)."

        lines = []
        for e in events:
            start = e["start"].get("dateTime", e["start"].get("date"))
            lines.append(f"- {e.get('summary', '(no title)')} at {start}")
        return "\n".join(lines)
    except Exception as exc:  # noqa: BLE001
        return f"Failed to list calendar events: {exc}"


@mcp.tool()
def calendar_create_event(summary: str, start_iso: str, end_iso: str, description: str = "") -> str:
    """Create a Google Calendar event. start_iso/end_iso must be ISO 8601
    datetimes, e.g. '2026-08-13T15:00:00'. Assumes the user's local timezone.
    """
    try:
        event = {
            "summary": summary,
            "description": description,
            "start": {"dateTime": start_iso},
            "end": {"dateTime": end_iso},
        }
        created = _calendar().events().insert(calendarId="primary", body=event).execute()
        return f"Created event '{summary}' ({created.get('htmlLink', '')})."
    except Exception as exc:  # noqa: BLE001
        return f"Failed to create calendar event: {exc}"


@mcp.tool()
def whatsapp_send(to: str, message: str) -> str:
    """Send a WhatsApp message. NOT YET IMPLEMENTED — always returns an
    explanation. Do not claim the message was sent.
    """
    return (
        "WhatsApp isn't connected yet — there's no free official API for personal "
        "accounts. This is a stub; tell the user their message could not be sent "
        "and that WhatsApp integration needs to be set up first."
    )


@mcp.tool()
def slack_send(channel: str, message: str) -> str:
    """Send a Slack message. NOT YET IMPLEMENTED — always returns an
    explanation. Do not claim the message was sent.
    """
    return (
        "Slack isn't connected yet — no workspace/bot token has been "
        "configured. This is a stub; tell the user their message could not "
        "be sent and that Slack integration needs to be set up first."
    )


@mcp.tool()
def outlook_send(to: str, subject: str, body: str) -> str:
    """Send an Outlook/Microsoft 365 email. NOT YET IMPLEMENTED — always
    returns an explanation. Do not claim the message was sent.
    """
    return (
        "Outlook isn't connected yet — no Microsoft Graph credentials have "
        "been configured. This is a stub; tell the user their message could "
        "not be sent and that Outlook integration needs to be set up first."
    )


# =================================================================== notion ==

_client: Client | None = None

_TEXT_BLOCK_TYPES = {
    "paragraph", "heading_1", "heading_2", "heading_3",
    "bulleted_list_item", "numbered_list_item", "to_do",
    "quote", "callout", "toggle", "code",
}


def _notion() -> Client:
    global _client
    if _client is None:
        if not settings.notion_token:
            raise RuntimeError("NOTION_TOKEN is not set in .env")
        _client = Client(auth=settings.notion_token)
    return _client


def _flatten_rich_text(rich_text: list) -> str:
    return "".join(t.get("plain_text", "") for t in rich_text)


def _block_to_text(block: dict) -> str:
    block_type = block.get("type")
    if block_type not in _TEXT_BLOCK_TYPES:
        return ""

    payload = block.get(block_type, {})
    text = _flatten_rich_text(payload.get("rich_text", []))
    if not text:
        return ""

    if block_type == "to_do":
        checked = "x" if payload.get("checked") else " "
        return f"[{checked}] {text}"
    if block_type == "bulleted_list_item":
        return f"- {text}"
    if block_type == "numbered_list_item":
        return f"1. {text}"
    if block_type.startswith("heading"):
        return f"## {text}"
    return text


def _property_to_text(prop: dict) -> str:
    ptype = prop.get("type")
    if ptype in ("title", "rich_text"):
        return _flatten_rich_text(prop.get(ptype, []))
    if ptype == "select":
        sel = prop.get("select")
        return sel.get("name", "") if sel else ""
    if ptype == "status":
        st = prop.get("status")
        return st.get("name", "") if st else ""
    if ptype == "multi_select":
        return ", ".join(o.get("name", "") for o in prop.get("multi_select", []))
    if ptype == "number":
        val = prop.get("number")
        return str(val) if val is not None else ""
    if ptype == "checkbox":
        return "yes" if prop.get("checkbox") else "no"
    if ptype == "date":
        d = prop.get("date")
        return d.get("start", "") if d else ""
    if ptype == "people":
        return ", ".join(p.get("name", "") for p in prop.get("people", []) if p.get("name"))
    if ptype in ("url", "email", "phone_number"):
        return prop.get(ptype) or ""
    return ""


@mcp.tool()
def notion_search(query: str, max_results: int = 5) -> str:
    """Search the user's Notion workspace (pages and databases) by title/text."""
    max_results = min(max_results, _MAX_SEARCH_RESULTS)
    try:
        resp = _notion().search(query=query, page_size=max_results)
        results = resp.get("results", [])
        if not results:
            return "No matching Notion pages/databases."

        lines = []
        for item in results:
            title = "(untitled)"
            props = item.get("properties", {})
            for prop in props.values():
                if prop.get("type") == "title" and prop.get("title"):
                    title = "".join(t.get("plain_text", "") for t in prop["title"])
                    break
            lines.append(f"- [{item['object']}] {title} ({item['id']})")
        return "\n".join(lines)
    except Exception as exc:  # noqa: BLE001
        return f"Notion search failed: {exc}"


@mcp.tool()
def notion_create_page(parent_page_id: str, title: str, content: str = "") -> str:
    """Create a new Notion page under the given parent page id, with an
    optional body of plain text content as the first paragraph block.
    """
    try:
        children = []
        if content:
            children.append({
                "object": "block",
                "type": "paragraph",
                "paragraph": {"rich_text": [{"type": "text", "text": {"content": content}}]},
            })
        page = _notion().pages.create(
            parent={"page_id": parent_page_id},
            properties={"title": {"title": [{"text": {"content": title}}]}},
            children=children,
        )
        return f"Created Notion page '{title}' ({page['id']})."
    except Exception as exc:  # noqa: BLE001
        return f"Failed to create Notion page: {exc}"


@mcp.tool()
def notion_append_text(page_id: str, text: str) -> str:
    """Append a plain-text paragraph block to an existing Notion page."""
    try:
        _notion().blocks.children.append(
            block_id=page_id,
            children=[{
                "object": "block",
                "type": "paragraph",
                "paragraph": {"rich_text": [{"type": "text", "text": {"content": text}}]},
            }],
        )
        return "Appended text to the page."
    except Exception as exc:  # noqa: BLE001
        return f"Failed to append to Notion page: {exc}"


@mcp.tool()
def notion_get_page_content(page_id: str, max_blocks: int = 200) -> str:
    """Read the actual text content of a Notion page — headings, paragraphs,
    lists, to-dos, quotes, code blocks. Always call this after notion_search
    when the user asks what's actually IN a page; notion_search only ever
    returns titles, never content, so never answer content questions from a
    search result alone.
    """
    max_blocks = min(max_blocks, 500)
    try:
        lines: list[str] = []
        cursor = None
        fetched = 0
        while fetched < max_blocks:
            resp = _notion().blocks.children.list(block_id=page_id, start_cursor=cursor, page_size=100)
            for block in resp.get("results", []):
                text = _block_to_text(block)
                if text:
                    lines.append(text)
                fetched += 1
                if fetched >= max_blocks:
                    break

            if not resp.get("has_more") or fetched >= max_blocks:
                break
            cursor = resp.get("next_cursor")

        if not lines:
            return (
                "This page has no readable text content (it may only contain "
                "images, embeds, or nested pages/databases)."
            )
        return "\n".join(lines)
    except Exception as exc:  # noqa: BLE001
        return f"Failed to read Notion page content: {exc}"


@mcp.tool()
def notion_query_database(database_id: str, max_results: int = 10) -> str:
    """List rows from a Notion database with their property values. Use this
    (not notion_search, which only returns the database's own title) when
    the user asks what's in a database or wants to see its entries.
    """
    max_results = min(max_results, _MAX_SEARCH_RESULTS)
    try:
        resp = _notion().databases.query(database_id=database_id, page_size=max_results)
        results = resp.get("results", [])
        if not results:
            return "This database has no rows."

        lines = []
        for row in results:
            props = row.get("properties", {})
            parts = [
                f"{name}: {value}"
                for name, prop in props.items()
                if (value := _property_to_text(prop))
            ]
            lines.append("- " + "; ".join(parts) if parts else f"- (row {row.get('id')})")
        return "\n".join(lines)
    except Exception as exc:  # noqa: BLE001
        return f"Failed to query Notion database: {exc}"


# ==================================================================== coder ==

def _run_claude(task: str, project_dir: str) -> str:
    target_dir = Path(project_dir or settings.claude_default_project_dir).expanduser()

    # On Windows, Node-installed CLIs are often `.cmd` shims that
    # subprocess.run() won't resolve the way cmd.exe's own PATH search does
    # (CreateProcess doesn't try PATHEXT extensions). shutil.which() does the
    # right thing on every platform, so resolve explicitly rather than
    # passing the bare name straight to subprocess.
    resolved_bin = shutil.which(settings.claude_cli_bin) or settings.claude_cli_bin

    try:
        result = subprocess.run(
            [resolved_bin, "-p", task, "--output-format", "text"],
            cwd=str(target_dir) if target_dir.exists() else None,
            capture_output=True,
            text=True,
            timeout=900,
        )
    except FileNotFoundError:
        return (
            f"Couldn't find the '{settings.claude_cli_bin}' CLI on this machine. "
            "Make sure Claude Code is installed and on PATH."
        )
    except subprocess.TimeoutExpired:
        return "Claude Code took too long (over 15 minutes) and was stopped. Try a narrower task."

    if result.returncode != 0:
        return f"Claude Code exited with an error:\n{result.stderr.strip()[:2000]}"

    output = result.stdout.strip()
    return output[:4000] if output else "Claude Code finished with no output."


@mcp.tool()
async def run_claude_code(task: str, project_dir: str = "") -> str:
    """Hand a coding task off to Claude Code (writing, editing, debugging, or
    explaining code) and return its final output. `project_dir` optionally
    points Claude at a specific project folder; otherwise the configured
    default project directory is used.

    Wrapped in asyncio.to_thread deliberately: this server now shares one
    process/event loop with every other tool above (web search, Gmail,
    Notion, ...), so this blocking subprocess call must not block them while
    Claude Code runs, which can take a while.
    """
    return await asyncio.to_thread(_run_claude, task, project_dir)


if __name__ == "__main__":
    mcp.run(transport="stdio")
