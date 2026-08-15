"""Typed configuration for Maks, loaded from .env (see .env.example)."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SYSTEM_PROMPT_PATH = PROJECT_ROOT / "config" / "system_prompt.md"

# pydantic-settings' env_file below only populates the typed fields declared
# on Settings — it does NOT export the file into os.environ. Some vars (e.g.
# LANGSMITH_TRACING and friends, read directly by langchain-core) are meant
# to be picked up from the process environment by other libraries, not by
# Settings itself, so load them the plain way too.
load_dotenv(PROJECT_ROOT / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Groq (fast hosted inference — every agent's chat model)
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-20b"
    groq_temperature: float = 0.3
    # The deep researcher only (maks/agents/deep_research_agent.py). It runs
    # in the background, so its latency is nobody's problem, and it drives a
    # far bigger tool surface than any other agent — planning, a virtual
    # filesystem, sub-agents, plus the web tools. The 20b model measurably
    # struggles at that size of tool set (observed: inventing a web_search
    # signature that didn't match the real one); the 120b sibling handles it
    # and keeps the same tool-call format.
    deep_research_model: str = "openai/gpt-oss-120b"

    # Ollama — kept only for local embeddings (the fast-path router), no
    # longer used for chat.
    ollama_host: str = "http://localhost:11434"
    embedding_model: str = "all-minilm"
    router_similarity_threshold: float = 0.72

    # Context budgets (token-aware trimming — see maks/graph/supervisor.py's
    # _trim_history and each specialist's own pre_model_hook). Sized against
    # Groq's free-tier ~8000 tok/min ceiling, split across whichever of the
    # router's own call and one specialist call happen in the same turn.
    router_max_context_tokens: int = 3000

    # Background job progress heartbeats (see maks/graph/supervisor.py's
    # _heartbeat). A deep research sweep runs for minutes; without these
    # it's indistinguishable from a hang. Only spoken when the user has been
    # quiet for job_progress_quiet_seconds, so they never interrupt an
    # actual conversation.
    job_progress_interval_seconds: float = 30.0
    job_progress_quiet_seconds: float = 25.0
    chit_chat_max_context_tokens: int = 1500
    office_max_context_tokens: int = 2500
    coder_max_context_tokens: int = 2500

    # Wake word / voice
    wake_phrase: str = "daddy's home"
    wake_match_threshold: int = 78
    vosk_model_path: str = "./models/vosk-model-small-en-us-0.15"
    whisper_model_size: str = "small.en"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    mic_device: str | None = None
    speaker_device: str | None = None
    # 0 (least aggressive) - 3 (most aggressive) non-speech filtering.
    vad_aggressiveness: int = 3
    command_silence_seconds: float = 0.9
    # After the first wake-phrase activation each run, Maks stays lit and
    # switches to this system-wide "hold Ctrl" shortcut instead of requiring
    # the wake phrase again — see maks/voice/hotkey.py.
    hotkey_hold_seconds: float = 2.0

    # Weather / greeting
    weather_city: str = "Mumbai"
    weather_lat: float | None = None
    weather_lon: float | None = None

    # Dashboard
    dashboard_host: str = "0.0.0.0"
    dashboard_port: int = 8420

    # Mac companion
    mac_companion_url: str = "http://192.168.1.50:8765"
    mac_companion_token: str = ""
    # The MCP streamable-HTTP client's own default is 30s -- fine for the
    # real app (paid once, at startup, in the background) but far too slow
    # for `langgraph dev`, which reconnects on every graph-introspection
    # call (see maks/mcp_client.py's load_tools_stateless()) -- so an
    # unreachable companion (no Mac around, the common case) turns into a
    # 30s tax on every single Studio interaction instead of a one-time cost.
    mac_companion_connect_timeout: float = 3.0

    # Google
    google_credentials_path: str = "./secrets/google_credentials.json"
    google_token_path: str = "./secrets/google_token.json"

    # Notion
    notion_token: str = ""

    # Claude handoff
    claude_default_project_dir: str = "~/projects"
    claude_cli_bin: str = "claude"
    # run_claude_code (maks/mcp_servers/api_connectors.py) runs Claude Code
    # headlessly, so there's no user watching to approve a tool call that
    # needs permission -- it'll hang instead of prompting. Set this (e.g.
    # "acceptEdits", "bypassPermissions") to opt into a specific permission
    # mode for that headless run; left empty (the CLI's own interactive
    # default) rather than silently choosing a permissive one.
    claude_permission_mode: str = ""

    # Fish Audio (cloud TTS)
    fish_audio_api_key: str = ""
    fish_audio_model: str = "s2.1-pro-free"
    fish_audio_voice_id: str = ""

    def read_system_prompt(self) -> str:
        try:
            return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return "You are Maks, a helpful personal voice assistant."


settings = Settings()
