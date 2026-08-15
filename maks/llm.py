"""Factories for the models Maks uses: a fast hosted Groq chat model for
every agent, and a small local Ollama embedding model for the fast-path
router (maks/graph/router.py) — nothing else needs an embedding model, so it
lives here next to the chat model factory rather than in its own module.
"""

from __future__ import annotations

from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings

from maks.settings import settings

_embeddings: OllamaEmbeddings | None = None


def get_llm(temperature: float | None = None, model: str | None = None) -> ChatGroq:
    """Return a ChatGroq instance pointed at the configured hosted model.

    A single fast model powers every agent (supervisor + specialists);
    temperature can be overridden per-agent (e.g. lower for tool-heavy
    agents), and `model` can be overridden for work where quality matters
    more than latency. The deep researcher is the one caller that does that
    (see maks/agents/deep_research_agent.py): it runs in the background, so
    nobody is sitting waiting on it, and it drives a much larger tool
    surface than any other agent — exactly the trade where a bigger, slower
    model is worth it.

    Deliberately NOT wrapped in .with_retry() — that returns a RunnableRetry,
    which doesn't expose .bind_tools(), and create_react_agent calls
    model.bind_tools(...) internally. Retry for this model's occasional
    GroqAPIError lives instead in maks/agents/_common.py's invoke_specialist,
    at the one place a retry can't risk re-running an already-executed
    side-effecting tool call.
    """
    return ChatGroq(
        api_key=settings.groq_api_key,
        model=model or settings.groq_model,
        temperature=temperature if temperature is not None else settings.groq_temperature,
    )


def get_embeddings() -> OllamaEmbeddings:
    """Lazy singleton: the local embedding model used only by the fast-path
    router to classify an utterance before deciding whether an LLM call is
    even needed. Kept local (not Groq) since it's cheap, doesn't need
    tool-calling, and avoids a network round trip on the hot path.
    """
    global _embeddings
    if _embeddings is None:
        _embeddings = OllamaEmbeddings(base_url=settings.ollama_host, model=settings.embedding_model)
    return _embeddings
