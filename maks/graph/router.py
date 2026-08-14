"""Embedding-based fast-path router: for an obvious, single-intent request,
route straight to the right specialist with a cosine-similarity lookup
against a local embedding model (all-minilm via Ollama) — no LLM call at
all. Anything ambiguous or that looks like it spans more than one agent
falls back to `agent=None`, which tells the caller (maks/pipeline.py) to run
the full supervisor instead, which can actually reason about it.

Deliberately conservative: guessing wrong here means silently doing the
wrong thing, so the bar to short-circuit is "clearly, single-agent obvious"
— everything else pays for the (still fast) supervisor call.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from maks.llm import get_embeddings
from maks.settings import settings

# A handful of representative example utterances per route. Kept short and
# unambiguous on purpose — these are centroids, not an exhaustive intent
# catalog; anything that doesn't clearly resemble one of these should (and,
# per the threshold below, will) fall back to the supervisor.
_ROUTE_EXAMPLES: dict[str, list[str]] = {
    "chit_chat_agent": [
        "hello, how are you",
        "tell me a joke",
        "what do you think about that",
        "good morning",
        "what's the capital of France",
        "thanks, that's all",
    ],
    "office_agent": [
        "what's the weather going to be tomorrow",
        "search the web for the latest news on",
        "check my email for anything from my boss",
        "send an email to",
        "what's on my calendar today",
        "schedule a meeting for",
        "what does my notion page about the project say",
        "add a note to my notion database",
        "send a slack message to the team",
    ],
    "coder_agent": [
        "fix the bug in this function",
        "write a script that",
        "review this pull request",
        "explain what this code does",
        "add a new feature to my project",
        "debug why this test is failing",
    ],
    "mac_control_agent": [
        "play some music on spotify",
        "play a video on youtube",
        "pause the music",
        "find a file on my mac",
        "check how much battery my mac has left",
        "how's my mac doing",
    ],
}

# Above this on the *second*-best route (relative to the best), treat the
# request as ambiguous/compound rather than risk routing it wrong.
_AMBIGUOUS_MARGIN = 0.05

_COMPOUND_CUES = (
    " and then ", " after that ", " and also ", " then also ",
    " and once ", " once you're done ",
)

_embedded_examples: dict[str, list[list[float]]] | None = None


@dataclass
class RouteDecision:
    agent: str | None
    confidence: float


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


async def _get_embedded_examples() -> dict[str, list[list[float]]]:
    global _embedded_examples
    if _embedded_examples is None:
        embeddings = get_embeddings()
        _embedded_examples = {
            agent: await embeddings.aembed_documents(examples)
            for agent, examples in _ROUTE_EXAMPLES.items()
        }
    return _embedded_examples


def _looks_compound(text: str) -> bool:
    lowered = f" {text.lower()} "
    return any(cue in lowered for cue in _COMPOUND_CUES)


async def route(text: str) -> RouteDecision:
    """Classify `text` against the known routes. Returns agent=None (falls
    back to the full supervisor) when confidence is too low, the text looks
    compound, or the top two routes are too close to call.
    """
    if _looks_compound(text):
        return RouteDecision(agent=None, confidence=0.0)

    examples_by_agent = await _get_embedded_examples()
    embeddings = get_embeddings()
    query_vec = await embeddings.aembed_query(text)

    scores: list[tuple[str, float]] = []
    for agent, vectors in examples_by_agent.items():
        best_for_agent = max(_cosine(query_vec, v) for v in vectors)
        scores.append((agent, best_for_agent))
    scores.sort(key=lambda pair: pair[1], reverse=True)

    best_agent, best_score = scores[0]
    second_score = scores[1][1] if len(scores) > 1 else 0.0

    if best_score < settings.router_similarity_threshold:
        return RouteDecision(agent=None, confidence=best_score)
    if best_score - second_score < _AMBIGUOUS_MARGIN:
        return RouteDecision(agent=None, confidence=best_score)

    return RouteDecision(agent=best_agent, confidence=best_score)
