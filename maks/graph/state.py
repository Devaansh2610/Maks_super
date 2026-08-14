"""Shared graph-level constants.

A single long-lived thread id is used for the whole running process so a wake
session (and any quick follow-ups right after it) keeps conversational
context via the supervisor graph's checkpointer.
"""

DEFAULT_THREAD_ID = "maks-main-session"

# Stable per-specialist thread ids so a specialist's own create_react_agent
# checkpoint (see maks/agents/_common.py's invoke_specialist thread_id param)
# accumulates history across separate delegations/sticky turns, the same way
# DEFAULT_THREAD_ID does for the router's own conversation.
#
# mac_control_agent is deliberately excluded: its graph
# (maks/graph/dynamic_worker.py) uses an `operator.add`-reduced `results`
# field meant to accumulate only *within* one request's Send-based
# fan-out/aggregate cycle. Giving it a persistent thread id would silently
# accumulate results across unrelated Mac-control requests (e.g. mixing an
# earlier "play a song" into a later "check battery" reply). It stays fully
# isolated per call, exactly as before sticky sessions existed.
SPECIALIST_THREAD_IDS: dict[str, str] = {
    "chit_chat_agent": "specialist-chit_chat_agent",
    "office_agent": "specialist-office_agent",
    "coder_agent": "specialist-coder_agent",
}
