"""Shared graph-level constants.

A single long-lived thread id is used for the whole running process so a wake
session (and any quick follow-ups right after it) keeps conversational
context via the supervisor graph's checkpointer.
"""

DEFAULT_THREAD_ID = "maks-main-session"
