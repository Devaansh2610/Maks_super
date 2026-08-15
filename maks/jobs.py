"""In-process registry of background jobs — work the supervisor dispatched
and deliberately did *not* wait for.

The point of this file is the "unblocked supervisor": when a long-horizon
task (deep research, a big coding handoff) is requested, the supervisor
registers it here, fires it off, and immediately returns to the user instead
of blocking the whole turn until it finishes. The user can then ask for
something else, ask how the job is going, and get told when it lands.

Deliberately pure state — no speaking, no event-bus calls, no LLM. The
dispatch/announce side lives in maks/graph/supervisor.py, which owns the
specialist instances and the announcement helpers; this module just answers
"what did we kick off, and how is it doing?". That split keeps this
importable from anywhere (dashboard, pipeline, tools) without dragging the
whole agent stack along.

Not persisted across restarts, on purpose: a job is a live asyncio task on
maks/runtime.py's loop, so it cannot survive the process that runs it. A
restart kills the work itself, and a registry entry claiming "still running"
would be a lie. (Making jobs genuinely restart-durable means running them as
LangGraph Platform background runs instead of asyncio tasks — a much bigger
change; see architecture.md.)
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone

# How many finished jobs to keep around for "what happened with that?"
# questions. Old ones are dropped oldest-first — this is a voice assistant's
# short-term working memory, not an audit log.
_MAX_FINISHED = 20

RUNNING = "running"
DONE = "done"
FAILED = "failed"


@dataclass
class Job:
    id: str
    agent: str
    task: str
    status: str = RUNNING
    result: str | None = None
    error: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None
    # What the job has actually been doing, newest last — fed by the
    # callback handler in maks/agents/_common.py as tools fire. Without
    # this a long research sweep is four silent minutes with no way to tell
    # progress from a hang.
    activity: list[str] = field(default_factory=list)
    # How many sub-agents the job has spawned (deepagents' `task` tool).
    # Tracked separately because "it's running three sub-researchers" is the
    # single most useful thing to say about a deep research job in flight.
    subagents_spawned: int = 0

    @property
    def elapsed_seconds(self) -> float:
        end = self.finished_at or datetime.now(timezone.utc)
        return (end - self.created_at).total_seconds()

    @property
    def last_activity(self) -> str | None:
        return self.activity[-1] if self.activity else None

    def progress_line(self) -> str:
        """One spoken-friendly sentence about what's happening right now."""
        minutes = int(self.elapsed_seconds // 60)
        elapsed = f"{minutes} minute{'s' if minutes != 1 else ''} in" if minutes else "just started"
        parts = [f"Job {self.id}, {elapsed}"]
        if self.subagents_spawned:
            parts.append(
                f"{self.subagents_spawned} sub-agent{'s' if self.subagents_spawned != 1 else ''} spawned"
            )
        if self.last_activity:
            parts.append(f"currently {self.last_activity}")
        return " — ".join(parts) + "."

    def describe(self) -> str:
        """One spoken-friendly line. Job ids stay short precisely so they can
        be said out loud and repeated back ("job 3"), not read off a screen.
        """
        minutes = int(self.elapsed_seconds // 60)
        ago = f"{minutes} min" if minutes else "under a minute"
        if self.status == RUNNING:
            return f"Job {self.id} ({self.agent}): still running, {ago} so far — {self.task}"
        if self.status == DONE:
            return f"Job {self.id} ({self.agent}): finished after {ago} — {self.task}"
        return f"Job {self.id} ({self.agent}): failed after {ago} — {self.task} ({self.error})"


_lock = threading.Lock()
_jobs: dict[str, Job] = {}
_next_id = 1

# How many activity entries to keep per job. Only the newest matters for
# "what's it doing now"; the rest is for the dashboard.
_MAX_ACTIVITY = 30

# When the user last said something. Progress heartbeats check this so they
# only speak into silence — interrupting an actual conversation to report
# that a background job is still searching would be worse than saying
# nothing. Updated from maks/pipeline.py on every incoming utterance.
_last_user_activity = datetime.now(timezone.utc)


def mark_user_active() -> None:
    global _last_user_activity
    with _lock:
        _last_user_activity = datetime.now(timezone.utc)


def seconds_since_user_activity() -> float:
    with _lock:
        return (datetime.now(timezone.utc) - _last_user_activity).total_seconds()


def note_activity(job_id: str, description: str, *, is_subagent: bool = False) -> None:
    """Record what a running job is doing. Called from a callback handler as
    tools fire, so it must stay cheap and never raise — a progress-reporting
    failure must not take down the job it's reporting on.
    """
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        job.activity.append(description)
        del job.activity[:-_MAX_ACTIVITY]
        if is_subagent:
            job.subagents_spawned += 1


def create(agent: str, task: str) -> Job:
    global _next_id
    with _lock:
        job = Job(id=str(_next_id), agent=agent, task=task)
        _next_id += 1
        _jobs[job.id] = job
        _prune_locked()
    return job


def complete(job_id: str, result: str) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        job.status = DONE
        job.result = result
        job.finished_at = datetime.now(timezone.utc)


def fail(job_id: str, error: str) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        job.status = FAILED
        job.error = error
        job.finished_at = datetime.now(timezone.utc)


def get(job_id: str) -> Job | None:
    with _lock:
        return _jobs.get(job_id)


def active() -> list[Job]:
    with _lock:
        return [j for j in _jobs.values() if j.status == RUNNING]


def all_jobs() -> list[Job]:
    with _lock:
        return list(_jobs.values())


def summary() -> str:
    """What the supervisor's check_background_jobs tool reads back. Phrased
    as prose rather than a table because it's usually spoken aloud.
    """
    jobs = all_jobs()
    if not jobs:
        return "No background jobs have been dispatched this session."

    running = [j for j in jobs if j.status == RUNNING]
    finished = [j for j in jobs if j.status != RUNNING]

    lines: list[str] = []
    if running:
        lines.append(f"{len(running)} still running:")
        lines += [f"  - {j.describe()}" for j in running]
    else:
        lines.append("Nothing is running right now.")

    if finished:
        lines.append(f"{len(finished)} finished:")
        for j in finished:
            lines.append(f"  - {j.describe()}")
            if j.status == DONE and j.result:
                # The full result matters here -- this is how the user
                # actually collects the output of work they walked away from.
                lines.append(f"    Result: {j.result}")
    return "\n".join(lines)


def _prune_locked() -> None:
    """Drop the oldest finished jobs past _MAX_FINISHED. Caller holds _lock.
    Running jobs are never pruned — they still have a live task behind them.
    """
    finished = sorted(
        (j for j in _jobs.values() if j.status != RUNNING),
        key=lambda j: j.finished_at or j.created_at,
    )
    for job in finished[: max(0, len(finished) - _MAX_FINISHED)]:
        _jobs.pop(job.id, None)
