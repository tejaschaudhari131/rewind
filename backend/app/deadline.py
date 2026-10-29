"""Bounded provider collection for the checkpoint gate (adopted from external audit 2026-10-29, C2)."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable


@dataclass(frozen=True)
class ProviderCompletion:
    status: str
    elapsed_ms: float
    values: tuple[Any, ...] = ()


def _consume_finished_task(task):
    if not task.cancelled():
        task.exception()


async def collect_before_deadline(*jobs: Awaitable[Any], started_at: float,
                                  deadline_seconds: float = 2.8):
    """Bound waiting; elapsed includes earlier capture/local work.

    Jobs MUST be pure producers, never state writers. Timeout/cancellation does
    not authorize late writes. Commit still needs a separate token/epoch CAS.
    Move blocking CPU work off the event loop. Network/browser return latency
    and scheduler stalls cannot be hard-bounded by this coroutine.
    """
    loop = asyncio.get_running_loop()
    tasks = [asyncio.create_task(job) for job in jobs]
    try:
        remaining = max(0.0, started_at + deadline_seconds - loop.time())
        done, pending = await asyncio.wait(tasks, timeout=remaining)
        elapsed = (loop.time() - started_at) * 1000
        if pending or elapsed > deadline_seconds * 1000:
            return ProviderCompletion('unknown_timeout', elapsed)
        try:
            values = tuple(task.result() for task in tasks)
        except Exception:
            return ProviderCompletion('unknown_provider_error', elapsed)
        return ProviderCompletion('complete', elapsed, values)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
            task.add_done_callback(_consume_finished_task)

