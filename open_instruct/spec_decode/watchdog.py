"""Abort a run that has stopped making progress, as opposed to one that has died.

WHY A THREAD AND NOT A CHECK. Three of the five OLMoE probe attempts ended as a hang, and none of
them was detectable by asking whether something was alive:

- attempt 3 retried ``/completions`` against a dead engine for 116 minutes;
- attempt 4 emitted a ``shm_broadcast`` heartbeat every 60 seconds for 12 minutes;
- attempt 5 OOMed in the engine's TP workers, never marked the client ``errored``, and then
  produced the same heartbeat for 21 minutes.

In all three the process was genuinely alive and emitting fresh log lines, so both AWS Batch and a
liveness check reported a healthy run. Only *progress* separates the two.

An in-loop check cannot do this either, because the main thread spends the hang blocked inside
``ray.get``. So this is a daemon thread, which keeps running while the main thread is blocked, and
it exits the process rather than raising: an exception raised on a background thread would be
ignored by a main thread that is not waiting for it.

``os._exit`` deliberately, not ``sys.exit``: the latter raises ``SystemExit`` on this thread only
and would leave the hang untouched. The cost is that atexit handlers and buffered output are
skipped, which is why the diagnosis is printed and flushed first.
"""

from __future__ import annotations

import os
import sys
import threading
import time

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)

#: Generous enough for a slow first step -- image pull, a 13 GB model load, torch.compile and
#: cudagraph capture have all happened before the first step lands, and 20 steps of the probe take
#: about as long again. Short enough that a hang costs minutes rather than a runtime cap. Every hang
#: observed so far was unambiguous within two minutes.
DEFAULT_STALL_SECONDS = 900.0

#: Exit code for a stall. Distinct from 1 so it is greppable in a job record and cannot be confused
#: with an ordinary Python failure.
STALL_EXIT_CODE = 75

_last_progress = time.monotonic()
_lock = threading.Lock()
_started = False


def touch(what: str = "") -> None:
    """Record that something advanced. Cheap enough to call on every step."""
    global _last_progress
    with _lock:
        _last_progress = time.monotonic()
    if what:
        logger.debug("watchdog: progress (%s)", what)


def seconds_since_progress() -> float:
    with _lock:
        return time.monotonic() - _last_progress


def start(stall_seconds: float = DEFAULT_STALL_SECONDS, poll_seconds: float = 30.0) -> None:
    """Start the watchdog once. Subsequent calls are ignored.

    :param stall_seconds: abort when nothing has called :func:`touch` for this long.
    :param poll_seconds: how often to check; only affects how promptly a stall is noticed.
    """
    global _started
    with _lock:
        if _started:
            return
        _started = True
    touch("watchdog started")

    def _watch() -> None:
        while True:
            time.sleep(poll_seconds)
            stalled = seconds_since_progress()
            if stalled < stall_seconds:
                continue
            message = (
                f"watchdog: no progress for {stalled:.0f}s (limit {stall_seconds:.0f}s). "
                "The process is alive but nothing is advancing, which is how attempts 3, 4 and 5 "
                "burned their runtime caps. Aborting so the failure is cheap and legible; read the "
                f"log above this line for the cause. Exiting {STALL_EXIT_CODE}."
            )
            logger.error(message)
            print(message, file=sys.stderr, flush=True)
            sys.stderr.flush()
            sys.stdout.flush()
            # Hard exit: the main thread is blocked in ray.get and will not observe an exception
            # raised here, and a graceful shutdown is exactly what a hung process cannot perform.
            os._exit(STALL_EXIT_CODE)

    threading.Thread(target=_watch, name="progress-watchdog", daemon=True).start()
    logger.info("watchdog: armed, aborting after %.0fs without progress", stall_seconds)


def reset_for_testing() -> None:
    global _started
    with _lock:
        _started = False
    touch()
