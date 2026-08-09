"""Guards for the two bugs that made run_019fe36d cost two hours and report nothing.

Neither is about speculative decoding; both live here because this is where the fork's tests for
the vLLM integration already are, and `conftest.py` already knows to skip this directory's
vLLM-dependent files where vLLM is absent.

WHAT HAPPENED. The engine died 3.7 minutes in. Two independent defects turned that into a full
two-hour cap with no diagnosis:

1. ``engine_error_handler`` reports failures by reading ``req.app.state.server``. vLLM's own
   ``serve_http()`` sets that in ``entrypoints/launcher.py``; open-instruct drives uvicorn directly
   and did not, so the handler raised ``AttributeError`` *while handling the error* and the original
   exception was lost.
2. Nothing noticed the engine was gone. The OpenAI client is constructed with ``timeout=3600``, so
   each request hung for an hour before retrying, and Batch reported ``RUNNING`` throughout because
   the process was alive and looping.

The second is the expensive one: it converts any four-minute failure into a full-cap bill.
"""

from __future__ import annotations

import inspect
import types

import pytest

from open_instruct import vllm_utils


class DeadEngine:
    """Stands in for an ``AsyncLLM`` that has failed. Mirrors its liveness API."""

    def __init__(self, cause: BaseException):
        self.errored = True
        self.dead_error = cause


class LiveEngine:
    def __init__(self):
        self.errored = False
        self.dead_error = AssertionError("should not be raised while the engine is healthy")


class DoneFuture:
    def done(self):
        return True

    def result(self):
        return None


def actor_with(engine) -> vllm_utils.LLMRayActor:
    """An LLMRayActor with only the attributes check_background_threads touches.

    Built without __init__ on purpose: the real one starts an engine and a uvicorn server, and the
    method under test is deliberately independent of all of that.
    """
    actor = object.__new__(vllm_utils.LLMRayActor)
    actor.llm_engine = engine
    actor._prefetch_future = DoneFuture()
    actor._process_future = DoneFuture()
    actor.active_tasks = {}
    actor.loop_thread = types.SimpleNamespace(is_alive=lambda: True)
    return actor


class TestDeadEngineAborts:
    def test_a_dead_engine_raises_its_own_cause(self):
        # The cause matters as much as the raise: it is what tells the next reader whether the
        # engine died of a weight-sync failure, an OOM, or something else entirely.
        cause = RuntimeError("EngineCore died: something specific and diagnosable")
        with pytest.raises(RuntimeError, match="something specific and diagnosable"):
            actor_with(DeadEngine(cause)).check_background_threads()

    def test_a_healthy_engine_does_not_raise(self):
        actor_with(LiveEngine()).check_background_threads()

    def test_an_engine_without_the_liveness_api_is_tolerated(self):
        # Older or stubbed engines may not expose `errored`. Absent means "no evidence of death",
        # not "assume dead" -- this check must never be the thing that fails a working run.
        actor_with(types.SimpleNamespace()).check_background_threads()

    def test_no_engine_yet_is_tolerated(self):
        actor_with(None).check_background_threads()

    def test_engine_death_is_checked_before_the_thread_checks(self):
        # Ordering is deliberate: a dead engine is usually what killed the loop thread, so its
        # cause is the more informative of the two. If the thread check ran first it would mask
        # the engine's error with a generic "loop thread has died".
        actor = actor_with(DeadEngine(RuntimeError("engine cause")))
        actor.loop_thread = types.SimpleNamespace(is_alive=lambda: False)
        with pytest.raises(RuntimeError, match="engine cause"):
            actor.check_background_threads()


class TestErrorHandlerCanReport:
    def test_the_server_is_bound_to_app_state(self):
        # Asserted on the source because the real path builds an engine. What matters is that the
        # uvicorn Server is named and attached rather than constructed inline and discarded --
        # `uvicorn.Server(config).serve()` was the original, and it is why app.state.server was
        # never set.
        source = inspect.getsource(vllm_utils.LLMRayActor._setup_and_start_async_engine)
        assert "app.state.server = server" in source, (
            "vLLM's engine_error_handler reads app.state.server; without it every engine error "
            "is replaced by an AttributeError raised from inside the handler"
        )
        assert "uvicorn.Server(config).serve()" not in source, (
            "the Server must be bound to a name so it can be attached to app.state, not "
            "constructed inline and discarded"
        )
