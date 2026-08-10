"""Tests for the progress watchdog.

The central test launches a real subprocess whose main thread is blocked forever, because that is
the situation the watchdog exists for and the one an in-process assertion cannot reproduce: an
exception on a background thread is invisible to a main thread blocked in ``ray.get``. If the
watchdog cannot kill *that*, it does not work, whatever the unit tests say.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import textwrap
import threading
import time

from open_instruct.spec_decode import watchdog

STALL_EXIT_CODE = 75


class TestBookkeeping:
    def setup_method(self):
        watchdog.reset_for_testing()

    def test_touch_resets_the_clock(self):
        time.sleep(0.05)
        before = watchdog.seconds_since_progress()
        watchdog.touch("test")
        assert watchdog.seconds_since_progress() < before

    def test_start_is_idempotent(self):
        # main() may be re-entered in tests or by a resume; a second watchdog thread would double
        # the abort risk for no benefit.
        watchdog.start(stall_seconds=10_000)
        watchdog.start(stall_seconds=10_000)
        threads = [t for t in threading.enumerate() if t.name == "progress-watchdog"]
        assert len(threads) == 1


class TestItKillsABlockedProcess:
    """The property that matters, exercised against a genuinely stuck main thread."""

    def run_child(self, body: str, timeout: float = 60.0) -> subprocess.CompletedProcess:
        script = textwrap.dedent(body)
        return subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=timeout)

    def test_a_main_thread_blocked_forever_is_killed(self):
        # threading.Event().wait() with no timeout is an uninterruptible block, the same shape as
        # a ray.get that will never return.
        result = self.run_child(
            """
            import threading
            from open_instruct.spec_decode import watchdog
            watchdog.start(stall_seconds=2.0, poll_seconds=0.2)
            threading.Event().wait()
            print("UNREACHABLE")
            """
        )
        assert result.returncode == STALL_EXIT_CODE, result.stderr[-2000:]
        assert "UNREACHABLE" not in result.stdout
        assert "no progress" in result.stderr

    def test_the_abort_explains_itself(self):
        # The message is the whole diagnostic value: it has to say what happened and point at where
        # the cause will be, because by definition nothing else printed an error.
        result = self.run_child(
            """
            import threading
            from open_instruct.spec_decode import watchdog
            watchdog.start(stall_seconds=2.0, poll_seconds=0.2)
            threading.Event().wait()
            """
        )
        assert "watchdog" in result.stderr
        assert "read the log above" in result.stderr

    def test_progress_keeps_it_alive(self):
        # The false-positive case, and the one that would be worst: killing a slow but healthy run.
        result = self.run_child(
            """
            import time
            from open_instruct.spec_decode import watchdog
            watchdog.start(stall_seconds=2.0, poll_seconds=0.2)
            for _ in range(15):
                time.sleep(0.3)
                watchdog.touch("still working")
            print("SURVIVED")
            """
        )
        assert result.returncode == 0, result.stderr[-2000:]
        assert "SURVIVED" in result.stdout

    def test_a_fast_run_is_untouched(self):
        result = self.run_child(
            """
            from open_instruct.spec_decode import watchdog
            watchdog.start(stall_seconds=2.0, poll_seconds=0.2)
            print("DONE")
            """
        )
        assert result.returncode == 0
        assert "DONE" in result.stdout


class TestWiring:
    def test_grpo_fast_arms_it_and_marks_each_step(self):
        source = (pathlib.Path(__file__).resolve().parents[2] / "open_instruct" / "grpo_fast.py").read_text()
        assert "spec_decode_watchdog.start()" in source, "the watchdog must be armed before the slow setup"
        assert "spec_decode_watchdog.touch(" in source, "a completed step must count as progress"
