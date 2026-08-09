"""Tests for the interpreter-startup hook that reaches vLLM's subprocesses.

Runs without vLLM. That is the point of testing it here rather than only on a node: the hook is
executed by *every* Python process in the repository, so the properties that matter most are
"stays out of the way when not wanted" and "never breaks interpreter startup", and both are
checkable on a laptop.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
ENABLE_VAR = "OPEN_INSTRUCT_REGISTER_OLMOE"


def run_python(env_extra: dict[str, str]) -> subprocess.CompletedProcess:
    """Start a fresh interpreter with the repo on PYTHONPATH, as the platform image does."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    env.pop(ENABLE_VAR, None)
    env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-c", "print('interpreter-ok')"], capture_output=True, text=True, env=env, timeout=120
    )


class TestGating:
    def test_does_nothing_without_the_env_var(self):
        # Every DeepSpeed learner process imports this file too. Importing vLLM there would cost
        # seconds per process for nothing, so the default must be to return immediately.
        result = run_python({})
        assert result.returncode == 0
        assert "interpreter-ok" in result.stdout
        assert "could not register" not in result.stderr

    def test_attempts_registration_when_enabled(self):
        # On a machine without vLLM the attempt fails, which is the interesting case: it must warn
        # and still let the interpreter start. On a node it succeeds silently.
        result = run_python({ENABLE_VAR: "1"})
        assert result.returncode == 0, f"interpreter startup broke: {result.stderr}"
        assert "interpreter-ok" in result.stdout

    def test_a_registration_failure_is_never_fatal(self):
        # The failure mode this guards against is catastrophic and easy to cause: an exception at
        # interpreter startup would break every process in the repo, including ones with no
        # connection to vLLM.
        result = run_python({ENABLE_VAR: "1"})
        assert result.returncode == 0
        assert "Traceback" not in result.stderr

    def test_a_failure_says_what_was_lost(self):
        # Only meaningful where vLLM is absent, which is any laptop. A silent failure here would
        # produce a run that looks configured and quietly uses upstream's model class.
        result = run_python({ENABLE_VAR: "1"})
        if "could not register" in result.stderr:
            assert "unfuse" in result.stderr or "EAGLE-3" in result.stderr


class TestItIsWhereTheImageWillFindIt:
    def test_sitecustomize_sits_at_the_repo_root(self):
        # PYTHONPATH is /opt/open-instruct, so `sitecustomize` is importable only from the root.
        assert (REPO_ROOT / "sitecustomize.py").is_file()

    def test_it_reads_the_same_variable_the_actor_sets(self):
        source = (REPO_ROOT / "sitecustomize.py").read_text(encoding="utf-8")
        actor = (REPO_ROOT / "open_instruct" / "vllm_utils.py").read_text(encoding="utf-8")
        assert ENABLE_VAR in source
        assert f'"{ENABLE_VAR}": "1"' in actor, (
            "create_vllm_engines must set the variable sitecustomize gates on, or the override "
            "never reaches EngineCore and its workers"
        )
