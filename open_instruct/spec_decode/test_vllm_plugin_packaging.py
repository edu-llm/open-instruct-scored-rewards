"""Guards for the entry point that carries the OLMoE override into vLLM's subprocesses.

Runs without vLLM, and checks the packaging rather than the behaviour, because packaging is what
has failed three times: the registration itself has always been correct, and what varied was
whether it reached the process that builds the model.

The history, since it is the reason each assertion exists:

- registering in the Ray actor reached the one process that never builds a model;
- ``sitecustomize`` reached every process but imported vLLM at interpreter startup, deadlocking
  EngineCore's handshake with its workers;
- an entry point is loaded by vLLM in every process at a point of its own choosing, which is right,
  and needs installed metadata, which the image had to be taught to provide.
"""

from __future__ import annotations

import pathlib

import tomllib

PLUGIN_DIR = pathlib.Path(__file__).resolve().parents[2] / "packaging" / "olmoe_vllm_plugin"
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
GROUP = "vllm.general_plugins"


def pyproject() -> dict:
    return tomllib.loads((PLUGIN_DIR / "pyproject.toml").read_text(encoding="utf-8"))


class TestEntryPoint:
    def test_declares_the_group_vllm_loads_in_every_process(self):
        # vllm/plugins/__init__.py: DEFAULT_PLUGINS_GROUP = "vllm.general_plugins", documented there
        # as loaded in process0, the engine core process and the worker processes. Any other group
        # would load somewhere that does not build models.
        entry_points = pyproject()["project"]["entry-points"][GROUP]
        assert entry_points["olmoe_eagle3"] == "olmoe_vllm_plugin:register"

    def test_the_target_callable_exists(self):
        source = (PLUGIN_DIR / "olmoe_vllm_plugin.py").read_text(encoding="utf-8")
        assert "def register()" in source

    def test_it_delegates_rather_than_duplicating_the_registration(self):
        # One definition of what gets registered. A second copy here could drift from the real one
        # and would do so silently, since only the subprocesses would use it.
        source = (PLUGIN_DIR / "olmoe_vllm_plugin.py").read_text(encoding="utf-8")
        assert "registration.register_olmoe_eagle3()" in source
        # Checked as a call, not as a substring: the docstring legitimately discusses ModelRegistry.
        assert "ModelRegistry.register_model(" not in source, "the plugin must delegate, not register directly"


class TestItDoesNotDisturbThePinnedEnvironment:
    def test_declares_no_dependencies(self):
        # vllm and open_instruct are already present wherever this installs. Naming them would let
        # pip resolve or reinstall them and disturb a dependency set pinned deliberately.
        assert pyproject()["project"]["dependencies"] == []

    def test_does_not_use_setuptools_scm(self):
        # .dockerignore excludes .git, so a version derived from git metadata cannot be computed in
        # the image. A static version is what makes this installable there at all.
        # Checked against the parsed build requirements, not the file text: the comments explain
        # why setuptools_scm is avoided, and a substring match would trip on the explanation.
        requires = pyproject()["build-system"]["requires"]
        assert not any("setuptools_scm" in requirement for requirement in requires), requires
        assert pyproject()["project"]["version"]


class TestTheImageInstallsIt:
    def test_dockerfile_installs_the_plugin_after_copying_the_tree(self):
        dockerfile = (REPO_ROOT / ".edullm" / "Dockerfile").read_text(encoding="utf-8")
        copy_at = dockerfile.index("COPY . /opt/open-instruct")
        install_at = dockerfile.index("packaging/olmoe_vllm_plugin")
        assert install_at > copy_at, "the plugin cannot be installed before the tree it lives in"
        assert "--no-deps" in dockerfile[install_at - 300 : install_at + 300]

    def test_the_build_asserts_the_entry_point_is_discoverable(self):
        # Without this the image builds happily and the override silently does not load, which is
        # the failure mode that has cost the most time on this project.
        dockerfile = (REPO_ROOT / ".edullm" / "Dockerfile").read_text(encoding="utf-8")
        assert "vllm.general_plugins" in dockerfile
        assert "entry point missing" in dockerfile


class TestTheOldMechanismsAreGone:
    def test_sitecustomize_is_removed(self):
        # It worked, and it deadlocked vLLM's worker spawn by importing vLLM at interpreter
        # startup. Leaving it beside the entry point would mean registering twice, once at the
        # worst possible moment.
        assert not (REPO_ROOT / "sitecustomize.py").exists()

    def test_the_actor_no_longer_sets_the_sitecustomize_variable(self):
        source = (REPO_ROOT / "open_instruct" / "vllm_utils.py").read_text(encoding="utf-8")
        assert "OPEN_INSTRUCT_REGISTER_OLMOE" not in source
