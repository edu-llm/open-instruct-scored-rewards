"""Import every module in this package, and the upstream chain that depends on it.

WHY THIS IS SEPARATE FROM THE OTHER TESTS. ``test_imports_resolve.py`` checks that names moved
between *our* modules still line up, using the AST, so it runs without vLLM. It cannot check that
the names we import from *vLLM* still exist -- that needs vLLM, and a symbol vLLM renamed is
invisible until something imports it on a GPU node.

The 2026-08-08 baseline runs made the gap concrete in a second way: the sweep script imports only
``spec_decode.config``, so ``spec_decode.metrics`` was never imported by anything that ran. Its
vLLM imports were therefore unverified, and ``vllm_utils`` imports it -- meaning a stale symbol in
``metrics`` would break every ``grpo_fast`` run, discovered at engine startup rather than here.

So this imports each module explicitly, plus ``open_instruct.vllm_utils`` as the end of the chain
that ``grpo_fast`` actually walks. It is a few seconds of a GPU job and it converts a class of
100-seconds-into-a-paid-run failure into a test.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect

import pytest

# Every module in open_instruct/spec_decode/ that is not itself a test.
SPEC_DECODE_MODULES = [
    "open_instruct.spec_decode",
    "open_instruct.spec_decode.config",
    "open_instruct.spec_decode.metrics",
    "open_instruct.spec_decode.olmoe_eagle3",
    "open_instruct.spec_decode.registration",
    "open_instruct.spec_decode.reporting",
]


@pytest.mark.parametrize("module_name", SPEC_DECODE_MODULES)
def test_module_imports(module_name: str):
    assert importlib.import_module(module_name) is not None


def test_vllm_utils_imports():
    """The chain grpo_fast walks: vllm_utils imports spec_decode.metrics and .registration.

    Asserted separately and by name because this is the import whose failure would be most
    expensive: it does not touch speculative decoding at all, so it would break ordinary RL runs
    that never asked for a draft.
    """
    vllm_utils = importlib.import_module("open_instruct.vllm_utils")
    # The two hooks the fork adds. If either name moves, this fails here rather than at the
    # moment an engine is built.
    assert hasattr(vllm_utils, "create_vllm_engines")
    assert hasattr(vllm_utils.LLMRayActor, "drain_spec_decode_metrics")


def test_create_vllm_engines_accepts_the_new_parameters():
    """The two parameters the fork threads through, checked on the signature.

    A rebase that drops them would otherwise surface as a TypeError inside a Ray actor, which is
    a much worse place to read an error from.
    """
    vllm_utils = importlib.import_module("open_instruct.vllm_utils")
    parameters = inspect.signature(vllm_utils.create_vllm_engines).parameters
    for name in ("speculative_config", "collect_spec_decode_stats"):
        assert name in parameters, f"create_vllm_engines lost the {name!r} parameter"


def test_vllm_config_exposes_the_speculative_flags():
    """The CLI surface. ArgumentParserPlus derives flags from these dataclass fields."""
    data_loader = importlib.import_module("open_instruct.data_loader")
    fields = {f.name for f in dataclasses.fields(data_loader.VLLMConfig)}
    expected = {
        "vllm_speculative_method",
        "vllm_speculative_model",
        "vllm_num_speculative_tokens",
        "vllm_speculative_draft_tensor_parallel_size",
        "vllm_collect_spec_decode_stats",
    }
    assert expected <= fields, f"VLLMConfig is missing {sorted(expected - fields)}"
    # And that an unflagged config is byte-identical to upstream's behaviour: no speculation.
    assert data_loader.VLLMConfig().speculative_config() is None
