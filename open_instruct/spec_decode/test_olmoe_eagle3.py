"""Guards for the OLMoE EAGLE-3 registration.

Two kinds of test here, and the second kind is the point of the file.

The ordinary kind checks that our classes do what vLLM asks of an EAGLE-3 target. The other
kind checks the *assumptions we make about upstream*, because this module subclasses a class
that ``@support_torch_compile`` mutates and restates one ``__init__`` it cannot inherit. Each
of those assumptions fails silently if vLLM changes -- a target that quietly stops returning
auxiliary hidden states leaves speculative decoding configured and inert, and a restated
``__init__`` that misses a newly added attribute leaves a half-built model. So they are
asserted rather than trusted.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

import pytest
from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper
from vllm.model_executor.models.interfaces import EagleModelMixin, supports_eagle3
from vllm.model_executor.models.olmoe import OlmoeForCausalLM, OlmoeModel
from vllm.model_executor.models.registry import ModelRegistry

from open_instruct.spec_decode.olmoe_eagle3 import OlmoeForCausalLMEagle3, OlmoeModelEagle3
from open_instruct.spec_decode.registration import (
    OLMOE_ARCH,
    OLMOE_EAGLE3_MODULE,
    assert_registered,
    register_olmoe_eagle3,
)

# OLMoE-1B-7B has 16 layers, and SupportsEagle3's default is (2, n // 2, n - 3).
OLMOE_NUM_LAYERS = 16
EXPECTED_AUX_LAYERS = (2, 8, 13)


@pytest.fixture
def clean_registry():
    """Restore the registry entry for OLMoE, so one test's registration cannot leak."""
    original = ModelRegistry.models.get(OLMOE_ARCH)
    yield
    if original is not None:
        ModelRegistry.models[OLMOE_ARCH] = original


def _assigned_self_attrs(func) -> set[str]:
    """Names assigned to ``self.<name>`` in a function body."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    return {
        target.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self"
    }


class TestEagle3Contract:
    def test_our_target_supports_eagle3_and_upstream_does_not(self):
        # The second half is what makes the first meaningful: if upstream ever gains EAGLE-3
        # support, this whole module becomes dead weight and should be deleted rather than
        # maintained. This test is where that shows up.
        assert supports_eagle3(OlmoeForCausalLMEagle3)
        assert not supports_eagle3(OlmoeForCausalLM), (
            "vLLM's OlmoeForCausalLM now supports EAGLE-3 on its own. Prefer upstream and "
            "delete open_instruct/spec_decode/olmoe_eagle3.py."
        )

    def test_inner_model_is_an_eagle_mixin(self):
        # SupportsEagle3.set_aux_hidden_state_layers asserts isinstance(parent.model,
        # EagleModelMixin) before it will configure anything.
        assert issubclass(OlmoeModelEagle3, EagleModelMixin)

    def test_default_aux_layers_are_early_middle_late(self):
        layers = OlmoeForCausalLMEagle3.get_eagle3_default_aux_hidden_state_layers
        stub = _StubForAuxLayers(OLMOE_NUM_LAYERS)
        assert layers(stub) == EXPECTED_AUX_LAYERS

    def test_forward_returns_bare_tensor_until_layers_are_requested(self):
        # The 2-tuple is only correct for an EAGLE-3 drafter. Every other caller, including
        # an ordinary rollout with no speculative config, expects one tensor.
        source = inspect.getsource(OlmoeModelEagle3.forward)
        assert "if len(aux_hidden_states) > 0:" in source
        assert source.rstrip().endswith("return hidden_states")


class TestUpstreamAssumptions:
    def test_compile_wrapper_is_inherited_exactly_once(self):
        # Re-applying @support_torch_compile to the subclass would append the wrapper a
        # second time, because the decorator's guard reads direct __bases__ only.
        occurrences = sum(base is TorchCompileWithNoGuardsWrapper for base in OlmoeModelEagle3.__mro__)
        assert occurrences == 1, f"wrapper appears {occurrences}x in the MRO; is the subclass decorated?"

    def test_subclass_does_not_define_its_own_init(self):
        # Inheriting __init__ is what keeps the compile wrapper's setup running exactly once.
        assert "__init__" not in OlmoeModelEagle3.__dict__

    def test_weight_loading_is_inherited_not_copied(self):
        # OlmoeModel.load_weights maps 64 experts per layer and the stacked qkv_proj. A copy
        # of it going stale would mis-load weights in silence, so assert we never copied it.
        for method in ("load_weights", "get_expert_mapping"):
            assert method not in OlmoeModelEagle3.__dict__, f"{method} should be inherited, not copied"
            assert getattr(OlmoeModelEagle3, method) is getattr(OlmoeModel, method)

    def test_restated_init_sets_exactly_what_upstream_sets(self):
        # The one method we could not inherit, because upstream hard-codes
        # `self.model = OlmoeModel(...)`. If upstream starts setting another attribute, our
        # copy would leave it unset and the model would be half-built.
        upstream = _assigned_self_attrs(OlmoeForCausalLM.__init__)
        ours = _assigned_self_attrs(OlmoeForCausalLMEagle3.__init__)
        assert ours == upstream, (
            "OlmoeForCausalLMEagle3.__init__ has drifted from upstream's.\n"
            f"  missing from ours: {sorted(upstream - ours)}\n"
            f"  extra in ours:     {sorted(ours - upstream)}"
        )

    def test_upstream_init_still_hardcodes_the_model_class(self):
        # The reason the restatement exists at all. If upstream ever takes the inner model
        # class as a parameter, the copy can go away.
        assert "self.model = OlmoeModel(" in inspect.getsource(OlmoeForCausalLM.__init__)


class TestRegistration:
    def test_register_then_assert_passes(self, clean_registry):
        register_olmoe_eagle3()
        assert_registered()

    def test_registration_is_lazy_by_string(self, clean_registry):
        # A class object here would import torch's CUDA state into the parent process, which
        # breaks the forked engine workers open-instruct runs.
        register_olmoe_eagle3()
        entry = ModelRegistry.models[OLMOE_ARCH]
        assert entry.module_name == OLMOE_EAGLE3_MODULE
        assert entry.class_name == "OlmoeForCausalLMEagle3"

    def test_register_is_idempotent(self, clean_registry):
        register_olmoe_eagle3()
        first = ModelRegistry.models[OLMOE_ARCH]
        register_olmoe_eagle3()
        assert ModelRegistry.models[OLMOE_ARCH] is first

    def test_assert_registered_raises_when_upstream_is_registered(self, clean_registry):
        # The silent failure this exists to catch: a worker that never ran the registration
        # holds upstream's class, and would draft against a target with no aux hidden states.
        ModelRegistry.register_model(OLMOE_ARCH, "vllm.model_executor.models.olmoe:OlmoeForCausalLM")
        with pytest.raises(RuntimeError, match="is registered to"):
            assert_registered()


class _StubForAuxLayers:
    """Minimal stand-in for the parts of a model that the default-layers helper reads."""

    def __init__(self, num_layers: int):
        self.model = type("_M", (), {"layers": [None] * num_layers})()
