"""Tests for unfusing transformers-5 OLMoE expert tensors.

Needs torch but not vLLM, so it runs on a laptop. That matters: the bug this guards against loads
weights that are *wrong rather than absent* if the split order is mistaken, and a wrong-but-loaded
policy is far more expensive to discover than a crash.

The central test is :meth:`TestSplitMatchesTheModule.test_split_reproduces_the_modules_own_math` --
it does not check the slicing against my reading of the layout, it checks it against the arithmetic
``OlmoeExperts.forward`` performs.
"""

from __future__ import annotations

import torch

from open_instruct.spec_decode.moe_weights import (
    FUSED_DOWN_SUFFIX,
    FUSED_GATE_UP_SUFFIX,
    stream_is_fused,
    unfuse_olmoe_experts,
)

# OLMoE-1B-7B geometry, from allenai/OLMoE-1B-7B-0125-DPO's config.
NUM_EXPERTS = 4  # smaller than the real 64; the split is per-expert so 4 exercises it
HIDDEN = 8
INTERMEDIATE = 6
LAYER = "model.layers.0"
GATE_UP_NAME = f"{LAYER}{FUSED_GATE_UP_SUFFIX}"
DOWN_NAME = f"{LAYER}{FUSED_DOWN_SUFFIX}"


def fused_gate_up() -> torch.Tensor:
    # transformers: nn.Parameter(torch.empty(num_experts, 2 * intermediate_dim, hidden_dim))
    return torch.arange(NUM_EXPERTS * 2 * INTERMEDIATE * HIDDEN, dtype=torch.float32).reshape(
        NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN
    )


def fused_down() -> torch.Tensor:
    # transformers: nn.Parameter(torch.empty(num_experts, hidden_dim, intermediate_dim))
    return torch.arange(NUM_EXPERTS * HIDDEN * INTERMEDIATE, dtype=torch.float32).reshape(
        NUM_EXPERTS, HIDDEN, INTERMEDIATE
    )


class TestNames:
    def test_gate_up_expands_to_two_names_per_expert(self):
        out = dict(unfuse_olmoe_experts([(GATE_UP_NAME, fused_gate_up())]))
        assert len(out) == 2 * NUM_EXPERTS
        for expert in range(NUM_EXPERTS):
            assert f"{LAYER}.mlp.experts.{expert}.gate_proj.weight" in out
            assert f"{LAYER}.mlp.experts.{expert}.up_proj.weight" in out

    def test_down_expands_to_one_name_per_expert(self):
        out = dict(unfuse_olmoe_experts([(DOWN_NAME, fused_down())]))
        assert set(out) == {f"{LAYER}.mlp.experts.{e}.down_proj.weight" for e in range(NUM_EXPERTS)}

    def test_shapes_are_nn_linear_convention(self):
        out = dict(unfuse_olmoe_experts([(GATE_UP_NAME, fused_gate_up()), (DOWN_NAME, fused_down())]))
        assert out[f"{LAYER}.mlp.experts.0.gate_proj.weight"].shape == (INTERMEDIATE, HIDDEN)
        assert out[f"{LAYER}.mlp.experts.0.up_proj.weight"].shape == (INTERMEDIATE, HIDDEN)
        assert out[f"{LAYER}.mlp.experts.0.down_proj.weight"].shape == (HIDDEN, INTERMEDIATE)


class TestPassthrough:
    def test_already_per_expert_names_are_untouched(self):
        # The Hub checkpoint was saved by transformers 4.x, so the initial load is already
        # per-expert. This transform runs unconditionally, so it must be a no-op for that stream.
        original = [
            (f"{LAYER}.mlp.experts.0.gate_proj.weight", torch.zeros(INTERMEDIATE, HIDDEN)),
            (f"{LAYER}.mlp.experts.0.down_proj.weight", torch.zeros(HIDDEN, INTERMEDIATE)),
            (f"{LAYER}.self_attn.q_proj.weight", torch.zeros(HIDDEN, HIDDEN)),
            (f"{LAYER}.mlp.gate.weight", torch.zeros(NUM_EXPERTS, HIDDEN)),
        ]
        out = list(unfuse_olmoe_experts(original))
        assert [n for n, _ in out] == [n for n, _ in original]

    def test_non_expert_tensors_pass_through(self):
        out = dict(unfuse_olmoe_experts([("lm_head.weight", torch.zeros(3, HIDDEN))]))
        assert list(out) == ["lm_head.weight"]

    def test_stream_is_fused_detects_both_directions(self):
        assert stream_is_fused([GATE_UP_NAME])
        assert stream_is_fused([DOWN_NAME])
        assert not stream_is_fused([f"{LAYER}.mlp.experts.0.gate_proj.weight", "lm_head.weight"])


class TestSplitMatchesTheModule:
    """The split is checked against OlmoeExperts.forward's arithmetic, not against a description."""

    def test_split_reproduces_the_modules_own_math(self):
        # transformers 5.4 OlmoeExperts.forward, for one expert:
        #     gate, up = F.linear(x, gate_up_proj[e]).chunk(2, dim=-1)
        #     h = act(gate) * up
        #     out = F.linear(h, down_proj[e])
        gate_up = fused_gate_up()
        down = fused_down()
        unfused = dict(unfuse_olmoe_experts([(GATE_UP_NAME, gate_up), (DOWN_NAME, down)]))
        x = torch.randn(5, HIDDEN)
        act = torch.nn.functional.silu  # config.hidden_act == "silu"

        for expert in range(NUM_EXPERTS):
            gate_ref, up_ref = torch.nn.functional.linear(x, gate_up[expert]).chunk(2, dim=-1)
            expected = torch.nn.functional.linear(act(gate_ref) * up_ref, down[expert])

            # ...and the same thing computed from the per-expert tensors vLLM will receive.
            gate_w = unfused[f"{LAYER}.mlp.experts.{expert}.gate_proj.weight"]
            up_w = unfused[f"{LAYER}.mlp.experts.{expert}.up_proj.weight"]
            down_w = unfused[f"{LAYER}.mlp.experts.{expert}.down_proj.weight"]
            gate = torch.nn.functional.linear(x, gate_w)
            up = torch.nn.functional.linear(x, up_w)
            actual = torch.nn.functional.linear(act(gate) * up, down_w)

            torch.testing.assert_close(actual, expected)

    def test_a_transposed_split_would_fail_this_test(self):
        # Guards the guard: if gate/up were taken as columns instead of rows, the test above must
        # not still pass. Demonstrated by doing it wrong on purpose.
        gate_up = fused_gate_up()
        x = torch.randn(5, HIDDEN)
        correct_gate = torch.nn.functional.linear(x, gate_up[0][:INTERMEDIATE, :])
        wrong_gate = torch.nn.functional.linear(x, gate_up[0][INTERMEDIATE:, :])
        assert not torch.allclose(correct_gate, wrong_gate)


class TestMatchesVllmsNamingContract:
    """The names produced here have to satisfy vLLM's expert matcher, which is a substring test.

    vLLM 0.21 builds its mapping as ``f"experts.{expert_id}.{weight_name}."`` -- note the trailing
    dot and the absence of any ``weight`` suffix -- and matches with ``if weight_name not in name``
    (``FusedMoE.make_expert_params_mapping`` and ``olmoe.py``'s loader). Pinning that here means a
    future vLLM that changes the pattern breaks a laptop test instead of a weight sync on a paid
    node, which is how this class of bug was found in the first place.
    """

    def test_generated_names_contain_the_substring_vllm_matches_on(self):
        out = dict(unfuse_olmoe_experts([(GATE_UP_NAME, fused_gate_up()), (DOWN_NAME, fused_down())]))
        for expert in range(NUM_EXPERTS):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                vllm_weight_name = f"experts.{expert}.{projection}."
                assert any(vllm_weight_name in name for name in out), (
                    f"no generated name contains {vllm_weight_name!r}, so vLLM's expert mapping "
                    "would skip it and fall through to a bare params_dict lookup"
                )

    def test_fused_names_would_not_match_that_substring(self):
        # The bug, stated as a test: this is why the unfused stream is required at all.
        for projection in ("gate_proj", "up_proj", "down_proj"):
            assert f"experts.0.{projection}." not in GATE_UP_NAME
            assert f"experts.0.{projection}." not in DOWN_NAME


class TestFailsLoudly:
    def test_a_two_dimensional_fused_tensor_is_rejected(self):
        try:
            list(unfuse_olmoe_experts([(GATE_UP_NAME, torch.zeros(2 * INTERMEDIATE, HIDDEN))]))
        except ValueError as error:
            assert "3D" in str(error)
        else:
            raise AssertionError("expected ValueError for a 2D fused tensor")

    def test_an_odd_fused_dimension_is_rejected(self):
        # Cannot be a gate/up concatenation, so refuse rather than split somewhere arbitrary.
        try:
            list(unfuse_olmoe_experts([(GATE_UP_NAME, torch.zeros(NUM_EXPERTS, 7, HIDDEN))]))
        except ValueError as error:
            assert "odd" in str(error)
        else:
            raise AssertionError("expected ValueError for an odd fused dimension")
