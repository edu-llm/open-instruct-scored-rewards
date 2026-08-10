"""CPU tests for the per-token weighting. No GPU, no network, no model download.

The properties worth pinning are the ones whose failure is silent. A weighting bug does not
crash: it produces a run that trains, logs a falling loss, and answers a different question
than the one asked.
"""

import math

import pytest
import torch

from projects.kl_reweighted_sft import modeling, weighting


def test_temperature_at_infinity_is_vanilla_sft():
    """T -> inf must reproduce ordinary SFT EXACTLY, since that is the control arm's claim."""
    signals = [[0.1, 5.0, float("nan")], [2.0, 0.3, 0.9]]
    kinds = ["pedagogy", "pedagogy"]
    out = weighting.multipliers_from_signal(signals, kinds, float("inf"))
    assert out == [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]


def test_multipliers_average_to_one():
    """Mean 1 over reweighted tokens is what keeps the effective LR equal to the control's."""
    signals = [[0.1, 5.0, 2.0], [2.0, 0.3, 0.9]]
    kinds = ["pedagogy", "pedagogy"]
    for temperature in (0.5, 1.0, 2.0, 8.0):
        out = weighting.multipliers_from_signal(signals, kinds, temperature)
        flat = [v for row in out for v in row]
        assert math.isclose(sum(flat) / len(flat), 1.0, rel_tol=1e-9)


def test_surprising_tokens_are_downweighted():
    """The whole point: a token the base found unlikely gets LESS gradient, not more."""
    signals = [[0.0, 10.0]]
    out = weighting.multipliers_from_signal(signals, ["pedagogy"], 1.0)
    assert out[0][0] > out[0][1]


def test_colder_temperature_reweights_harder():
    signals = [[0.0, 3.0]]
    spread = []
    for temperature in (4.0, 1.0, 0.25):
        w = weighting.multipliers_from_signal(signals, ["pedagogy"], temperature)[0]
        spread.append(w[0] - w[1])
    assert spread[0] < spread[1] < spread[2]


def test_general_rows_are_never_reweighted():
    """General rows hold the scale still; reweighting them would move the pedagogy:general ratio."""
    signals = [[1.0, 2.0], [float("nan"), float("nan")]]
    out = weighting.multipliers_from_signal(signals, ["pedagogy", "general"], 1.0)
    assert out[1] == [1.0, 1.0]


def test_loss_positions_offset_by_one():
    """labels[i] is the target AT i, predicted by the logit at i-1. Position 0 is never a target."""
    input_ids = [5, 6, 7, 8]
    labels = [-100, -100, 7, 8]
    pos, pred, targets = weighting.loss_positions(input_ids, labels)
    assert pos == [2, 3]
    assert pred == [1, 2]
    assert targets == [7, 8]


def test_robust_zscore_ignores_nan_and_resists_outliers():
    center, scale = weighting.robust_zscore_params([1.0, 2.0, 3.0, 4.0, float("nan"), 1000.0])
    assert math.isclose(center, 3.0)  # the median, not the 202.0 mean
    assert scale < 10.0  # a mean/std scale would be ~446


def test_zero_mad_keeps_the_median_as_centre():
    """With a majority of identical values the MAD is 0 and only the SCALE may fall back.

    Returning the mean here would put the centre at 200.8 -- i.e. hand it to the single outlier
    the median exists to ignore -- and every multiplier downstream would be computed against it.
    """
    center, scale = weighting.robust_zscore_params([1.0, 1.0, 1.0, 1.0, 1000.0])
    assert math.isclose(center, 1.0)
    assert scale > 0.0


def test_variant_a_signal_is_negative_log_prob():
    torch.manual_seed(0)
    logits = torch.randn(4, 11)
    targets = torch.tensor([0, 3, 7, 10])
    got = weighting.signal_from_logits(logits, None, targets, "a", chunk=2)
    want = -torch.log_softmax(logits.float(), dim=-1).gather(1, targets.unsqueeze(1)).squeeze(1)
    torch.testing.assert_close(got, want)


def test_variant_b_signal_is_forward_kl():
    torch.manual_seed(0)
    base, sft = torch.randn(3, 9), torch.randn(3, 9)
    got = weighting.signal_from_logits(base, sft, torch.zeros(3, dtype=torch.long), "b", chunk=2)
    lp0, lp1 = torch.log_softmax(base.float(), -1), torch.log_softmax(sft.float(), -1)
    torch.testing.assert_close(got, (lp0.exp() * (lp0 - lp1)).sum(-1))


def test_forward_kl_is_zero_for_identical_distributions():
    logits = torch.randn(3, 9)
    got = weighting.signal_from_logits(logits, logits.clone(), torch.zeros(3, dtype=torch.long), "b", chunk=8)
    torch.testing.assert_close(got, torch.zeros(3), atol=1e-6, rtol=0)


def test_chunk_shrinks_as_vocab_grows():
    """gpt-oss's 201k vocab must not use the chunk that suited a 100k one."""
    assert weighting.position_chunk(201088) < weighting.position_chunk(100352)
    assert weighting.position_chunk(201088) >= 32


def test_variant_b_requires_the_sft_adapter():
    with pytest.raises(ValueError, match="vanilla SFT"):
        weighting.compute_signal([], "b", model=None, sft_attached=False)


def test_unknown_variant_is_rejected():
    with pytest.raises(ValueError, match="unknown variant"):
        weighting.compute_signal([], "c", model=None, sft_attached=True)


class _Config:
    def __init__(self, model_type, num_local_experts=None, num_experts=None):
        self.model_type = model_type
        if num_local_experts is not None:
            self.num_local_experts = num_local_experts
        if num_experts is not None:
            self.num_experts = num_experts


@pytest.mark.parametrize("model_type", ["gpt_oss", "olmoe", "qwen3_moe"])
def test_supported_moe_types_have_experts_and_a_router_marker(model_type):
    assert modeling.is_moe(_Config(model_type))
    assert modeling.expert_parameters(_Config(model_type))
    assert modeling.ROUTER_MARKERS[model_type]


def test_dense_models_are_not_moe():
    assert not modeling.is_moe(_Config("llama"))
    assert modeling.expert_parameters(_Config("llama")) == []


def test_router_marker_differs_where_the_architecture_does():
    """gpt-oss names its router mlp.router; OLMoE and Qwen3-MoE both name it mlp.gate.

    Using one marker for all three would make the freeze check vacuous on whichever does not
    match, and a vacuous check is worse than none: it reports success.
    """
    assert modeling.ROUTER_MARKERS["gpt_oss"] != modeling.ROUTER_MARKERS["olmoe"]
    assert modeling.ROUTER_MARKERS["qwen3_moe"] == modeling.ROUTER_MARKERS["olmoe"]


def test_expert_rank_scales_down_by_expert_count():
    """r=8 over 32 experts is 32x the budget of r=8 on one dense layer, hence the division."""
    assert modeling.default_expert_rank(_Config("gpt_oss", 32), lora_r=8) == 1
    assert modeling.default_expert_rank(_Config("olmoe", 64), lora_r=64) == 1
    assert modeling.default_expert_rank(_Config("gpt_oss", 4), lora_r=8) == 2
    assert modeling.default_expert_rank(_Config("gpt_oss", 32), lora_r=1) == 1  # never zero


def test_expert_count_read_from_either_config_spelling():
    """gpt-oss says num_local_experts, Qwen3-MoE says num_experts; both must be honoured.

    Missing the field falls back to 1 expert, which would leave the expert rank at the full
    lora_r -- 128x the intended budget on Qwen3-MoE, and no error anywhere.
    """
    assert modeling.default_expert_rank(_Config("qwen3_moe", num_experts=128), lora_r=8) == 1
    assert modeling.default_expert_rank(_Config("qwen3_moe", num_experts=4), lora_r=8) == 2
