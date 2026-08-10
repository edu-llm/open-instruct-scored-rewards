"""CPU tests for atlas tracing, delta metrics, and the control-first analysis.

No model is downloaded and no GPU is used. Toy tokenizers pin the position
logic, toy mixture-of-experts blocks pin the routing capture, and synthetic
traces pin every metric against a value worked out by hand.
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from projects.learning_science_semantic_atlas import analyze_representations as analysis
from projects.learning_science_semantic_atlas import checkpoint_delta as delta
from projects.learning_science_semantic_atlas.trace import (
    ROUTER_FIDELITY_TOLERANCE,
    AtlasTrace,
    build_replay,
    collect_record,
    expand_to_slots,
    load_atlas_trace,
    router_fidelity,
    routing_paths,
    sha256_text,
    slot_plan,
    surface_features,
    token_histogram,
    trace_records,
    write_router_weights,
    write_trace,
)
from projects.pedagogy_mech_interp.modeling import decoder_layers, route, selected_expert_contributions
from projects.pedagogy_mech_interp.trace import TraceCollector
from torch import nn

PROMPT = "PROMPT>"
RESPONSE = "Hi there. Next step?"


# --------------------------------------------------------------------------
# toy tokenizers
# --------------------------------------------------------------------------


class Encoding(dict):
    def __getattr__(self, name):
        return self[name]


class CharacterTokenizer:
    """One token per character, with exact offsets."""

    group = 1

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        body = "|".join(f"{message['role']}:{message['content']}" for message in messages)
        return f"{body}|assistant:" if add_generation_prompt else body

    def spans(self, text):
        return [(start, min(start + self.group, len(text))) for start in range(0, len(text), self.group)]

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False, return_tensors=None):
        spans = self.spans(text)
        data = Encoding(
            input_ids=torch.tensor([[ord(text[start]) for start, _ in spans]]),
            attention_mask=torch.ones(1, len(spans), dtype=torch.long),
        )
        if return_offsets_mapping:
            data["offset_mapping"] = torch.tensor([spans])
        return data


class PairTokenizer(CharacterTokenizer):
    """Two characters per token, so an odd-length prompt straddles the boundary."""

    group = 2


def record(prompt=PROMPT, response=RESPONSE, **extra):
    return {"record_id": "r1", "prompt": prompt, "response": response, **extra}


# --------------------------------------------------------------------------
# position resolution
# --------------------------------------------------------------------------


def test_slot_plan_lists_prompt_generated_boundary_then_content_last():
    plan = slot_plan(first_k=2, max_boundaries=2)
    assert plan.roles == ("prompt_last", "generated_0", "generated_1", "boundary_0", "boundary_1", "content_last")
    assert plan.bases == ("prompt_last", "generated", "generated", "boundary", "boundary", "content_last")


def test_replay_resolves_every_role_to_the_right_character():
    plan = slot_plan(first_k=3, max_boundaries=2)
    replay = build_replay(CharacterTokenizer(), record(), plan, max_len=64, first_k=3, max_boundaries=2)
    characters = [replay.text[position] if position >= 0 else None for position in replay.slot_positions]
    # "PROMPT>" then "Hi there. Next step?"
    assert characters == [">", "H", "i", " ", ".", "?", "?"]
    assert replay.content_start == len(PROMPT)
    assert replay.content_stop == len(PROMPT) + len(RESPONSE)
    assert replay.boundary_count == 2
    assert not replay.merged_prompt_boundary


def test_absent_boundaries_stay_masked_instead_of_falling_back():
    plan = slot_plan(first_k=1, max_boundaries=3)
    replay = build_replay(
        CharacterTokenizer(), record(response="no terminator here"), plan, max_len=64, first_k=1, max_boundaries=3
    )
    boundary_slots = [index for index, base in enumerate(plan.bases) if base == "boundary"]
    assert [replay.slot_positions[slot] for slot in boundary_slots] == [-1, -1, -1]
    assert replay.valid_slots == [0, 1, 5]


def test_generated_slots_beyond_the_response_are_masked():
    plan = slot_plan(first_k=5, max_boundaries=0)
    replay = build_replay(CharacterTokenizer(), record(response="ab"), plan, max_len=64, first_k=5, max_boundaries=0)
    assert [replay.slot_positions[slot] for slot in (1, 2, 3, 4, 5)] == [7, 8, -1, -1, -1]


def test_truncation_that_eats_the_prompt_fails_closed():
    plan = slot_plan(first_k=1, max_boundaries=0)
    with pytest.raises(ValueError, match="prompt token"):
        build_replay(CharacterTokenizer(), record(), plan, max_len=len(RESPONSE), first_k=1, max_boundaries=0)


def test_left_truncation_moves_the_window_without_moving_the_slots():
    # Every slot is resolved in the truncated frame, so two checkpoints replaying
    # the same text land on the same tokens whatever the budget. A slot that drifted
    # with max_len would make the whole checkpoint comparison meaningless.
    plan = slot_plan(first_k=3, max_boundaries=2)
    long_prompt = "X" * 40 + PROMPT
    resolved = {}
    for max_len in (128, 32, len(RESPONSE) + 1):
        replay = build_replay(
            CharacterTokenizer(), record(prompt=long_prompt), plan, max_len=max_len, first_k=3, max_boundaries=2
        )
        kept = replay.text[len(replay.text) - int(replay.input_ids.shape[1]) :]
        resolved[max_len] = [kept[position] for position in replay.slot_positions]
        assert kept[replay.content_start : replay.content_stop] == RESPONSE
    assert list(resolved.values()) == [[">", "H", "i", " ", ".", "?", "?"]] * 3


def test_a_token_straddling_the_prompt_boundary_is_reported():
    plan = slot_plan(first_k=1, max_boundaries=0)
    replay = build_replay(PairTokenizer(), record(), plan, max_len=64, first_k=1, max_boundaries=0)
    assert replay.merged_prompt_boundary


def test_chat_template_records_replay_the_generation_prefix():
    plan = slot_plan(first_k=1, max_boundaries=0)
    messages = [{"role": "user", "content": "Q"}]
    replay = build_replay(
        CharacterTokenizer(),
        {"record_id": "m1", "messages": messages, "response": "A."},
        plan,
        max_len=64,
        first_k=1,
        max_boundaries=0,
    )
    assert replay.text == "user:Q|assistant:A."
    assert replay.text[replay.content_start : replay.content_stop] == "A."


def test_surface_and_token_histogram_features_are_deterministic():
    values = surface_features("Two words. Right?", 5)
    assert values[0] == 5.0 and values[1] == 3.0
    assert values[3] == 2.0 and values[4] == 1.0
    histogram = token_histogram(np.array([1, 1, 514]), bins=512)
    assert histogram.sum() == pytest.approx(1.0)
    assert histogram[1] == pytest.approx(2 / 3)
    assert histogram[2] == pytest.approx(1 / 3)


# --------------------------------------------------------------------------
# toy mixture of experts
# --------------------------------------------------------------------------


class ScaleExpert(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = float(scale)

    def forward(self, value):
        return value * self.scale


class ToyMoeBlock(nn.Module):
    def __init__(self, hidden=6, experts=4, top_k=2, norm_topk_prob=False, scales=None, seed=0):
        super().__init__()
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        self.num_experts = experts
        generator = torch.Generator().manual_seed(seed)
        self.gate = nn.Linear(hidden, experts, bias=False)
        with torch.no_grad():
            # A gentle router keeps the top-k mass away from a saturated 1.0, so a
            # renormalization bug cannot hide inside floating-point rounding.
            self.gate.weight.copy_(torch.randn(experts, hidden, generator=generator) * 0.35)
        scales = scales if scales is not None else [1.0 + index for index in range(experts)]
        self.experts = nn.ModuleList([ScaleExpert(scale) for scale in scales])

    def forward(self, hidden_states):
        routing = route(self, hidden_states)
        value = selected_expert_contributions(self, hidden_states, routing).sum(dim=1)
        return value.reshape_as(hidden_states)


class ToyLayer(nn.Module):
    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp

    def forward(self, hidden_states, **_kwargs):
        return hidden_states + self.mlp(hidden_states)


class ToyBase(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)


class ToyModel(nn.Module):
    def __init__(self, *, hidden=6, layers=2, vocab=128, norm_topk_prob=False, seed=0):
        super().__init__()
        self.model = ToyBase(
            [
                ToyLayer(ToyMoeBlock(hidden=hidden, norm_topk_prob=norm_topk_prob, seed=seed + index))
                for index in range(layers)
            ]
        )
        generator = torch.Generator().manual_seed(seed)
        self.embedding = nn.Embedding(vocab, hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.randn(vocab, hidden, generator=generator) * 0.7)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        hidden = self.embedding(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden)
        return SimpleNamespace(logits=self.lm_head(hidden))


def trace_toy(norm_topk_prob=False, positions=(1, 3, 5)):
    model = ToyModel(norm_topk_prob=norm_topk_prob)
    input_ids = torch.arange(8).unsqueeze(0)
    with TraceCollector(model, [0, 1], list(positions)) as collector:
        model(input_ids=input_ids)
    return collector, collect_record(collector, [0, 1], ("mlp_in", "mlp_out"), store_expert_outputs=True)


def test_trace_records_true_logits_probabilities_and_expert_paths():
    collector, captured = trace_toy()
    assert captured["mlp_in"].shape == (2, 3, 6)
    assert captured["router_logits"].shape == (2, 3, 4)
    assert captured["topk_indices"].shape == (2, 3, 2)
    probabilities = captured["router_probs"].astype(np.float64)
    assert np.allclose(probabilities.sum(axis=-1), 1.0, atol=1e-3)
    recomputed = torch.softmax(torch.tensor(captured["router_logits"].astype(np.float32)), dim=-1).numpy()
    assert np.allclose(recomputed, probabilities, atol=1e-3)
    assert captured["expert_outputs"].shape == (2, 3, 2, 6)
    reconstructed = torch.tensor(captured["expert_outputs"].astype(np.float32)).sum(dim=2)
    assert np.allclose(reconstructed.numpy(), captured["mlp_out"].astype(np.float32), atol=2e-2)
    assert set(collector.routing) == {0, 1}


def test_native_unnormalized_topk_weights_are_stored_unrenormalized():
    _, native = trace_toy(norm_topk_prob=False)
    _, normalized = trace_toy(norm_topk_prob=True)
    native_mass = native["topk_weights"].astype(np.float64).sum(axis=-1)
    normalized_mass = normalized["topk_weights"].astype(np.float64).sum(axis=-1)
    assert np.all(native_mass < 0.99)
    assert np.allclose(normalized_mass, 1.0, atol=1e-3)
    assert np.all(normalized_mass > native_mass)
    assert np.array_equal(native["topk_indices"], normalized["topk_indices"])
    # The native weights are exactly the softmax mass on the selected experts.
    gathered = np.take_along_axis(
        native["router_probs"].astype(np.float64), native["topk_indices"].astype(np.int64), axis=-1
    ).sum(axis=-1)
    assert np.allclose(native_mass, gathered, atol=1e-3)


def test_router_fidelity_holds_under_both_normalization_conventions():
    for normalize in (False, True):
        _, captured = trace_toy(norm_topk_prob=normalize)
        error, agreement = router_fidelity(captured, norm_topk_prob=normalize)
        assert error < 1e-2
        assert agreement == 1.0
    # Read under the wrong convention the same numbers fail: normalized weights sum
    # to one while the raw softmax mass on two of four experts does not.
    _, normalized = trace_toy(norm_topk_prob=True)
    assert router_fidelity(normalized, norm_topk_prob=False)[0] > ROUTER_FIDELITY_TOLERANCE


def biased_router_model(seed=0):
    """A router with a bias, which recomputing from the weight matrix alone misses."""
    model = ToyModel(seed=seed)
    for layer in decoder_layers(model):
        gate = layer.mlp.gate
        biased = nn.Linear(gate.in_features, gate.out_features, bias=True)
        with torch.no_grad():
            biased.weight.copy_(gate.weight)
            biased.bias.copy_(torch.tensor([3.0, -2.0, 1.0, 0.0]))
        layer.mlp.gate = biased
    return model


def dense_model(hidden=6, layers=2, seed=0):
    """A toy model whose blocks are plain MLPs, which is the dense OLMo-2 shape."""
    model = ToyModel(hidden=hidden, layers=layers, seed=seed)
    for layer in decoder_layers(model):
        layer.mlp = nn.Linear(hidden, hidden)
    return model


def corpus(n_records=6):
    return [
        {
            "record_id": f"r{index}",
            "prompt": f"Student {index} asks>",
            "response": "What do you notice? Try the first step." if index % 2 else "The answer is four.",
            "item_id": f"i{index // 2}",
            "sibling_id": f"i{index // 2}",
            "label": index % 2,
            "quality": 0.5,
        }
        for index in range(n_records)
    ]


def trace_corpus(tmp_path, rows, *, model=None, plan=None, store_expert_outputs=True):
    plan = plan or slot_plan(first_k=2, max_boundaries=2)
    model = model or ToyModel()
    replays = [build_replay(CharacterTokenizer(), row, plan, max_len=128, first_k=2, max_boundaries=2) for row in rows]
    result = trace_records(
        model,
        rows,
        replays,
        plan=plan,
        layer_indices=[0, 1],
        sites=("mlp_in", "residual_out"),
        store_expert_outputs=store_expert_outputs,
        progress_every=0,
    )
    path = tmp_path / "trace.npz"
    write_trace(path, result)
    return result, load_atlas_trace(path)


def test_a_trace_round_trips_through_npz_with_every_slot_intact(tmp_path):
    rows = corpus()
    _, trace = trace_corpus(tmp_path, rows)
    assert trace.n_records == len(rows)
    assert trace.roles == ["prompt_last", "generated_0", "generated_1", "boundary_0", "boundary_1", "content_last"]
    assert trace.sites == ["residual_out", "mlp_in"] or set(trace.sites) == {"mlp_in", "residual_out"}
    assert trace.has_routing
    assert trace.features("mlp_in", 1, trace.slot("prompt_last")).shape == (len(rows), 6)
    assert np.array_equal(trace["labels"], np.array([0, 1, 0, 1, 0, 1], dtype=np.int32))
    # "The answer is four." has one boundary, the question has two.
    assert trace["position_mask"][:, trace.slot("boundary_1")].tolist() == [False, True] * 3
    assert len(set(trace["text_sha256"].tolist())) == len(rows)
    assert trace["text_sha256"][0] == sha256_text(rows[0]["prompt"] + rows[0]["response"])
    assert trace.metadata["norm_topk_prob"] is False
    assert trace.metadata["parameters"]["value_sha256"]
    assert np.isfinite(trace["response_logprob"]).all()


def test_a_dense_model_traces_every_slot_and_claims_nothing_about_routing(tmp_path):
    rows = corpus()
    result, trace = trace_corpus(tmp_path, rows, model=dense_model())
    assert not trace.has_routing
    assert result.router_weights == {}
    assert trace.features("mlp_in", 1, trace.slot("content_last")).shape == (len(rows), 6)
    assert trace["position_mask"][:, trace.slot("boundary_1")].tolist() == [False, True] * 3
    assert np.isfinite(trace["response_logprob"]).all()
    # There are no top-k weights in a dense trace, so the artifact makes no claim
    # about them rather than an unfalsifiable true one.
    assert trace.metadata["mixture_of_experts"] is False
    assert trace.metadata["topk_weights_are_native"] is None
    assert trace.metadata["norm_topk_prob"] is None
    assert trace.metadata["router_fidelity"] is None


def test_a_router_the_tracer_cannot_recompute_is_refused_rather_than_written(tmp_path):
    # The stored logits come from the router weight while the stored top-k weights
    # come from the gate itself. A bias makes those two different routers, and a
    # trace holding both would contradict itself.
    with pytest.raises(ValueError, match="two different routers"):
        trace_corpus(tmp_path, corpus(), model=biased_router_model())


def test_a_plain_softmax_router_records_the_fidelity_it_was_checked_against(tmp_path):
    _, trace = trace_corpus(tmp_path, corpus())
    fidelity = trace.metadata["router_fidelity"]
    assert fidelity["max_native_weight_error"] < 1e-2
    assert fidelity["mean_selection_agreement"] == 1.0
    assert fidelity["tolerance"] == ROUTER_FIDELITY_TOLERANCE


def test_the_router_weight_sidecar_reproduces_the_traced_probabilities(tmp_path):
    rows = corpus()
    result, trace = trace_corpus(tmp_path, rows)
    write_router_weights(tmp_path / "router.npz", result, model="toy", adapter=None)
    weights = delta.load_router_weights(tmp_path / "router.npz")
    assert sorted(weights) == [0, 1]
    report = delta.decompose_routing_change(trace, trace, weights, weights)
    assert report["total_js_bits"] == pytest.approx(0.0, abs=1e-9)
    assert report["router_frozen_everywhere"] is True
    assert all(row["reconstruction_max_error"] < 1e-2 for row in report["per_layer"])


def test_two_checkpoints_over_the_same_corpus_are_comparable(tmp_path):
    rows = corpus()
    base_result, base = trace_corpus(tmp_path / "base", rows, model=ToyModel(seed=0))
    tuned_result, tuned = trace_corpus(tmp_path / "tuned", rows, model=ToyModel(seed=4))
    delta.check_comparable(base, tuned)
    assert base_result.metadata["parameters"]["value_sha256"] != tuned_result.metadata["parameters"]["value_sha256"]
    report = delta.routing_delta(base, tuned)
    assert report["router_js_bits"] > 0.0
    assert report["tokens"] == int(base["position_mask"].sum())
    assert delta.expert_output_change(base, tuned)


def test_a_half_sparse_layer_stack_is_refused_rather_than_misaligned():
    model = ToyModel()
    model.model.layers[1].mlp = nn.Linear(6, 6)
    input_ids = torch.arange(8).unsqueeze(0)
    with TraceCollector(model, [0, 1], [1, 3]) as collector:
        model(input_ids=input_ids)
    with pytest.raises(ValueError, match="separately"):
        collect_record(collector, [0, 1], ("mlp_in",), store_expert_outputs=False)


def test_expand_to_slots_leaves_missing_slots_at_zero():
    value = np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3)
    expanded = expand_to_slots(value, rows=[0, 1], slots=[0, 3], n_slots=4)
    assert expanded.shape == (2, 4, 3)
    assert np.array_equal(expanded[:, 0], value[:, 0])
    assert np.array_equal(expanded[:, 3], value[:, 1])
    assert not expanded[:, 1].any() and not expanded[:, 2].any()


def test_expert_paths_ignore_selection_order_and_skip_masked_slots():
    indices = np.array([[[1, 2], [2, 1], [0, 3]], [[0, 1], [0, 1], [3, 3]]])
    valid = np.array([True, True, False])
    set_ids, path, top1 = routing_paths(indices, valid)
    assert set_ids[0, 0] == set_ids[0, 1]
    assert set_ids[0, 0] != set_ids[0, 2] == 0
    assert path[0] == path[1] and path[2] == 0
    assert top1[0, 2] == -1 and top1[0, 0] == 1


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


def test_js_divergence_spans_zero_to_one_bit():
    uniform = np.full(4, 0.25)
    assert delta.js_divergence(uniform, uniform) == pytest.approx(0.0, abs=1e-12)
    first = np.array([1.0, 0.0, 0.0, 0.0])
    second = np.array([0.0, 1.0, 0.0, 0.0])
    assert delta.js_divergence(first, second) == pytest.approx(1.0, abs=1e-6)


def test_entropy_is_reported_in_bits():
    assert delta.entropy_bits(np.full(4, 0.25)) == pytest.approx(2.0)
    assert delta.entropy_bits(np.array([1.0, 0.0, 0.0, 0.0])) == pytest.approx(0.0, abs=1e-6)


def test_topk_overlap_counts_shared_experts():
    first = np.array([[0, 1, 2, 3]])
    assert delta.topk_overlap(first, np.array([[0, 1, 2, 3]]))[0] == pytest.approx(1.0)
    assert delta.topk_overlap(first, np.array([[0, 1, 7, 8]]))[0] == pytest.approx(0.5)
    assert delta.topk_overlap(first, np.array([[4, 5, 6, 7]]))[0] == pytest.approx(0.0)


def test_rank_swaps_reach_one_when_the_order_reverses():
    indices = np.array([[[0, 1, 2]]])
    base_logits = np.array([[[3.0, 2.0, 1.0]]])
    assert delta.rank_swaps(base_logits, base_logits, indices)[0, 0] == pytest.approx(0.0)
    assert delta.rank_swaps(base_logits, -base_logits, indices)[0, 0] == pytest.approx(1.0)
    one_swap = np.array([[[2.0, 3.0, 1.0]]])
    assert delta.rank_swaps(base_logits, one_swap, indices)[0, 0] == pytest.approx(1 / 3)


def test_usage_histogram_normalizes_over_selections():
    usage = delta.usage_histogram(np.array([[0, 1], [0, 2]]), experts=4)
    assert usage.tolist() == [0.5, 0.25, 0.25, 0.0]


def test_linear_cka_is_scale_and_rotation_invariant():
    rng = np.random.default_rng(0)
    values = rng.normal(size=(40, 6))
    rotation = np.linalg.qr(rng.normal(size=(6, 6)))[0]
    assert delta.linear_cka(values, values) == pytest.approx(1.0)
    assert delta.linear_cka(values, values * 7.0) == pytest.approx(1.0)
    assert delta.linear_cka(values, values @ rotation) == pytest.approx(1.0)
    assert delta.linear_cka(values, rng.normal(size=(40, 6))) < 0.5


def test_cosine_and_relative_l2_agree_with_hand_values():
    first = np.array([[1.0, 0.0], [0.0, 2.0]])
    assert delta.mean_cosine(first, first) == pytest.approx(1.0)
    assert delta.mean_cosine(first, -first) == pytest.approx(-1.0)
    assert delta.relative_l2(first, first) == pytest.approx(0.0)
    assert delta.relative_l2(first, first * 2) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# synthetic traces
# --------------------------------------------------------------------------


def build_trace(
    *,
    n_groups=14,
    hidden=8,
    layers=(4, 8),
    roles=("prompt_last", "content_last"),
    experts=6,
    top_k=2,
    signal=2.5,
    lexical_signal=0.0,
    routing=True,
    router_weight=None,
    hidden_shift=0.0,
    labels=None,
    seed=0,
) -> AtlasTrace:
    """A trace with a planted, controllable structure.

    ``signal`` moves the activations with the label, ``lexical_signal`` moves
    the token histogram with it, and ``hidden_shift`` displaces the router input
    the way an upstream adaptation would.
    """
    rng = np.random.default_rng(seed)
    n_records = 2 * n_groups
    label_values = np.tile([0, 1], n_groups) if labels is None else np.asarray(labels)
    siblings = np.repeat([f"s{index}" for index in range(n_groups)], 2)
    direction = rng.normal(size=hidden)
    direction /= np.linalg.norm(direction)

    arrays: dict[str, np.ndarray] = {
        "ids": np.asarray([f"r{index}" for index in range(n_records)]),
        "item_ids": siblings.copy(),
        "sibling_ids": siblings,
        "constructs": np.asarray(["elicitation"] * n_records),
        "labels": label_values.astype(np.int32),
        "named_labels": label_values.astype(np.int32),
        "enacted_labels": label_values.astype(np.int32),
        "quality": np.repeat(rng.normal(size=n_groups), 2).astype(np.float32),
        "layers": np.asarray(layers, dtype=np.int16),
        "position_roles": np.asarray(roles),
        "position_bases": np.asarray(roles),
        "position_ordinals": np.zeros(len(roles), dtype=np.int16),
        "position_mask": np.ones((n_records, len(roles)), dtype=bool),
        "position_index": np.tile(np.arange(len(roles), dtype=np.int32), (n_records, 1)),
        "position_token_id": rng.integers(0, 40, size=(n_records, len(roles))).astype(np.int32),
        "text_sha256": np.asarray([f"text{index}" for index in range(n_records)]),
        "token_sha256": np.asarray([f"token{index}" for index in range(n_records)]),
        "surface_features": rng.normal(size=(n_records, 4)).astype(np.float32),
        "token_histogram": rng.normal(size=(n_records, 16)).astype(np.float32),
        "response_logprob": rng.normal(size=n_records).astype(np.float32),
    }
    arrays["token_histogram"][:, 3] += lexical_signal * label_values

    shape = (n_records, len(layers), len(roles), hidden)
    states = rng.normal(size=shape) * 0.4 + hidden_shift
    states += signal * label_values[:, None, None, None] * direction
    for site in ("mlp_in", "mlp_out", "residual_in", "residual_out"):
        arrays[site] = states.astype(np.float16)

    if routing:
        weight = rng.normal(size=(experts, hidden)) if router_weight is None else router_weight
        logits = arrays["mlp_in"].astype(np.float64) @ weight.T
        probabilities = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        indices = np.argsort(-probabilities, axis=-1)[..., :top_k]
        arrays["router_logits"] = logits.astype(np.float16)
        arrays["router_probs"] = probabilities.astype(np.float16)
        arrays["topk_indices"] = indices.astype(np.int16)
        arrays["topk_weights"] = np.take_along_axis(probabilities, indices, axis=-1).astype(np.float16)
        arrays["expert_outputs"] = (states[..., None, :] * np.arange(1, top_k + 1)[:, None]).astype(np.float16)
        set_ids = np.zeros((n_records, len(layers), len(roles)), dtype=np.int64)
        path_ids = np.zeros((n_records, len(roles)), dtype=np.int64)
        for record_index in range(n_records):
            per_slot, path, _ = routing_paths(indices[record_index], np.ones(len(roles), dtype=bool))
            set_ids[record_index] = per_slot
            path_ids[record_index] = path
        arrays["expert_set_id"] = set_ids
        arrays["expert_path_id"] = path_ids

    metadata = {
        "schema": "semantic-atlas-trace/v1",
        "model": "toy",
        "layers": list(layers),
        "slots": list(roles),
        "mixture_of_experts": routing,
        "norm_topk_prob": False,
    }
    return AtlasTrace(arrays=arrays, metadata=metadata)


def routed_pair(*, hidden_shift=0.0, tuned_router=None, seed=0):
    """Two traces over the same texts, differing only as requested."""
    base_weight = np.random.default_rng(99).normal(size=(6, 8)) * 0.5
    tuned_weight = base_weight if tuned_router is None else tuned_router
    base = build_trace(router_weight=base_weight, seed=seed)
    tuned = build_trace(router_weight=tuned_weight, hidden_shift=hidden_shift, seed=seed)
    return (base, tuned, dict.fromkeys(base.layers, base_weight), dict.fromkeys(base.layers, tuned_weight))


# --------------------------------------------------------------------------
# trace comparison
# --------------------------------------------------------------------------


def test_comparison_refuses_traces_over_different_texts():
    base = build_trace()
    tuned = build_trace()
    tuned["text_sha256"][2] = "rewritten"
    with pytest.raises(ValueError, match="texts are not fixed"):
        delta.check_comparable(base, tuned)


def test_identical_checkpoints_produce_a_zero_delta():
    base = build_trace()
    report = delta.routing_delta(base, build_trace())
    assert report["router_js_bits"] == pytest.approx(0.0, abs=1e-9)
    assert report["topk_overlap"] == pytest.approx(1.0)
    assert report["expert_path_change_rate"] == pytest.approx(0.0)
    activation = delta.activation_delta(base, build_trace())
    assert all(row["linear_cka"] == pytest.approx(1.0) for row in activation)
    assert all(row["relative_l2"] == pytest.approx(0.0) for row in activation)


def test_routing_delta_reports_every_requested_metric():
    base, tuned, _, _ = routed_pair(hidden_shift=0.8)
    report = delta.routing_delta(base, tuned)
    assert report["tokens"] == base.n_records * len(base.roles)
    assert report["router_js_bits"] > 0.0
    assert 0.0 <= report["topk_overlap"] < 1.0
    assert report["expert_path_change_rate"] > 0.0
    layer = report["per_layer"][0]
    for key in (
        "router_js_bits",
        "topk_overlap",
        "rank_swaps_within_base_topk",
        "entropy_bits_delta",
        "native_topk_mass_base",
        "selected_probability_mass_tuned",
        "utilization_js_bits",
        "usage_entropy_base",
        "max_expert_load_tuned",
        "expert_set_change_rate",
    ):
        assert key in layer, key
    assert report["native_topk_weights_normalized"] is False


def test_expert_output_change_restricts_to_matched_selections():
    base, tuned, _, _ = routed_pair(hidden_shift=0.6)
    rows = delta.expert_output_change(base, tuned)
    assert rows and all(0.0 <= row["matched_fraction"] <= 1.0 for row in rows)
    matched = [row for row in rows if row["matched_fraction"] > 0]
    assert matched and all("claim_limit" in row for row in matched)


def test_expert_output_change_without_stored_outputs_says_what_to_rerun():
    base = build_trace()
    tuned = build_trace()
    del tuned.arrays["expert_outputs"]
    with pytest.raises(ValueError, match="store-expert-outputs"):
        delta.expert_output_change(base, tuned)


def test_two_mixtures_of_different_widths_are_refused_rather_than_broadcast():
    with pytest.raises(ValueError, match="expert count or top-k"):
        delta.check_comparable(build_trace(experts=6), build_trace(experts=8))


def test_a_normalization_disagreement_is_refused_because_the_masses_differ_in_scale():
    base, tuned, _, _ = routed_pair()
    tuned.metadata["norm_topk_prob"] = True
    with pytest.raises(ValueError, match="different scales"):
        delta.routing_delta(base, tuned)


def test_a_trace_missing_the_derived_expert_arrays_asks_to_be_re_traced():
    base = build_trace()
    tuned = build_trace()
    del tuned.arrays["expert_path_id"]
    with pytest.raises(ValueError, match="re-trace"):
        delta.routing_delta(base, tuned)


# --------------------------------------------------------------------------
# routing decomposition
# --------------------------------------------------------------------------


def test_a_frozen_router_still_reroutes_when_the_hidden_state_moves():
    base, tuned, base_router, tuned_router = routed_pair(hidden_shift=0.9)
    report = delta.decompose_routing_change(base, tuned, base_router, tuned_router)
    assert report["router_frozen_everywhere"] is True
    assert report["router_parameter_shift_js_bits"] == pytest.approx(0.0, abs=1e-12)
    assert report["router_input_shift_js_bits"] > 0.0
    assert report["total_js_bits"] > 0.0
    assert all(row["routing_changed_with_frozen_router"] for row in report["per_layer"])
    assert "do not imply identical routing" in report["note"]


def test_a_router_parameter_change_shows_up_in_its_own_channel():
    base_weight = np.random.default_rng(99).normal(size=(6, 8)) * 0.5
    nudged = base_weight + np.random.default_rng(7).normal(size=(6, 8)) * 0.3
    base, tuned, base_router, tuned_router = routed_pair(hidden_shift=0.0, tuned_router=nudged)
    report = delta.decompose_routing_change(base, tuned, base_router, tuned_router)
    assert report["router_frozen_everywhere"] is False
    assert report["router_parameter_shift_js_bits"] > 0.0
    assert report["router_input_shift_js_bits"] == pytest.approx(0.0, abs=1e-12)
    assert all(not row["routing_changed_with_frozen_router"] for row in report["per_layer"])


def test_a_mismatched_router_sidecar_is_rejected_rather_than_averaged():
    base, tuned, base_router, _ = routed_pair(hidden_shift=0.4)
    noise = np.random.default_rng(5).normal(size=(6, 8))
    wrong = {index: value + noise for index, value in base_router.items()}
    with pytest.raises(ValueError, match="does not match this trace"):
        delta.decompose_routing_change(base, tuned, wrong, wrong)


def test_a_constant_offset_on_the_router_is_correctly_seen_as_no_change():
    # Adding the same constant to every router weight shifts all logits by the
    # same amount, so the softmax is untouched. A metric that flagged this would
    # be reporting parameter arithmetic rather than routing.
    base, tuned, base_router, _ = routed_pair(hidden_shift=0.0)
    offset = {index: value + 3.0 for index, value in base_router.items()}
    report = delta.decompose_routing_change(base, tuned, base_router, offset)
    assert report["router_frozen_everywhere"] is False
    assert report["router_parameter_shift_js_bits"] == pytest.approx(0.0, abs=1e-9)


def test_expert_adaptation_holds_selection_fixed():
    hidden = torch.randn(12, 6, generator=torch.Generator().manual_seed(3))
    base_block = ToyMoeBlock(seed=1)
    same = ToyMoeBlock(seed=1)
    identical = delta.expert_adaptation_from_blocks(base_block, same, hidden)
    assert identical["mean_cosine"] == pytest.approx(1.0, abs=1e-5)
    assert identical["linear_cka"] == pytest.approx(1.0, abs=1e-5)
    assert identical["selection_held_fixed"] is True

    adapted = ToyMoeBlock(seed=1, scales=[3.0, -1.0, 0.5, 2.0])
    changed = delta.expert_adaptation_from_blocks(base_block, adapted, hidden)
    assert changed["relative_l2"] > 0.0
    assert changed["tokens"] == 12
    assert changed["norm_topk_prob"] is False


def test_the_block_decomposition_separates_experts_hidden_state_and_selection():
    generator = torch.Generator().manual_seed(3)
    base_hidden = torch.randn(24, 6, generator=generator)
    base_block = ToyMoeBlock(seed=1)
    adapted = ToyMoeBlock(seed=1, scales=[3.0, -1.0, 0.5, 2.0])

    unchanged = delta.decompose_block_output_change(base_block, base_block, base_hidden, base_hidden)
    for channel in ("expert_adaptation", "hidden_state_shift", "selection_shift", "observed_total"):
        assert unchanged[channel]["relative_l2"] == pytest.approx(0.0, abs=1e-6)
    assert unchanged["selection_topk_overlap"] == pytest.approx(1.0)

    experts_only = delta.decompose_block_output_change(base_block, adapted, base_hidden, base_hidden)
    assert experts_only["expert_adaptation"]["relative_l2"] > 0.0
    assert experts_only["hidden_state_shift"]["relative_l2"] == pytest.approx(0.0, abs=1e-6)
    assert experts_only["selection_shift"]["relative_l2"] == pytest.approx(0.0, abs=1e-6)
    assert experts_only["channels_are_additive"] is False


def test_a_frozen_router_still_shows_a_selection_shift_when_the_input_moves():
    # The tuned block shares the base block's router weights exactly, so no
    # router parameter changed. The hidden state entering it did, which is
    # enough to reroute tokens, and the report has to say so.
    generator = torch.Generator().manual_seed(11)
    base_hidden = torch.randn(64, 6, generator=generator)
    tuned_hidden = base_hidden + torch.randn(64, 6, generator=generator) * 1.5
    base_block = ToyMoeBlock(seed=1)
    frozen_router_tuned = ToyMoeBlock(seed=1, scales=[2.0, 0.5, 1.5, -0.5])

    report = delta.decompose_block_output_change(base_block, frozen_router_tuned, base_hidden, tuned_hidden)
    assert report["router_weights_identical"] is True
    assert report["selection_topk_overlap"] < 1.0
    assert report["selection_changed_with_frozen_router"] is True
    assert report["selection_shift"]["relative_l2"] > 0.0
    assert report["hidden_state_shift"]["relative_l2"] > 0.0
    assert "do not imply identical routing" in report["note"]


# --------------------------------------------------------------------------
# checkpoint description
# --------------------------------------------------------------------------


def write_adapter(directory: Path, **overrides) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    body = {
        "peft_type": "LORA",
        "base_model_name_or_path": "allenai/OLMo-2-1124-7B-Instruct",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "r": 32,
        **overrides,
    }
    (directory / "adapter_config.json").write_text(json.dumps(body))
    (directory / "adapter_model.safetensors").write_bytes(b"")
    return directory


def test_the_local_dense_tutor_adapter_is_recognized_as_dense_olmo2(tmp_path):
    description = delta.describe_checkpoint(write_adapter(tmp_path / "pedagogy-tutor-armF"))
    assert description.kind == "peft_adapter"
    assert description.loadable
    assert description.base_model == "allenai/OLMo-2-1124-7B-Instruct"
    assert description.architecture == "dense"
    assert not description.adapts_experts
    assert not description.adapts_router
    description.require_loadable()


def test_gate_proj_is_not_mistaken_for_the_router(tmp_path):
    description = delta.describe_checkpoint(write_adapter(tmp_path / "attention_and_mlp"))
    assert "gate_proj" in description.target_modules
    assert not description.adapts_router


def test_an_adapter_on_mlp_gate_is_reported_as_adapting_the_router(tmp_path):
    description = delta.describe_checkpoint(
        write_adapter(tmp_path / "trainable_router", target_modules=["mlp.gate", "q_proj"])
    )
    assert description.adapts_router


def test_per_expert_lora_is_reported_as_adapting_experts(tmp_path):
    description = delta.describe_checkpoint(
        write_adapter(
            tmp_path / "olmoe_expert_lora",
            base_model_name_or_path="allenai/OLMoE-1B-7B-0924-Instruct",
            target_parameters=["mlp.experts.gate_up_proj", "mlp.experts.down_proj"],
        )
    )
    assert description.architecture == "mixture_of_experts"
    assert description.adapts_experts
    assert not description.adapts_router


def test_deepspeed_zero_states_are_described_and_refused(tmp_path):
    step = tmp_path / "pedagogy_olmoe" / "global_step200"
    step.mkdir(parents=True)
    (step / "mp_rank_00_model_states.pt").write_bytes(b"")
    (step / "zero_pp_rank_0_mp_rank_00_optim_states.pt").write_bytes(b"")

    description = delta.describe_checkpoint(step)
    assert description.kind == "deepspeed_zero"
    assert not description.loadable
    assert "zero_to_fp32" in description.reason
    assert "adapter_config.json" in description.reason
    assert description.architecture == "mixture_of_experts"
    with pytest.raises(delta.AdapterLoadingUnsupported, match="zero_to_fp32"):
        description.require_loadable()
    with pytest.raises(delta.AdapterLoadingUnsupported):
        delta.apply_adapter(object(), step)


def test_a_consolidated_export_inside_a_global_step_directory_is_read_as_a_model(tmp_path):
    # This is the layout compare_rl_checkpoints.sbatch asks the operator to create.
    # Classifying it by its name would refuse the very export that fixes the refusal.
    step = tmp_path / "pedagogy_olmoe_exported" / "global_step200"
    step.mkdir(parents=True)
    (step / "config.json").write_text(json.dumps({"model_type": "olmoe"}))
    (step / "model.safetensors").write_bytes(b"")
    description = delta.describe_checkpoint(step)
    assert description.kind == "hf_model"
    assert description.loadable
    assert description.architecture == "mixture_of_experts"
    description.require_loadable()


def test_a_config_without_weights_beside_it_is_refused(tmp_path):
    directory = tmp_path / "half_written_export"
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps({"model_type": "olmo2"}))
    description = delta.describe_checkpoint(directory)
    assert description.kind == "hf_model"
    assert not description.loadable
    assert "no weight file" in description.reason


def test_a_step_directory_with_no_model_shards_is_still_a_zero_state(tmp_path):
    step = tmp_path / "pedagogy_olmoe" / "global_step190"
    step.mkdir(parents=True)
    (step / "zero_pp_rank_0_mp_rank_00_optim_states.pt").write_bytes(b"")
    description = delta.describe_checkpoint(step)
    assert description.kind == "deepspeed_zero"
    assert "may be partial" in description.reason
    empty = tmp_path / "pedagogy_olmoe" / "global_step191"
    empty.mkdir()
    assert delta.describe_checkpoint(empty).kind == "deepspeed_zero"


def test_a_zipped_whole_model_says_to_extract_it_rather_than_claiming_to_load(tmp_path):
    archive = tmp_path / "merged.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("merged/config.json", json.dumps({"model_type": "olmoe"}))
        handle.writestr("merged/model.safetensors", b"")
    description = delta.describe_checkpoint(archive)
    assert description.kind == "hf_model_zip"
    assert not description.loadable
    assert "extract" in description.reason
    with pytest.raises(delta.AdapterLoadingUnsupported):
        delta.apply_adapter(object(), archive)


def test_describe_can_be_asked_to_exit_non_zero_on_an_unloadable_checkpoint(tmp_path, monkeypatch, capsys):
    step = tmp_path / "pedagogy_olmoe" / "global_step200"
    step.mkdir(parents=True)
    (step / "mp_rank_00_model_states.pt").write_bytes(b"")
    monkeypatch.setattr(sys, "argv", ["checkpoint_delta", "--describe", str(step)])
    delta.main()  # classifying a ZeRO state is a successful classification
    assert "deepspeed_zero" in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", ["checkpoint_delta", "--describe", str(step), "--require-loadable"])
    with pytest.raises(SystemExit, match="cannot be loaded"):
        delta.main()


def test_a_missing_checkpoint_is_reported_not_guessed(tmp_path):
    description = delta.describe_checkpoint(tmp_path / "absent")
    assert description.kind == "missing" and not description.loadable


def test_non_lora_peft_types_are_refused(tmp_path):
    description = delta.describe_checkpoint(write_adapter(tmp_path / "prompt_tuning", peft_type="PROMPT_TUNING"))
    assert not description.loadable
    assert "PROMPT_TUNING" in description.reason


# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------


def test_grouped_folds_never_place_a_group_on_both_sides():
    groups = np.repeat([f"g{index}" for index in range(10)], 3)
    labels = np.tile([0, 1, 1], 10)
    folds = analysis.grouped_folds(groups, labels, n_splits=5, seed=0)
    assert len(folds) == 5
    analysis.assert_group_disjoint(folds, groups)
    covered = np.concatenate([test for _, test in folds])
    assert sorted(covered.tolist()) == list(range(len(groups)))


def test_group_leakage_is_an_error_not_a_warning():
    groups = np.array(["a", "a", "b", "b"])
    with pytest.raises(AssertionError, match="leaks groups"):
        analysis.assert_group_disjoint([(np.array([0, 2]), np.array([1, 3]))], groups)


def test_sibling_margins_pair_inside_a_group_only():
    scores = np.array([0.2, 1.0, 0.5, 0.1, 4.0])
    labels = np.array([0, 1, 0, 1, 1])
    siblings = np.array(["a", "a", "b", "b", "c"])
    margins = analysis.sibling_margins(scores, labels, siblings)
    assert margins == {"a": pytest.approx(0.8), "b": pytest.approx(-0.4)}


def test_cluster_bootstrap_and_sign_flip_agree_on_a_clear_effect():
    margins = {f"s{index}": 1.0 + 0.01 * index for index in range(30)}
    estimate, low, high = analysis.cluster_bootstrap_ci(margins, samples=500, seed=0)
    assert low > 0 and estimate == pytest.approx(1.145, abs=0.01) and high > estimate
    assert analysis.sign_flip_test(margins, samples=500, seed=0) < 0.01
    assert analysis.sign_flip_test({f"s{index}": (-1.0) ** index for index in range(30)}, samples=500) > 0.5


def test_a_planted_activation_signal_beats_every_control():
    trace = build_trace(n_groups=16, signal=3.0, lexical_signal=0.0, seed=1)
    report = analysis.quality_matched_sibling_discrimination(
        trace, site="mlp_in", layer_index=8, slot=0, n_splits=4, bootstrap_samples=200
    )
    assert report["activation"]["forced_choice_accuracy"] > 0.9
    assert report["activation"]["margin_ci95"][0] > 0
    assert report["selectivity_over_best_control"] > 0.3
    assert report["slot_coverage"] == 1.0
    assert {control["name"] for control in report["controls"]} == {
        "surface",
        "global_quality",
        "token_histogram",
        "token_identity",
    }


def test_a_position_shared_by_both_siblings_is_flagged_not_read_as_a_null():
    # Quality-matched siblings share a prompt, so the final prompt token is
    # byte-identical for both. The probe scores zero for a reason that has
    # nothing to do with the construct, and the report has to say which.
    trace = build_trace(n_groups=12, signal=3.0, seed=13)
    shared = trace["mlp_in"].copy()
    shared[1::2, :, 0] = shared[0::2, :, 0]
    trace.arrays["mlp_in"] = shared
    report = analysis.quality_matched_sibling_discrimination(
        trace, site="mlp_in", layer_index=8, slot=0, n_splits=3, bootstrap_samples=200
    )
    assert report["within_sibling_variation"] == 0.0
    assert report["position_shared_by_siblings"] is True
    assert report["activation"]["forced_choice_accuracy"] == 0.0
    assert "has not read either response" in report["claim_limit"]

    unshared = analysis.quality_matched_sibling_discrimination(
        trace, site="mlp_in", layer_index=8, slot=1, n_splits=3, bootstrap_samples=200
    )
    assert unshared["within_sibling_variation"] == 1.0
    assert unshared["position_shared_by_siblings"] is False


def test_a_lexically_carried_signal_is_absorbed_by_the_token_histogram_control():
    trace = build_trace(n_groups=16, signal=3.0, lexical_signal=6.0, seed=2)
    report = analysis.quality_matched_sibling_discrimination(
        trace, site="mlp_in", layer_index=8, slot=0, n_splits=4, bootstrap_samples=200
    )
    assert report["best_control"] == "token_histogram"
    assert report["selectivity_over_best_control"] <= 0.0


def test_the_lexical_control_accepts_response_text():
    trace = build_trace(n_groups=12, signal=3.0, seed=3)
    texts = ["ask a question" if label else "state the answer" for label in trace["labels"]]
    report = analysis.quality_matched_sibling_discrimination(
        trace, site="mlp_in", layer_index=8, slot=0, texts=texts, n_splits=3, bootstrap_samples=200
    )
    lexical = next(control for control in report["controls"] if control["name"] == "lexical")
    assert lexical["forced_choice_accuracy"] > 0.9
    assert report["selectivity_over_best_control"] <= 0.0


def test_quality_residualization_runs_and_is_reported_separately():
    trace = build_trace(n_groups=14, signal=3.0, seed=4)
    report = analysis.quality_matched_sibling_discrimination(
        trace, site="mlp_in", layer_index=8, slot=0, n_splits=4, bootstrap_samples=200
    )
    assert report["activation_quality_residualized"] is not None
    assert report["activation_quality_residualized"]["forced_choice_accuracy"] > 0.8


def test_a_slot_missing_from_most_records_is_not_scored_on_zeros():
    trace = build_trace(n_groups=14, roles=("prompt_last", "boundary_0"), seed=5)
    trace["position_mask"][:, 1] = False
    with pytest.raises(ValueError, match="covers only 0 labelled records"):
        analysis.analysis_view(trace, 1)


def test_the_probe_follows_what_a_response_enacts_not_what_it_names():
    trace = build_trace(n_groups=20, signal=3.0, seed=6)
    named = trace["named_labels"].copy()
    named[:8] = 1 - named[:8]  # the first four sibling groups claim the opposite strategy
    trace.arrays["named_labels"] = named
    report = analysis.named_versus_enacted(trace, site="mlp_in", layer_index=8, slot=0, n_splits=3)
    assert report["n_mismatch"] == 8
    assert report["n_congruent"] == 32
    assert report["congruent_trained"]["mismatch_agreement_with_enacted"] > 0.8
    assert report["congruent_trained"]["enactment_preference"] > 0.0
    assert report["decodability"]["enacted"] > report["decodability"]["named"]
    assert report["reads_enactment_not_naming"] is True
    # Whole sibling groups were flipped, so no mismatch row has a trained twin.
    assert report["mismatch_rows_sharing_a_group_with_training"] == 0


def test_a_mismatch_row_whose_sibling_trained_the_probe_is_reported_as_such():
    trace = build_trace(n_groups=20, signal=3.0, seed=6)
    named = trace["named_labels"].copy()
    flipped = np.array([0, 2, 4, 6])  # one member of each of the first four sibling groups
    named[flipped] = 1 - named[flipped]
    trace.arrays["named_labels"] = named
    report = analysis.named_versus_enacted(trace, site="mlp_in", layer_index=8, slot=0, n_splits=3)
    assert report["n_mismatch"] == 4
    assert report["mismatch_rows_sharing_a_group_with_training"] == 4
    assert "seen a sibling of those rows" in report["claim_limit"]


def test_named_versus_enacted_refuses_when_there_is_no_mismatch():
    trace = build_trace(n_groups=8, seed=7)
    with pytest.raises(ValueError, match="no named/enacted mismatch"):
        analysis.named_versus_enacted(trace, site="mlp_in", layer_index=8, slot=0)


def test_named_versus_enacted_refuses_a_congruent_set_carrying_one_label():
    # Every record enacting strategy 1 names the other one, which leaves the probe
    # nothing but strategy-0 records to learn from.
    trace = build_trace(n_groups=8, seed=7)
    enacted = trace["enacted_labels"]
    named = enacted.copy()
    named[enacted == 1] = 0
    trace.arrays["named_labels"] = named
    with pytest.raises(ValueError, match="same enacted label"):
        analysis.named_versus_enacted(trace, site="mlp_in", layer_index=8, slot=0)


def test_routing_features_expose_entropy_native_mass_and_paths():
    trace = build_trace(n_groups=10, seed=8)
    entropy = analysis.router_entropy(trace, slot=0)
    assert entropy.shape == (20, 2) and np.all(entropy > 0)
    native = analysis.native_topk_mass(trace, slot=0)
    assert np.all(native < 1.0)
    assert np.allclose(native, analysis.selected_probability_mass(trace, slot=0), atol=1e-3)
    multihot = analysis.selection_multihot(trace, slot=0)
    assert multihot.shape == (20, 2 * 6)
    assert np.all(multihot.reshape(20, 2, 6).sum(axis=-1) == 2)
    usage = analysis.expert_usage(trace, [0])
    assert usage.shape == (2, 6) and np.allclose(usage.sum(axis=1), 1.0)


def test_routing_discrimination_reports_its_token_controls():
    trace = build_trace(n_groups=14, signal=3.0, seed=9)
    report = analysis.routing_discrimination(trace, slot=0, n_splits=4, bootstrap_samples=200)
    assert {item["name"] for item in report["routing"]} == {"router_probs", "selection_multihot", "routing_summary"}
    assert "token_identity" in {item["name"] for item in report["controls"]}
    assert "tokenization result" in report["claim_limit"]
    assert report["path_association"]["distinct_paths"] >= 1


def test_the_layer_sweep_adjusts_its_own_multiplicity():
    trace = build_trace(n_groups=12, signal=3.0, seed=10)
    rows = analysis.layer_probe_sweep(trace, sites=("mlp_in",), n_splits=3, bootstrap_samples=200)
    assert len(rows) == len(trace.layers) * len(trace.roles)
    assert all(row["p_sign_flip_bh"] >= row["p_sign_flip"] - 1e-12 for row in rows)


def test_a_dense_trace_says_why_routing_is_unavailable():
    trace = build_trace(n_groups=8, routing=False, seed=11)
    assert not trace.has_routing
    with pytest.raises(ValueError, match="dense model"):
        analysis.require_routing(trace)


def test_selectivity_change_compares_the_same_probe_protocol_across_checkpoints():
    base, tuned, _, _ = routed_pair(hidden_shift=0.5)
    report = delta.selectivity_change(
        base, tuned, site="mlp_in", layer_index=8, role="prompt_last", n_splits=3, bootstrap_samples=200
    )
    assert set(report) >= {"base", "tuned", "forced_choice_accuracy_delta", "selectivity_delta"}
    assert np.isfinite(report["forced_choice_accuracy_delta"])


def test_the_full_report_explains_each_section_it_cannot_produce():
    base = build_trace(n_groups=10, routing=False, seed=12)
    tuned = build_trace(n_groups=10, routing=False, seed=12)
    report = delta.compare_traces(base, tuned)
    assert report["texts_identical"] is True
    assert "unavailable" in report["routing"]
    assert "unavailable" in report["routing_decomposition"]
    assert "do not imply identical routing" in report["routing_decomposition"]["note"]
    assert "both checkpoints resident in memory" in report["block_output_decomposition"]["unavailable"]
    assert report["activation"]
