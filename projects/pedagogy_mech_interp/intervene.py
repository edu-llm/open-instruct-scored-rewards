"""Held-out activation, router, and expert-output interventions.

The endpoint is a letter-order-balanced forced choice between the matched tutor
responses. Candidate sites and experts must come from ``candidate_freeze.json``.

Example:
    python -m projects.pedagogy_mech_interp.intervene \
        --rows projects/pedagogy_mech_interp/data/pairs/confirmatory.jsonl \
        --key projects/pedagogy_mech_interp/data/pairs/confirmatory_key.jsonl \
        --freeze results/olmoe/discovery/candidate_freeze.json \
        --directions results/olmoe/discovery/directions.npz \
        --out results/olmoe/interventions.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch

from projects.pedagogy_mech_interp.modeling import (
    decoder_layers,
    is_moe_block,
    moe_norm_topk_prob,
    moe_top_k,
    normalize_gate_output,
    renormalize_topk,
    replace_output_tensor,
    route,
    selected_expert_contributions,
    tensor_from_output,
)
from projects.pedagogy_mech_interp.trace import TEACHER_SYSTEM, load_model, model_device

CHOICE_INSTRUCTION = """Choose the tutor response that better fits the student's work and supports learning.
Judge correctness, diagnosis, amount of help, and the next thinking step. Reply with only A or B."""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_rows(rows_path: Path, key_path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(rows_path)
    key = {row["variant_id"]: row for row in read_jsonl(key_path)}
    out = []
    for row in rows:
        if row["variant_id"] not in key:
            raise ValueError(f"missing sealed key for {row['variant_id']}")
        out.append({**row, **key[row["variant_id"]]})
    return out


def paired(rows: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["pair_id"], []).append(row)
    out = []
    for pair_id, variants in groups.items():
        positive = [row for row in variants if row["label"] == 1]
        negative = [row for row in variants if row["label"] == 0]
        if len(positive) != 1 or len(negative) != 1:
            raise ValueError(f"{pair_id}: expected exactly one positive and one negative variant")
        out.append((positive[0], negative[0]))
    return out


def choice_messages(positive: dict[str, Any], negative: dict[str, Any], positive_is_a: bool) -> list[dict[str, str]]:
    option_a, option_b = (positive, negative) if positive_is_a else (negative, positive)
    prompt = f"""Question the student is working on:
{positive["question"]}

Student:
{positive["student_before"]}

Tutor response A:
{option_a["tutor_turn"]}

Tutor response B:
{option_b["tutor_turn"]}

{CHOICE_INSTRUCTION}"""
    return [{"role": "system", "content": TEACHER_SYSTEM}, {"role": "user", "content": prompt}]


def choice_input(tokenizer, positive: dict, negative: dict, positive_is_a: bool, max_len: int):
    text = tokenizer.apply_chat_template(
        choice_messages(positive, negative, positive_is_a),
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(text, add_special_tokens=False, return_tensors="pt", truncation=True, max_length=max_len)
    return encoded


def letter_token_id(tokenizer, letter: str) -> int:
    candidates = [letter, f" {letter}"]
    for candidate in candidates:
        ids = tokenizer(candidate, add_special_tokens=False).input_ids
        if len(ids) == 1:
            return int(ids[0])
    raise ValueError(f"{letter!r} is not a single token for {type(tokenizer).__name__}")


class ActivationEditor(AbstractContextManager):
    def __init__(
        self,
        model,
        layer_index: int,
        site: str,
        positions: list[int],
        direction: torch.Tensor,
        mode: str,
        value: float = 0.0,
    ):
        self.layer = decoder_layers(model)[layer_index]
        self.site = site
        self.positions = positions
        self.direction = direction
        self.mode = mode
        self.value = value
        self.handle = None

    def edit(self, tensor: torch.Tensor) -> torch.Tensor:
        edited = tensor.clone()
        direction = self.direction.to(device=tensor.device, dtype=tensor.dtype)
        selected = edited[:, self.positions]
        projection = torch.einsum("bph,h->bp", selected, direction)
        if self.mode == "add":
            selected = selected + self.value * direction
        elif self.mode == "remove":
            selected = selected - projection[..., None] * direction
        elif self.mode == "set_projection":
            selected = selected + (self.value - projection)[..., None] * direction
        else:
            raise ValueError(f"unknown activation edit mode {self.mode!r}")
        edited[:, self.positions] = selected
        return edited

    def __enter__(self):
        module = self.layer if self.site.startswith("residual_") else self.layer.mlp
        if self.site.endswith("_in"):

            def pre_hook(_module, args):
                return (self.edit(args[0]), *args[1:])

            self.handle = module.register_forward_pre_hook(pre_hook)
        elif self.site.endswith("_out"):

            def post_hook(_module, _args, output):
                return replace_output_tensor(output, self.edit(tensor_from_output(output)))

            self.handle = module.register_forward_hook(post_hook)
        else:
            raise ValueError(f"unsupported site {self.site!r}")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.handle is not None:
            self.handle.remove()
        return False


class RouterEditor(AbstractContextManager):
    def __init__(self, block, positions: list[int], expert_index: int, mode: str, delta: float = 20.0):
        self.block = block
        self.positions = positions
        self.expert_index = expert_index
        self.mode = mode
        self.delta = delta
        self.handle = None

    def __enter__(self):
        def hook(_module, _args, output):
            routing = normalize_gate_output(self.block, output)
            logits = routing.logits.clone()
            if self.mode == "suppress":
                logits[self.positions, self.expert_index] = torch.finfo(logits.dtype).min
            elif self.mode == "force":
                maximum = logits[self.positions].max(dim=-1).values
                logits[self.positions, self.expert_index] = maximum + self.delta
            else:
                raise ValueError(f"unknown router mode {self.mode!r}")
            changed = renormalize_topk(logits, moe_top_k(self.block), moe_norm_topk_prob(self.block))
            if isinstance(output, tuple) and len(output) >= 3:
                return (changed.logits, changed.weights, changed.indices, *output[3:])
            return changed.logits

        self.handle = self.block.gate.register_forward_hook(hook)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.handle is not None:
            self.handle.remove()
        return False


class ExpertOutputScaler(AbstractContextManager):
    """Scale one expert contribution while retaining the original routing."""

    def __init__(
        self,
        block,
        positions: list[int],
        expert_index: int | list[int] | None,
        scale: float,
        active_rank: int = 0,
    ):
        self.block = block
        self.positions = positions
        self.expert_index = expert_index
        self.scale = scale
        self.active_rank = active_rank
        self.block_input = None
        self.handles = []

    def __enter__(self):
        def pre_hook(_module, args):
            self.block_input = args[0]

        def post_hook(_module, _args, output):
            if self.block_input is None:
                raise RuntimeError("expert output hook ran without captured input")
            routing = route(self.block, self.block_input)
            contributions = selected_expert_contributions(self.block, self.block_input, routing)
            if self.expert_index is None:
                expert_mask = torch.zeros_like(routing.indices, dtype=torch.bool)
                expert_mask[self.positions, self.active_rank % routing.indices.shape[1]] = True
            elif isinstance(self.expert_index, list):
                wanted = torch.as_tensor(self.expert_index, device=routing.indices.device)
                expert_mask = torch.isin(routing.indices, wanted)
            else:
                expert_mask = routing.indices == self.expert_index
            selected = (contributions * expert_mask[..., None]).sum(dim=1)
            value = tensor_from_output(output).clone()
            flat = value.reshape(-1, value.shape[-1])
            flat[self.positions] += (self.scale - 1.0) * selected[self.positions].to(flat.dtype)
            return replace_output_tensor(output, flat.reshape_as(value))

        self.handles.append(self.block.register_forward_pre_hook(pre_hook))
        self.handles.append(self.block.register_forward_hook(post_hook))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self.handles:
            handle.remove()
        return False


def activation_editor_factory(model, layer_index, site, direction, mode, value=0.0):
    def make(position):
        return ActivationEditor(model, layer_index, site, [position], direction, mode, value)

    return make


def router_editor_factory(block, expert_index, mode):
    def make(position):
        return RouterEditor(block, [position], expert_index, mode)

    return make


def expert_editor_factory(block, expert_index, scale, active_rank=0):
    def make(position):
        return ExpertOutputScaler(block, [position], expert_index, scale, active_rank)

    return make


def score_order(
    model,
    tokenizer,
    positive: dict,
    negative: dict,
    positive_is_a: bool,
    max_len: int,
    editor: Callable[[int], AbstractContextManager] | None = None,
) -> dict[str, float]:
    encoded = choice_input(tokenizer, positive, negative, positive_is_a, max_len)
    device = model_device(model)
    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)
    decision_position = int(input_ids.shape[1] - 1)
    context = editor(decision_position) if editor else nullcontext()
    with context, torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits[0, -1].float()
    a_id, b_id = letter_token_id(tokenizer, "A"), letter_token_id(tokenizer, "B")
    raw = float((logits[a_id] - logits[b_id]).item())
    return {
        "signed_logit_difference": raw if positive_is_a else -raw,
        "letter_logit_difference": raw,
    }


def balanced_choice_score(model, tokenizer, positive, negative, max_len, editor=None) -> dict[str, float]:
    first = score_order(model, tokenizer, positive, negative, True, max_len, editor)
    second = score_order(model, tokenizer, positive, negative, False, max_len, editor)
    return {
        "score": 0.5 * (first["signed_logit_difference"] + second["signed_logit_difference"]),
        "order_a_positive": first["signed_logit_difference"],
        "order_b_positive": second["signed_logit_difference"],
        "letter_bias": 0.5 * (first["letter_logit_difference"] + second["letter_logit_difference"]),
    }


def random_orthogonal(direction: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    value = rng.standard_normal(direction.shape).astype(np.float32)
    value -= np.dot(value, direction) * direction
    norm = np.linalg.norm(value)
    if norm == 0:
        raise ValueError("random direction collapsed after orthogonalization")
    return value / norm


def annotated_preference(positive: dict, negative: dict, key: str) -> int | None:
    """Return +1 when the pedagogical positive owns an independent Yes label."""
    positive_value = str((positive.get("annotations") or {}).get(key, "")).lower()
    negative_value = str((negative.get("annotations") or {}).get(key, "")).lower()
    positive_yes = positive_value == "yes"
    negative_yes = negative_value == "yes"
    if positive_yes == negative_yes:
        return None
    return 1 if positive_yes else -1


def write_result(handle, *, positive, model_name, intervention, baseline, changed, extra=None):
    negative = (extra or {}).get("_negative")
    clean_extra = {key: value for key, value in (extra or {}).items() if not key.startswith("_")}
    coherence_preference = annotated_preference(positive, negative, "Coherence") if negative else None
    fluency_preference = annotated_preference(positive, negative, "humanlikeness") if negative else None
    row = {
        "schema": "pedagogy-mech-intervention/v1",
        "model": model_name,
        "concept": positive["concept"],
        "item_id": positive["item_id"],
        "pair_id": positive["pair_id"],
        "label_source": positive.get("label_source", "unknown"),
        "intervention": intervention,
        "baseline": baseline["score"],
        "changed": changed["score"],
        "effect": changed["score"] - baseline["score"],
        "baseline_letter_bias": baseline["letter_bias"],
        "changed_letter_bias": changed["letter_bias"],
        "coherence_choice_change": (
            coherence_preference * (changed["score"] - baseline["score"])
            if coherence_preference is not None
            else None
        ),
        "humanlikeness_choice_change": (
            fluency_preference * (changed["score"] - baseline["score"])
            if fluency_preference is not None
            else None
        ),
        **clean_extra,
    }
    handle.write(json.dumps(row) + "\n")
    handle.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--directions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="")
    parser.add_argument("--revision", default="")
    parser.add_argument("--device-map", default="")
    parser.add_argument("--max-len", type=int, default=2048)
    parser.add_argument("--limit-pairs", type=int, default=0)
    parser.add_argument("--force-failed-gates", action="store_true")
    parser.add_argument("--seed", type=int, default=1701)
    args = parser.parse_args()

    freeze = json.loads(args.freeze.read_text())
    model_name = args.model or freeze["model"]
    with np.load(args.directions, allow_pickle=False) as blob:
        direction_metadata = json.loads(str(blob["metadata"]))
        directions = {concept: blob[meta["array"]].astype(np.float32) for concept, meta in direction_metadata.items()}
    rows = load_rows(args.rows, args.key)
    pairs = paired(rows)
    if args.limit_pairs:
        pairs = pairs[: args.limit_pairs]
    model, tokenizer = load_model(model_name, args.revision, args.device_map)
    layers = decoder_layers(model)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        for pair_index, (positive, negative) in enumerate(pairs):
            concept = positive["concept"]
            candidate = freeze["candidates"][concept]
            gates_pass = bool(candidate["behavior_pass"] and candidate["decodability_pass"])
            layer_index = int(candidate["layer_index"])
            site = candidate["site"]
            direction = directions[concept]
            metadata = direction_metadata[concept]
            projection_std = max(float(metadata["projection_std"]), 1e-6)
            positive_centroid = float(metadata["positive_projection_mean"])
            direction_tensor = torch.from_numpy(direction)
            baseline = balanced_choice_score(model, tokenizer, positive, negative, args.max_len)
            if not args.force_failed_gates and not gates_pass:
                write_result(
                    handle,
                    positive=positive,
                    model_name=model_name,
                    intervention="behavior_baseline",
                    baseline=baseline,
                    changed=baseline,
                    extra={"_negative": negative, "gate_limited": True},
                )
                continue

            identity = balanced_choice_score(
                model,
                tokenizer,
                positive,
                negative,
                args.max_len,
                activation_editor_factory(model, layer_index, site, direction_tensor, "add", 0.0),
            )
            if abs(identity["score"] - baseline["score"]) > 1e-5:
                raise AssertionError(f"{positive['pair_id']}: zero-dose hook changed logits")

            for dose in (-2.0, -1.0, 1.0, 2.0):
                changed = balanced_choice_score(
                    model,
                    tokenizer,
                    positive,
                    negative,
                    args.max_len,
                    activation_editor_factory(
                        model, layer_index, site, direction_tensor, "add", dose * projection_std
                    ),
                )
                write_result(
                    handle,
                    positive=positive,
                    model_name=model_name,
                    intervention="direction_steer",
                    baseline=baseline,
                    changed=changed,
                    extra={"_negative": negative, "dose": dose, "layer_index": layer_index, "site": site},
                )

            ablated = balanced_choice_score(
                model,
                tokenizer,
                positive,
                negative,
                args.max_len,
                activation_editor_factory(model, layer_index, site, direction_tensor, "remove"),
            )
            write_result(
                handle,
                positive=positive,
                model_name=model_name,
                intervention="direction_ablation",
                baseline=baseline,
                changed=ablated,
                extra={"_negative": negative, "layer_index": layer_index, "site": site},
            )
            rescued = balanced_choice_score(
                model,
                tokenizer,
                positive,
                negative,
                args.max_len,
                activation_editor_factory(
                    model,
                    layer_index,
                    site,
                    direction_tensor,
                    "set_projection",
                    positive_centroid,
                ),
            )
            write_result(
                handle,
                positive=positive,
                model_name=model_name,
                intervention="direction_centroid_rescue",
                baseline=ablated,
                changed=rescued,
                extra={
                    "_negative": negative,
                    "clean_baseline": baseline["score"],
                    "layer_index": layer_index,
                    "site": site,
                    "target_projection": positive_centroid,
                },
            )

            random_direction = torch.from_numpy(random_orthogonal(direction, args.seed + pair_index))
            random_changed = balanced_choice_score(
                model,
                tokenizer,
                positive,
                negative,
                args.max_len,
                activation_editor_factory(
                    model, layer_index, site, random_direction, "add", 2.0 * projection_std
                ),
            )
            write_result(
                handle,
                positive=positive,
                model_name=model_name,
                intervention="norm_matched_random_steer",
                baseline=baseline,
                changed=random_changed,
                extra={"_negative": negative, "dose": 2.0, "layer_index": layer_index, "site": site},
            )

            router_candidate = candidate.get("router_candidate")
            block = layers[int(router_candidate["layer_index"])].mlp if router_candidate else None
            if router_candidate and is_moe_block(block):
                expert_index = int(router_candidate["expert_index"])
                for mode in ("suppress", "force"):
                    changed = balanced_choice_score(
                        model,
                        tokenizer,
                        positive,
                        negative,
                        args.max_len,
                        router_editor_factory(block, expert_index, mode),
                    )
                    write_result(
                        handle,
                        positive=positive,
                        model_name=model_name,
                        intervention=f"router_{mode}",
                        baseline=baseline,
                        changed=changed,
                        extra={
                            "_negative": negative,
                            "layer_index": router_candidate["layer_index"],
                            "expert_index": expert_index,
                        },
                    )
                expert_ablated = balanced_choice_score(
                    model,
                    tokenizer,
                    positive,
                    negative,
                    args.max_len,
                    expert_editor_factory(block, expert_index, 0.0),
                )
                write_result(
                    handle,
                    positive=positive,
                    model_name=model_name,
                    intervention="expert_output_ablation",
                    baseline=baseline,
                    changed=expert_ablated,
                    extra={
                        "_negative": negative,
                        "layer_index": router_candidate["layer_index"],
                        "expert_index": expert_index,
                    },
                )
                expert_amplified = balanced_choice_score(
                    model,
                    tokenizer,
                    positive,
                    negative,
                    args.max_len,
                    expert_editor_factory(block, expert_index, 2.0),
                )
                write_result(
                    handle,
                    positive=positive,
                    model_name=model_name,
                    intervention="expert_output_amplification",
                    baseline=baseline,
                    changed=expert_amplified,
                    extra={
                        "_negative": negative,
                        "layer_index": router_candidate["layer_index"],
                        "expert_index": expert_index,
                    },
                )
                write_result(
                    handle,
                    positive=positive,
                    model_name=model_name,
                    intervention="expert_output_rescue",
                    baseline=expert_ablated,
                    changed=baseline,
                    extra={
                        "_negative": negative,
                        "clean_baseline": baseline["score"],
                        "layer_index": router_candidate["layer_index"],
                        "expert_index": expert_index,
                        "claim_limit": "identity restoration control; not independent evidence of semantic specificity",
                    },
                )
                coalition = [int(value) for value in router_candidate.get("coalition_experts", [])]
                if len(coalition) > 1:
                    coalition_ablated = balanced_choice_score(
                        model,
                        tokenizer,
                        positive,
                        negative,
                        args.max_len,
                        expert_editor_factory(block, coalition, 0.0),
                    )
                    write_result(
                        handle,
                        positive=positive,
                        model_name=model_name,
                        intervention="expert_coalition_ablation",
                        baseline=baseline,
                        changed=coalition_ablated,
                        extra={
                            "_negative": negative,
                            "layer_index": router_candidate["layer_index"],
                            "expert_indices": coalition,
                        },
                    )
                random_ablated = balanced_choice_score(
                    model,
                    tokenizer,
                    positive,
                    negative,
                    args.max_len,
                    expert_editor_factory(
                        block, None, 0.0, active_rank=pair_index % moe_top_k(block)
                    ),
                )
                write_result(
                    handle,
                    positive=positive,
                    model_name=model_name,
                    intervention="random_active_rank_expert_ablation",
                    baseline=baseline,
                    changed=random_ablated,
                    extra={
                        "_negative": negative,
                        "layer_index": router_candidate["layer_index"],
                        "active_rank": pair_index % moe_top_k(block),
                    },
                )
            if (pair_index + 1) % 5 == 0:
                print(f"  {pair_index + 1}/{len(pairs)} pairs")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
