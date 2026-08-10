"""What fine-tuning changed, measured on texts that did not change.

Both checkpoints replay the same fixed prompts and the same fixed responses, so
every difference reported here is a difference in the model. ``compare_traces``
refuses to run if the two traces disagree about a single character.

The routing question this module exists for is easy to get wrong. The RL run
that produced ``ckpt/pedagogy_olmoe/global_step*`` froze ``mlp.gate`` and put
LoRA on the experts. A frozen router is a statement about parameters, not about
behaviour: expert adaptation changes the hidden state that the next layer's
router reads, so selection moves even though no router weight did. Nothing here
infers "routing unchanged" from "router frozen"; ``decompose_routing_change``
separates the two channels and reports each measured value.

Three channels are exposed:

- ``router_parameter_shift``: the router weights applied to the same input;
- ``router_input_shift``: the same router weights applied to the shifted input,
  which is the channel a frozen router leaves wide open;
- ``expert_adaptation``: the experts' own output under an identical input and
  an identical selection, which needs both checkpoints in memory.

Checkpoint support is deliberately narrow and loud. ``ckpt/pedagogy-tutor-armF``
is a PEFT LoRA adapter over dense ``allenai/OLMo-2-1124-7B-Instruct`` and loads.
``/orcd/.../ckpt/pedagogy_olmoe/global_step{180,190,200}`` are DeepSpeed ZeRO
partitions of an OLMoE policy whose adapter configuration is not stored beside
them; ``describe_checkpoint`` says so and loading raises rather than guessing.
Classification reads what a path contains rather than what it is called, so a
consolidated export written back into a ``global_step<N>`` directory is treated
as the checkpoint it is, and ``load_traced_model`` decides before it spends a
7B load rather than after.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from projects.learning_science_semantic_atlas.trace import SITES, AtlasTrace, load_atlas_trace
from projects.pedagogy_mech_interp.modeling import (
    Routing,
    decoder_layers,
    is_moe_block,
    moe_norm_topk_prob,
    moe_top_k,
    route,
    selected_expert_contributions,
)
from projects.pedagogy_mech_interp.trace import load_model

SCHEMA = "semantic-atlas-checkpoint-delta/v1"
# Traces store the router input in float16, so recomputing the router from the
# weight sidecar moves probabilities by roughly a percent on a 2048-wide hidden
# state. A sidecar belonging to a different checkpoint moves them by order one,
# which is why this threshold sits between the two rather than near zero.
ROUTING_RECONSTRUCTION_TOLERANCE = 0.05
# Everything a routing comparison reads. ``router_logits`` alone is what
# ``has_routing`` reports; the derived expert-set and path arrays are written by
# the same tracer pass and are named here so an older trace says so.
ROUTING_ARRAYS = ("router_logits", "router_probs", "topk_weights", "topk_indices", "expert_set_id", "expert_path_id")
FROZEN_ROUTER_NOTE = (
    "identical router parameters do not imply identical routing: selection reads the router input, "
    "which upstream adaptation shifts"
)
# A DeepSpeed ZeRO step directory is identified by its shards, not by its name: an
# export written to ``global_step<N>/`` is a perfectly ordinary checkpoint.
DEEPSPEED_SHARD_SUFFIXES = ("_model_states.pt", "_optim_states.pt")
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")
WEIGHT_INDEX_FILES = ("model.safetensors.index.json", "pytorch_model.bin.index.json")


class CheckpointError(RuntimeError):
    """A checkpoint could not be described or loaded."""


class AdapterLoadingUnsupported(CheckpointError):
    """The adapter format is recognized but cannot be applied to a live model."""


# --------------------------------------------------------------------------
# checkpoint description and loading
# --------------------------------------------------------------------------


@dataclass
class CheckpointDescription:
    """Everything decidable about a checkpoint without materializing weights."""

    path: str
    kind: str
    loadable: bool
    reason: str = ""
    base_model: str | None = None
    model_type: str | None = None
    architecture: str | None = None
    target_modules: list[str] = field(default_factory=list)
    target_parameters: list[str] = field(default_factory=list)
    adapts_experts: bool = False
    adapts_router: bool = False
    entries: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def require_loadable(self) -> CheckpointDescription:
        if not self.loadable:
            raise AdapterLoadingUnsupported(f"{self.path}: {self.reason}")
        return self


def _architecture_of(model_type: str | None, base_model: str | None) -> str | None:
    haystack = f"{model_type or ''} {base_model or ''}".lower()
    if "olmoe" in haystack or "moe" in haystack:
        return "mixture_of_experts"
    if "olmo" in haystack or "llama" in haystack or "qwen" in haystack:
        return "dense"
    return None


def _is_router_target(target: str) -> bool:
    """OLMoE's router is the module literally named ``gate``.

    ``gate_proj`` is a dense MLP projection and ``experts.gate_up_proj`` is a
    fused expert weight. Matching on the substring would report every ordinary
    MLP adapter as a router adapter, which is the one mistake this whole module
    exists to avoid.
    """
    leaf = target.rsplit(".", 1)[-1]
    return leaf == "gate" or "router" in leaf


def _describe_adapter_config(path: str, config: dict[str, Any], entries: list[str]) -> CheckpointDescription:
    modules = [str(value) for value in (config.get("target_modules") or [])]
    parameters = [str(value) for value in (config.get("target_parameters") or [])]
    targets = modules + parameters
    base_model = config.get("base_model_name_or_path")
    peft_type = str(config.get("peft_type", "")).upper()
    return CheckpointDescription(
        path=path,
        kind="peft_adapter",
        loadable=peft_type == "LORA",
        reason="" if peft_type == "LORA" else f"peft_type {peft_type or 'unknown'} is not supported here",
        base_model=base_model,
        architecture=_architecture_of(None, base_model),
        target_modules=modules,
        target_parameters=parameters,
        adapts_experts=any("expert" in target for target in targets),
        adapts_router=any(_is_router_target(target) for target in targets),
        entries=entries,
    )


def _has_weights(entries: list[str]) -> bool:
    return any(name.endswith(WEIGHT_SUFFIXES) for name in entries) or any(
        name in entries for name in WEIGHT_INDEX_FILES
    )


def _describe_hf_model(path: str, config: dict[str, Any], entries: list[str]) -> CheckpointDescription:
    """A directory holding a whole model. Loadable only if the weights are there too.

    A ``config.json`` on its own describes an architecture; it is not a
    checkpoint. Reporting it as loadable would move the failure from here to the
    middle of a 7B ``from_pretrained``.
    """
    model_type = config.get("model_type")
    weights = _has_weights(entries)
    return CheckpointDescription(
        path=path,
        kind="hf_model",
        loadable=weights,
        reason="" if weights else "config.json is present but no weight file is, so there is nothing to load",
        model_type=model_type,
        architecture=_architecture_of(model_type, None),
        entries=entries,
    )


def describe_checkpoint(path: Path | str) -> CheckpointDescription:
    """Classify a checkpoint. Reads json only; never loads a tensor."""
    path = Path(path)
    if not path.exists():
        return CheckpointDescription(path=str(path), kind="missing", loadable=False, reason="path does not exist")

    if path.is_file() and path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            entries = archive.namelist()
            adapter = next((name for name in entries if name.endswith("adapter_config.json")), None)
            if adapter:
                description = _describe_adapter_config(str(path), json.loads(archive.read(adapter)), entries[:40])
                description.kind = "peft_adapter_zip"
                return description
            config = next((name for name in entries if name.endswith("config.json")), None)
            if config:
                body = json.loads(archive.read(config))
                model_type = body.get("model_type")
                # An adapter zip is extracted and handed to PEFT; a whole-model zip
                # has no such path, because --model goes straight to from_pretrained.
                return CheckpointDescription(
                    path=str(path),
                    kind="hf_model_zip",
                    loadable=False,
                    reason="a zipped whole model cannot be loaded in place; extract it and pass the directory",
                    model_type=model_type,
                    architecture=_architecture_of(model_type, None),
                    entries=entries[:40],
                )
        return CheckpointDescription(
            path=str(path), kind="unknown_zip", loadable=False, reason="zip has neither adapter_config nor config"
        )

    if path.is_dir():
        entries = sorted(item.name for item in path.iterdir())
        if "adapter_config.json" in entries:
            body = json.loads((path / "adapter_config.json").read_text())
            return _describe_adapter_config(str(path), body, entries)
        if any(name.endswith(DEEPSPEED_SHARD_SUFFIXES) for name in entries):
            return _describe_deepspeed(path, entries)
        if "config.json" in entries:
            body = json.loads((path / "config.json").read_text())
            return _describe_hf_model(str(path), body, entries)
        if path.name.startswith("global_step"):
            return _describe_deepspeed(path, entries)
        return CheckpointDescription(
            path=str(path), kind="unknown_directory", loadable=False, reason="no config.json and no ZeRO shards"
        )

    return CheckpointDescription(path=str(path), kind="unknown", loadable=False, reason="not a directory or zip")


def _describe_deepspeed(path: Path, entries: list[str]) -> CheckpointDescription:
    """A ZeRO step directory. These are training states, not models.

    Reconstructing a model needs a consolidation pass and then one of two
    exports, and — because the policy was trained with per-expert LoRA behind a
    frozen router — an adapter export has to carry the configuration that names
    which parameters those extra tensors adapt.
    """
    siblings = sorted(item.name for item in path.parent.iterdir()) if path.parent.exists() else []
    reason = (
        "DeepSpeed ZeRO training state cannot be traced directly: consolidate it with "
        "deepspeed.utils.zero_to_fp32, then export either a merged model with a config.json or a PEFT "
        "adapter with the adapter_config.json naming the trained LoRA targets"
    )
    notes = []
    if "latest" not in siblings:
        notes.append("there is no 'latest' pointer beside this step, so the tag has to be passed explicitly")
    if not any(name.endswith("_model_states.pt") for name in entries):
        notes.append("this step holds no *_model_states.pt shards, so it may be partial")
    if notes:
        reason = f"{reason}. Also: {'; '.join(notes)}"
    return CheckpointDescription(
        path=str(path),
        kind="deepspeed_zero",
        loadable=False,
        reason=reason,
        architecture=_architecture_of(None, path.parent.name),
        entries=entries,
    )


def _adapter_directory(path: Path, stack) -> Path:
    if path.is_dir():
        return path
    import tempfile  # noqa: PLC0415

    target = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="atlas_adapter_")))
    with zipfile.ZipFile(path) as archive:
        archive.extractall(target)
    inner = next((item.parent for item in target.rglob("adapter_config.json")), None)
    if inner is None:
        raise AdapterLoadingUnsupported(f"{path}: zip contains no adapter_config.json")
    return inner


def apply_adapter(model, adapter: Path | str, *, merge: bool = True, allow_base_mismatch: bool = False):
    """Apply a PEFT LoRA adapter, merging it so hook-based tracing sees one plain model.

    Merging costs a small bf16 rounding error and buys an unwrapped module tree,
    which is what ``decoder_layers`` and the routing hooks expect.
    """
    from contextlib import ExitStack  # noqa: PLC0415

    adapter = Path(adapter)
    description = describe_checkpoint(adapter).require_loadable()
    if description.kind not in {"peft_adapter", "peft_adapter_zip"}:
        raise AdapterLoadingUnsupported(f"{adapter}: {description.kind} is not a PEFT adapter")
    configured_base = description.base_model
    loaded_base = getattr(getattr(model, "config", None), "_name_or_path", None)
    if configured_base and loaded_base and configured_base != loaded_base and not allow_base_mismatch:
        raise CheckpointError(
            f"{adapter} was trained on {configured_base!r} but the loaded base is {loaded_base!r}; "
            "pass allow_base_mismatch=True only if you intend to compare across bases"
        )
    try:
        from peft import PeftModel  # noqa: PLC0415
    except ImportError as exc:
        raise AdapterLoadingUnsupported(f"{adapter} needs the 'peft' package to load") from exc

    with ExitStack() as stack:
        directory = _adapter_directory(adapter, stack)
        wrapped = PeftModel.from_pretrained(model, str(directory), is_trainable=False)
        return wrapped.merge_and_unload() if merge else wrapped


def load_traced_model(
    model_name: str,
    *,
    revision: str = "",
    adapter: str = "",
    device_map: str = "",
    tokenizer=None,
    allow_base_mismatch: bool = False,
):
    """Load a base model, optionally with an adapter, ready for hook-based tracing.

    The adapter is classified first. A DeepSpeed ZeRO directory then costs a json
    read rather than a 7B ``from_pretrained`` followed by the same refusal.
    """
    if adapter:
        describe_checkpoint(adapter).require_loadable()
    model, loaded_tokenizer = load_model(model_name, revision, device_map)
    if adapter:
        model = apply_adapter(model, adapter, allow_base_mismatch=allow_base_mismatch)
    return model.eval(), tokenizer or loaded_tokenizer


# --------------------------------------------------------------------------
# array metrics
# --------------------------------------------------------------------------


def js_divergence(first: np.ndarray, second: np.ndarray, *, axis: int = -1) -> np.ndarray:
    """Jensen-Shannon divergence in bits, so 0 is identical and 1 is disjoint."""
    first = np.clip(np.asarray(first, dtype=np.float64), 1e-12, None)
    second = np.clip(np.asarray(second, dtype=np.float64), 1e-12, None)
    first = first / first.sum(axis=axis, keepdims=True)
    second = second / second.sum(axis=axis, keepdims=True)
    mean = 0.5 * (first + second)
    divergence = 0.5 * (first * np.log(first / mean)).sum(axis=axis) + 0.5 * (second * np.log(second / mean)).sum(
        axis=axis
    )
    return np.clip(divergence, 0.0, None) / np.log(2.0)


def linear_cka(first: np.ndarray, second: np.ndarray) -> float:
    """Linear CKA: 1 for identical geometry, invariant to rotation and scale."""
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.shape[0] != second.shape[0]:
        raise ValueError(f"CKA needs matched rows, got {first.shape} and {second.shape}")
    first = first - first.mean(axis=0, keepdims=True)
    second = second - second.mean(axis=0, keepdims=True)
    cross = float(np.linalg.norm(first.T @ second, ord="fro") ** 2)
    left = float(np.linalg.norm(first.T @ first, ord="fro"))
    right = float(np.linalg.norm(second.T @ second, ord="fro"))
    if left == 0.0 or right == 0.0:
        return float("nan")
    return cross / (left * right)


def mean_cosine(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    numerator = (first * second).sum(axis=-1)
    denominator = np.linalg.norm(first, axis=-1) * np.linalg.norm(second, axis=-1)
    valid = denominator > 0
    return float(np.mean(numerator[valid] / denominator[valid])) if valid.any() else float("nan")


def relative_l2(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    scale = np.linalg.norm(first, axis=-1)
    valid = scale > 0
    if not valid.any():
        return float("nan")
    return float(np.mean(np.linalg.norm(first - second, axis=-1)[valid] / scale[valid]))


def topk_overlap(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Fraction of the selected set the two checkpoints agree on, per token."""
    first = np.asarray(first)
    second = np.asarray(second)
    top_k = first.shape[-1]
    matches = (first[..., :, None] == second[..., None, :]).any(axis=-1).sum(axis=-1)
    return matches / float(top_k)


def rank_swaps(base_logits: np.ndarray, tuned_logits: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Normalized Kendall distance between the two orderings of the base top-k.

    Counts pairs of base-selected experts whose relative order flipped. Combined
    with ``topk_overlap``, which sees experts entering and leaving, this covers
    both ways a selection can move.
    """
    base_scores = np.take_along_axis(base_logits, indices, axis=-1).astype(np.float64)
    tuned_scores = np.take_along_axis(tuned_logits, indices, axis=-1).astype(np.float64)
    base_difference = np.sign(base_scores[..., :, None] - base_scores[..., None, :])
    tuned_difference = np.sign(tuned_scores[..., :, None] - tuned_scores[..., None, :])
    top_k = indices.shape[-1]
    upper = np.triu(np.ones((top_k, top_k), dtype=bool), k=1)
    discordant = ((base_difference != tuned_difference) & upper).sum(axis=(-2, -1))
    pairs = top_k * (top_k - 1) / 2
    return discordant / pairs if pairs else np.zeros_like(discordant, dtype=np.float64)


def entropy_bits(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, None)
    probabilities = probabilities / probabilities.sum(axis=-1, keepdims=True)
    return -(probabilities * np.log(probabilities)).sum(axis=-1) / np.log(2.0)


def usage_histogram(indices: np.ndarray, experts: int) -> np.ndarray:
    counts = np.bincount(np.asarray(indices).reshape(-1), minlength=experts).astype(np.float64)
    total = counts.sum()
    return counts / total if total else counts


# --------------------------------------------------------------------------
# trace comparison
# --------------------------------------------------------------------------


def check_comparable(base: AtlasTrace, tuned: AtlasTrace) -> None:
    """Refuse to compare unless both checkpoints saw byte-identical inputs."""
    if not np.array_equal(base["ids"], tuned["ids"]):
        raise ValueError("traces cover different records or a different record order")
    for key in ("text_sha256", "token_sha256"):
        differing = np.flatnonzero(base[key] != tuned[key])
        if differing.size:
            first = [str(value) for value in base["ids"][differing[:3]]]
            raise ValueError(f"{differing.size} records differ in {key}; the texts are not fixed, first: {first}")
    if not np.array_equal(base["layers"], tuned["layers"]):
        raise ValueError(f"traced layers differ: {base.layers} against {tuned.layers}")
    if base.roles != tuned.roles:
        raise ValueError("traced position roles differ")
    if not np.array_equal(base["position_index"], tuned["position_index"]):
        raise ValueError("resolved token positions differ despite identical text")
    if base.has_routing and tuned.has_routing:
        for key in ("router_logits", "topk_indices"):
            if base[key].shape[1:] != tuned[key].shape[1:]:
                raise ValueError(
                    f"{key} is {base[key].shape[1:]} in the base trace and {tuned[key].shape[1:]} in the tuned "
                    "trace; the two mixtures disagree on expert count or top-k and are not comparable"
                )


def require_comparable_routing(base: AtlasTrace, tuned: AtlasTrace, keys: Sequence[str] = ROUTING_ARRAYS) -> None:
    """Refuse a routing comparison whose two sides do not measure the same thing.

    ``has_routing`` only reports that router logits were stored. The derived
    per-layer expert sets and cross-layer paths come from the same tracer version,
    and the native top-k weights only mean the same thing on both sides when both
    checkpoints renormalize them the same way.
    """
    for trace, name in ((base, "base"), (tuned, "tuned")):
        if not trace.has_routing:
            raise ValueError("routing metrics need two mixture-of-experts traces")
        missing = [key for key in keys if key not in trace]
        if missing:
            raise ValueError(f"the {name} trace is missing {missing}; re-trace it with the current tracer")
    if base.metadata.get("norm_topk_prob") != tuned.metadata.get("norm_topk_prob"):
        raise ValueError(
            f"the base trace sets norm_topk_prob={base.metadata.get('norm_topk_prob')!r} and the tuned trace "
            f"sets {tuned.metadata.get('norm_topk_prob')!r}; the native top-k weights are on different scales"
        )


def resolve_slots(trace: AtlasTrace, slots: Sequence[int] | Sequence[str] | None) -> list[int]:
    if slots is None:
        return list(range(len(trace.roles)))
    return [trace.slot(value) if isinstance(value, str) else int(value) for value in slots]


def position_mask(base: AtlasTrace, tuned: AtlasTrace, slots: Sequence[int]) -> np.ndarray:
    """``[records, slots]`` positions both traces recorded."""
    return base["position_mask"][:, slots].astype(bool) & tuned["position_mask"][:, slots].astype(bool)


def flatten_positions(trace: AtlasTrace, key: str, slots: Sequence[int], mask: np.ndarray) -> np.ndarray:
    """``[record, layer, slot, ...]`` to ``[token, layer, ...]`` over unmasked positions."""
    value = trace[key]
    if value.ndim >= 3 and value.shape[1] == len(trace.layers):
        moved = np.moveaxis(value[:, :, slots], 2, 1)  # [record, slot, layer, ...]
        return moved[mask]
    return value[:, slots][mask]


def routing_delta(
    base: AtlasTrace, tuned: AtlasTrace, *, slots: Sequence[int] | Sequence[str] | None = None
) -> dict[str, Any]:
    """Router divergence, selection change, and expert-path change, per layer."""
    check_comparable(base, tuned)
    require_comparable_routing(base, tuned)
    slot_indices = resolve_slots(base, slots)
    mask = position_mask(base, tuned, slot_indices)
    if not mask.any():
        raise ValueError("the requested slots are empty in at least one trace")

    base_probs = flatten_positions(base, "router_probs", slot_indices, mask).astype(np.float64)
    tuned_probs = flatten_positions(tuned, "router_probs", slot_indices, mask).astype(np.float64)
    base_logits = flatten_positions(base, "router_logits", slot_indices, mask).astype(np.float64)
    tuned_logits = flatten_positions(tuned, "router_logits", slot_indices, mask).astype(np.float64)
    base_indices = flatten_positions(base, "topk_indices", slot_indices, mask).astype(np.int64)
    tuned_indices = flatten_positions(tuned, "topk_indices", slot_indices, mask).astype(np.int64)
    base_weights = flatten_positions(base, "topk_weights", slot_indices, mask).astype(np.float64)
    tuned_weights = flatten_positions(tuned, "topk_weights", slot_indices, mask).astype(np.float64)
    base_paths = base["expert_path_id"][:, slot_indices][mask]
    tuned_paths = tuned["expert_path_id"][:, slot_indices][mask]
    base_sets = flatten_positions(base, "expert_set_id", slot_indices, mask)
    tuned_sets = flatten_positions(tuned, "expert_set_id", slot_indices, mask)

    experts = base_probs.shape[-1]
    divergence = js_divergence(base_probs, tuned_probs)
    overlap = topk_overlap(base_indices, tuned_indices)
    swaps = rank_swaps(base_logits, tuned_logits, base_indices)
    base_entropy = entropy_bits(base_probs)
    tuned_entropy = entropy_bits(tuned_probs)
    base_selected = np.take_along_axis(base_probs, base_indices, axis=-1).sum(axis=-1)
    tuned_selected = np.take_along_axis(tuned_probs, tuned_indices, axis=-1).sum(axis=-1)

    layers = base.layers
    per_layer = []
    for position, layer_index in enumerate(layers):
        base_usage = usage_histogram(base_indices[:, position], experts)
        tuned_usage = usage_histogram(tuned_indices[:, position], experts)
        per_layer.append(
            {
                "layer_index": layer_index,
                "router_js_bits": float(divergence[:, position].mean()),
                "router_js_bits_p95": float(np.quantile(divergence[:, position], 0.95)),
                "topk_overlap": float(overlap[:, position].mean()),
                "rank_swaps_within_base_topk": float(swaps[:, position].mean()),
                "top1_change_rate": float((base_indices[:, position, 0] != tuned_indices[:, position, 0]).mean()),
                "expert_set_change_rate": float((base_sets[:, position] != tuned_sets[:, position]).mean()),
                "entropy_bits_base": float(base_entropy[:, position].mean()),
                "entropy_bits_tuned": float(tuned_entropy[:, position].mean()),
                "entropy_bits_delta": float((tuned_entropy - base_entropy)[:, position].mean()),
                "native_topk_mass_base": float(base_weights[:, position].sum(axis=-1).mean()),
                "native_topk_mass_tuned": float(tuned_weights[:, position].sum(axis=-1).mean()),
                "selected_probability_mass_base": float(base_selected[:, position].mean()),
                "selected_probability_mass_tuned": float(tuned_selected[:, position].mean()),
                "utilization_js_bits": float(js_divergence(base_usage, tuned_usage)),
                "experts_used_base": int((base_usage > 0).sum()),
                "experts_used_tuned": int((tuned_usage > 0).sum()),
                "usage_entropy_base": float(entropy_bits(base_usage) / np.log2(experts)),
                "usage_entropy_tuned": float(entropy_bits(tuned_usage) / np.log2(experts)),
                "max_expert_load_base": float(base_usage.max()),
                "max_expert_load_tuned": float(tuned_usage.max()),
            }
        )
    changed_layers = (base_sets != tuned_sets).sum(axis=1)
    return {
        "tokens": int(mask.sum()),
        "experts": experts,
        "slots": [base.roles[index] for index in slot_indices],
        "per_layer": per_layer,
        "expert_path_change_rate": float((base_paths != tuned_paths).mean()),
        "mean_layers_with_changed_expert_set": float(changed_layers.mean()),
        "native_topk_weights_normalized": bool(base.metadata.get("norm_topk_prob")),
        "router_js_bits": float(divergence.mean()),
        "topk_overlap": float(overlap.mean()),
    }


def activation_delta(
    base: AtlasTrace,
    tuned: AtlasTrace,
    *,
    sites: Sequence[str] | None = None,
    slots: Sequence[int] | Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """CKA, cosine, and relative L2 per site and layer over shared positions."""
    check_comparable(base, tuned)
    slot_indices = resolve_slots(base, slots)
    mask = position_mask(base, tuned, slot_indices)
    chosen = [site for site in (sites or SITES) if site in base and site in tuned]
    rows = []
    for site in chosen:
        base_values = flatten_positions(base, site, slot_indices, mask).astype(np.float32)
        tuned_values = flatten_positions(tuned, site, slot_indices, mask).astype(np.float32)
        for position, layer_index in enumerate(base.layers):
            rows.append(
                {
                    "site": site,
                    "layer_index": layer_index,
                    "linear_cka": linear_cka(base_values[:, position], tuned_values[:, position]),
                    "mean_cosine": mean_cosine(base_values[:, position], tuned_values[:, position]),
                    "relative_l2": relative_l2(base_values[:, position], tuned_values[:, position]),
                }
            )
    return rows


def selectivity_change(
    base: AtlasTrace,
    tuned: AtlasTrace,
    *,
    site: str,
    layer_index: int,
    role: str,
    texts: Sequence[str] | None = None,
    **scoring: Any,
) -> dict[str, Any]:
    """Did fine-tuning make the construct more or less linearly available?"""
    from projects.learning_science_semantic_atlas.analyze_representations import (  # noqa: PLC0415
        quality_matched_sibling_discrimination,
    )

    check_comparable(base, tuned)
    slot = base.slot(role)
    before = quality_matched_sibling_discrimination(
        base, site=site, layer_index=layer_index, slot=slot, texts=texts, **scoring
    )
    after = quality_matched_sibling_discrimination(
        tuned, site=site, layer_index=layer_index, slot=slot, texts=texts, **scoring
    )
    return {
        "site": site,
        "layer_index": layer_index,
        "role": role,
        "base": before,
        "tuned": after,
        "forced_choice_accuracy_delta": (
            after["activation"]["forced_choice_accuracy"] - before["activation"]["forced_choice_accuracy"]
        ),
        "selectivity_delta": after["selectivity_over_best_control"] - before["selectivity_over_best_control"],
        "claim_limit": (
            "a selectivity change is a change in linear availability at one site, not evidence that the "
            "model's use of the construct changed"
        ),
    }


# --------------------------------------------------------------------------
# decomposition
# --------------------------------------------------------------------------


def load_router_weights(path: Path | str) -> dict[int, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as blob:
        return {int(key.split("_")[1]): blob[key].astype(np.float64) for key in blob.files if key.startswith("layer_")}


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=-1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=-1, keepdims=True)


def decompose_routing_change(
    base: AtlasTrace,
    tuned: AtlasTrace,
    base_router: dict[int, np.ndarray],
    tuned_router: dict[int, np.ndarray],
    *,
    slots: Sequence[int] | Sequence[str] | None = None,
    tolerance: float = ROUTING_RECONSTRUCTION_TOLERANCE,
) -> dict[str, Any]:
    """Split the routing change into a router-parameter channel and an input channel.

    Four routings are compared at each layer: the base router on its own input,
    the base router on the tuned input, the tuned router on the base input, and
    the observed tuned routing. The parameter channel is exactly zero when the
    router is frozen; the input channel is what expert adaptation opens, and it
    is reported whether or not the parameters moved.

    The channels do not sum to the total. The residual is reported as
    ``interaction_js_bits`` rather than distributed.
    """
    check_comparable(base, tuned)
    require_comparable_routing(base, tuned, keys=("router_logits", "router_probs", "topk_indices"))
    if "mlp_in" not in base or "mlp_in" not in tuned:
        raise ValueError("routing decomposition needs the 'mlp_in' site, which is the router input")
    slot_indices = resolve_slots(base, slots)
    mask = position_mask(base, tuned, slot_indices)
    base_inputs = flatten_positions(base, "mlp_in", slot_indices, mask).astype(np.float64)
    tuned_inputs = flatten_positions(tuned, "mlp_in", slot_indices, mask).astype(np.float64)
    observed_base = flatten_positions(base, "router_probs", slot_indices, mask).astype(np.float64)
    observed_tuned = flatten_positions(tuned, "router_probs", slot_indices, mask).astype(np.float64)
    top_k = int(base["topk_indices"].shape[-1])

    rows = []
    for position, layer_index in enumerate(base.layers):
        if layer_index not in base_router or layer_index not in tuned_router:
            raise ValueError(f"router weights for layer {layer_index} are missing from the sidecar files")
        base_weight = base_router[layer_index]
        tuned_weight = tuned_router[layer_index]
        base_on_base = _softmax(base_inputs[:, position] @ base_weight.T)
        base_on_tuned = _softmax(tuned_inputs[:, position] @ base_weight.T)
        tuned_on_base = _softmax(base_inputs[:, position] @ tuned_weight.T)
        tuned_on_tuned = _softmax(tuned_inputs[:, position] @ tuned_weight.T)

        errors = np.abs(
            np.concatenate([base_on_base - observed_base[:, position], tuned_on_tuned - observed_tuned[:, position]])
        )
        reconstruction = float(errors.max())
        if reconstruction > tolerance:
            raise ValueError(
                f"layer {layer_index}: recomputed router probabilities differ from the trace by "
                f"{reconstruction:.4g} (> {tolerance:.4g}); the router weight sidecar does not match this trace"
            )

        total = js_divergence(base_on_base, tuned_on_tuned)
        input_channel = js_divergence(base_on_base, base_on_tuned)
        parameter_channel = js_divergence(base_on_base, tuned_on_base)
        identical = bool(np.allclose(base_weight, tuned_weight))
        rows.append(
            {
                "layer_index": layer_index,
                "total_js_bits": float(total.mean()),
                "router_input_shift_js_bits": float(input_channel.mean()),
                "router_parameter_shift_js_bits": float(parameter_channel.mean()),
                "interaction_js_bits": float((total - input_channel - parameter_channel).mean()),
                "router_input_shift_topk_overlap": float(
                    topk_overlap(
                        np.argsort(-base_on_base, axis=-1)[:, :top_k], np.argsort(-base_on_tuned, axis=-1)[:, :top_k]
                    ).mean()
                ),
                "router_parameter_shift_topk_overlap": float(
                    topk_overlap(
                        np.argsort(-base_on_base, axis=-1)[:, :top_k], np.argsort(-tuned_on_base, axis=-1)[:, :top_k]
                    ).mean()
                ),
                "router_parameters_identical": identical,
                "routing_changed_with_frozen_router": bool(identical and float(total.mean()) > 0.0),
                "router_weight_max_abs_difference": float(np.abs(base_weight - tuned_weight).max()),
                "reconstruction_max_error": reconstruction,
                "reconstruction_mean_error": float(errors.mean()),
            }
        )
    frozen = all(row["router_parameters_identical"] for row in rows)
    return {
        "tokens": int(mask.sum()),
        "per_layer": rows,
        "router_frozen_everywhere": frozen,
        "channels_are_additive": False,
        "total_js_bits": float(np.mean([row["total_js_bits"] for row in rows])),
        "router_input_shift_js_bits": float(np.mean([row["router_input_shift_js_bits"] for row in rows])),
        "router_parameter_shift_js_bits": float(np.mean([row["router_parameter_shift_js_bits"] for row in rows])),
        "note": FROZEN_ROUTER_NOTE,
    }


def expert_output_change(
    base: AtlasTrace, tuned: AtlasTrace, *, slots: Sequence[int] | Sequence[str] | None = None
) -> list[dict[str, Any]]:
    """Change in the summed selected-expert output where both checkpoints selected the same experts.

    This is an observational quantity: even at matched selection the two
    checkpoints feed their experts different hidden states, so it mixes expert
    adaptation with upstream drift. ``expert_adaptation_from_blocks`` is the
    clean version and needs both checkpoints in memory.
    """
    check_comparable(base, tuned)
    require_comparable_routing(base, tuned, keys=("expert_set_id",))
    for trace, name in ((base, "base"), (tuned, "tuned")):
        if "expert_outputs" not in trace:
            raise ValueError(f"the {name} trace has no expert outputs; re-trace with --store-expert-outputs")
    slot_indices = resolve_slots(base, slots)
    mask = position_mask(base, tuned, slot_indices)
    base_sets = flatten_positions(base, "expert_set_id", slot_indices, mask)
    tuned_sets = flatten_positions(tuned, "expert_set_id", slot_indices, mask)
    base_outputs = flatten_positions(base, "expert_outputs", slot_indices, mask).astype(np.float32).sum(axis=-2)
    tuned_outputs = flatten_positions(tuned, "expert_outputs", slot_indices, mask).astype(np.float32).sum(axis=-2)
    rows = []
    for position, layer_index in enumerate(base.layers):
        matched = base_sets[:, position] == tuned_sets[:, position]
        if not matched.any():
            rows.append({"layer_index": layer_index, "matched_fraction": 0.0})
            continue
        rows.append(
            {
                "layer_index": layer_index,
                "matched_fraction": float(matched.mean()),
                "mean_cosine": mean_cosine(base_outputs[matched, position], tuned_outputs[matched, position]),
                "relative_l2": relative_l2(base_outputs[matched, position], tuned_outputs[matched, position]),
                "linear_cka": linear_cka(base_outputs[matched, position], tuned_outputs[matched, position]),
                "claim_limit": "matched selection only; still confounded by the upstream hidden-state shift",
            }
        )
    return rows


def _block_output(block, hidden: torch.Tensor, routing: Routing) -> torch.Tensor:
    return selected_expert_contributions(block, hidden.reshape(-1, hidden.shape[-1]), routing).sum(dim=1)


def _against(reference: np.ndarray, other: np.ndarray) -> dict[str, float]:
    return {
        "mean_cosine": mean_cosine(reference, other),
        "relative_l2": relative_l2(reference, other),
        "linear_cka": linear_cka(reference, other),
    }


def decompose_block_output_change(
    base_block,
    tuned_block,
    base_hidden: torch.Tensor,
    tuned_hidden: torch.Tensor,
    *,
    tuned_routing: Routing | None = None,
) -> dict[str, Any]:
    """Attribute one sparse block's output change to three separable causes.

    Each channel moves exactly one thing away from the base configuration and
    is measured against the base output:

    - ``expert_adaptation``: the tuned experts, on the base input, under the
      base selection;
    - ``hidden_state_shift``: the tuned input, through the base experts and the
      base router, which is the channel an upstream adapter opens;
    - ``selection_shift``: the observed tuned selection, applied to the base
      input and the base experts.

    The channels are not additive and are not reported as if they were: they
    share the nonlinearity of top-k selection, so ``observed_total`` is given
    separately rather than as their sum.

    A frozen router does not zero ``selection_shift``. The selection compared
    here is the one the tuned checkpoint actually produced, which upstream
    adaptation moves even when no router weight did.
    """
    if not (is_moe_block(base_block) and is_moe_block(tuned_block)):
        raise ValueError("the block-output decomposition needs two mixture-of-experts blocks")
    base_flat = base_hidden.reshape(-1, base_hidden.shape[-1])
    tuned_flat = tuned_hidden.reshape(-1, tuned_hidden.shape[-1])
    if base_flat.shape != tuned_flat.shape:
        raise ValueError(f"hidden states must be matched, got {tuple(base_flat.shape)} and {tuple(tuned_flat.shape)}")

    with torch.no_grad():
        base_routing = route(base_block, base_flat)
        observed_routing = tuned_routing or route(tuned_block, tuned_flat)
        reference = _block_output(base_block, base_flat, base_routing).float().cpu().numpy()
        channels = {
            "expert_adaptation": _block_output(tuned_block, base_flat, base_routing),
            "hidden_state_shift": _block_output(base_block, tuned_flat, route(base_block, tuned_flat)),
            "selection_shift": _block_output(base_block, base_flat, observed_routing),
        }
        observed = _block_output(tuned_block, tuned_flat, observed_routing).float().cpu().numpy()

    overlap = topk_overlap(base_routing.indices.cpu().numpy(), observed_routing.indices.cpu().numpy())
    router_weights_identical = _router_weights_identical(base_block, tuned_block)
    return {
        "tokens": int(reference.shape[0]),
        "top_k": moe_top_k(base_block),
        "norm_topk_prob": moe_norm_topk_prob(base_block),
        "observed_total": _against(reference, observed),
        **{name: _against(reference, value.float().cpu().numpy()) for name, value in channels.items()},
        "selection_topk_overlap": float(overlap.mean()),
        "router_weights_identical": router_weights_identical,
        "selection_changed_with_frozen_router": bool(router_weights_identical and float(overlap.mean()) < 1.0),
        "channels_are_additive": False,
        "note": FROZEN_ROUTER_NOTE,
    }


def _router_weights_identical(base_block, tuned_block) -> bool:
    base_weight = getattr(base_block.gate, "weight", None)
    tuned_weight = getattr(tuned_block.gate, "weight", None)
    if base_weight is None or tuned_weight is None:
        return False
    return bool(torch.equal(base_weight.detach().float().cpu(), tuned_weight.detach().float().cpu()))


def expert_adaptation_from_blocks(
    base_block, tuned_block, hidden_states: torch.Tensor, *, routing: Routing | None = None
) -> dict[str, Any]:
    """Direct expert adaptation: identical input, identical selection, two expert sets.

    Both blocks are driven by the base block's routing, so selection is held
    fixed by construction and the only free variable is the expert parameters.
    """
    if not (is_moe_block(base_block) and is_moe_block(tuned_block)):
        raise ValueError("expert adaptation needs two mixture-of-experts blocks")
    flat = hidden_states.reshape(-1, hidden_states.shape[-1])
    with torch.no_grad():
        routing = routing or route(base_block, flat)
        base_output = selected_expert_contributions(base_block, flat, routing).sum(dim=1)
        tuned_output = selected_expert_contributions(tuned_block, flat, routing).sum(dim=1)
    first = base_output.float().cpu().numpy()
    second = tuned_output.float().cpu().numpy()
    return {
        "tokens": int(first.shape[0]),
        "top_k": moe_top_k(base_block),
        "norm_topk_prob": moe_norm_topk_prob(base_block),
        **_against(first, second),
        "selection_held_fixed": True,
    }


def expert_adaptation_from_checkpoints(
    base_model_name: str,
    tuned_checkpoint: Path | str,
    hidden_states: torch.Tensor,
    *,
    layer_index: int,
    revision: str = "",
    device_map: str = "",
) -> dict[str, Any]:
    """Load both checkpoints and isolate expert adaptation at one layer.

    Raises ``AdapterLoadingUnsupported`` when the tuned checkpoint cannot be
    turned into a live model, which is the case for a raw DeepSpeed ZeRO
    directory.
    """
    describe_checkpoint(tuned_checkpoint).require_loadable()
    base_model, _ = load_traced_model(base_model_name, revision=revision, device_map=device_map)
    tuned_model, _ = load_traced_model(
        base_model_name, revision=revision, adapter=str(tuned_checkpoint), device_map=device_map
    )
    base_block = decoder_layers(base_model)[layer_index].mlp
    tuned_block = decoder_layers(tuned_model)[layer_index].mlp
    return {"layer_index": layer_index, **expert_adaptation_from_blocks(base_block, tuned_block, hidden_states)}


# --------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------


def compare_traces(
    base: AtlasTrace,
    tuned: AtlasTrace,
    *,
    slots: Sequence[str] | None = None,
    base_router: dict[int, np.ndarray] | None = None,
    tuned_router: dict[int, np.ndarray] | None = None,
    selectivity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The whole delta report, with each unavailable section explaining itself."""
    check_comparable(base, tuned)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "records": base.n_records,
        "slots": list(slots) if slots else base.roles,
        "base_metadata": base.metadata,
        "tuned_metadata": tuned.metadata,
        "texts_identical": True,
        "activation": activation_delta(base, tuned, slots=slots),
    }
    for name, function in (
        ("routing", lambda: routing_delta(base, tuned, slots=slots)),
        ("expert_output_change", lambda: expert_output_change(base, tuned, slots=slots)),
    ):
        try:
            report[name] = function()
        except ValueError as exc:
            report[name] = {"unavailable": str(exc)}
    if base_router and tuned_router:
        try:
            report["routing_decomposition"] = decompose_routing_change(
                base, tuned, base_router, tuned_router, slots=slots
            )
        except ValueError as exc:
            report["routing_decomposition"] = {"unavailable": str(exc)}
    else:
        report["routing_decomposition"] = {
            "unavailable": "pass --base-router-weights and --tuned-router-weights from the traced runs",
            "note": FROZEN_ROUTER_NOTE,
        }
    report["block_output_decomposition"] = {
        "unavailable": (
            "separating direct expert adaptation from the hidden-state and selection channels needs both "
            "checkpoints resident in memory; call decompose_block_output_change with two loaded blocks"
        )
    }
    if selectivity:
        report["selectivity_change"] = selectivity
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--describe", type=Path, nargs="*", default=[], help="classify checkpoints and exit")
    parser.add_argument(
        "--require-loadable", action="store_true", help="with --describe, exit non-zero if any cannot be loaded"
    )
    parser.add_argument("--base-trace", type=Path)
    parser.add_argument("--tuned-trace", type=Path)
    parser.add_argument("--base-router-weights", type=Path)
    parser.add_argument("--tuned-router-weights", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--slots", default="", help="comma-separated position roles; empty uses every role")
    parser.add_argument("--selectivity-site", default="", choices=("", *SITES))
    parser.add_argument("--selectivity-role", default="content_last")
    parser.add_argument("--selectivity-layer", type=int, default=-1)
    parser.add_argument("--rows", type=Path, default=None, help="jsonl with response text for the lexical control")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1701)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.describe:
        descriptions = [describe_checkpoint(path) for path in args.describe]
        for description in descriptions:
            print(json.dumps(description.as_dict(), indent=2))
        refused = [description for description in descriptions if not description.loadable]
        if refused and args.require_loadable:
            # A caller that is about to spend a GPU asked to be stopped here.
            raise SystemExit(
                f"{len(refused)} of {len(descriptions)} checkpoints cannot be loaded: "
                + "; ".join(f"{description.path}: {description.reason}" for description in refused)
            )
        return
    if not (args.base_trace and args.tuned_trace and args.out):
        raise SystemExit("--base-trace, --tuned-trace and --out are required unless --describe is used")

    base = load_atlas_trace(args.base_trace)
    tuned = load_atlas_trace(args.tuned_trace)
    slots = [value.strip() for value in args.slots.split(",") if value.strip()] or None
    selectivity = None
    if args.selectivity_site:
        from projects.learning_science_semantic_atlas.analyze_representations import read_texts  # noqa: PLC0415

        layer_index = base.layers[-1] if args.selectivity_layer < 0 else args.selectivity_layer
        selectivity = selectivity_change(
            base,
            tuned,
            site=args.selectivity_site,
            layer_index=layer_index,
            role=args.selectivity_role,
            texts=read_texts(args.rows, base["ids"]),
            n_splits=args.folds,
            seed=args.seed,
        )
    report = compare_traces(
        base,
        tuned,
        slots=slots,
        base_router=load_router_weights(args.base_router_weights) if args.base_router_weights else None,
        tuned_router=load_router_weights(args.tuned_router_weights) if args.tuned_router_weights else None,
        selectivity=selectivity,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=float) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
