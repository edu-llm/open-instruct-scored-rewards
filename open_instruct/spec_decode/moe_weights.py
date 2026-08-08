"""Unfuse transformers-5 OLMoE expert tensors into the per-expert layout vLLM 0.21 expects.

NOT ABOUT SPECULATIVE DECODING. It lives beside that code only because both are delivered through
the same ``ModelRegistry`` override, and it is needed by any OLMoE GRPO run whether or not a draft
model exists.

THE BUG THIS FIXES. transformers 5.x packs a layer's experts into two tensors::

    model.layers.L.mlp.experts.gate_up_proj   [num_experts, 2 * intermediate, hidden]
    model.layers.L.mlp.experts.down_proj      [num_experts, hidden, intermediate]

vLLM 0.21's OLMoE expects the pre-refactor per-expert layout, which is what its
``fused_moe_make_expert_params_mapping(ckpt_gate_proj_name="gate_proj", ...)`` matches::

    ...mlp.experts.<i>.gate_proj.weight   [intermediate, hidden]
    ...mlp.experts.<i>.up_proj.weight     [intermediate, hidden]
    ...mlp.experts.<i>.down_proj.weight   [hidden, intermediate]

A fused name matches none of those mappings, falls through to a bare ``params_dict[name]`` lookup
and raises ``KeyError: 'layers.0.mlp.experts.gate_up_proj'``. This is invisible when loading from
the Hub -- that checkpoint was *saved* with the old per-expert names -- and appears only when a live
transformers-5 learner broadcasts its in-memory parameters, i.e. on the first GRPO weight sync.
Measured on run_019fe337 (2026-08-08).

WHY THE SPLIT ORDER IS NOT A GUESS. Getting it wrong would load plausible-looking garbage rather
than fail, so it is taken from the module that owns the tensor. ``OlmoeExperts.forward`` does::

    gate, up = nn.functional.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)

``F.linear(x, W)`` is ``x @ W.T`` with ``W`` of shape ``[2 * intermediate, hidden]``, so the output
is ``[..., 2 * intermediate]`` and ``chunk(2, -1)`` takes the first ``intermediate`` **rows** of the
fused tensor as gate and the second as up. Both halves are therefore already in ``nn.Linear``'s
``(out_features, in_features)`` convention, and ``down_proj[e]`` at ``[hidden, intermediate]``
already is too. So the transform is pure slicing: no transpose, no reshape.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import torch

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)

#: Suffixes of the fused tensors, with no expert index. A per-expert name ends in
#: ``experts.<i>.down_proj.weight``, so matching the exact suffix cannot confuse the two -- which
#: is what lets this run unconditionally, including on the Hub checkpoint that needs no transform.
FUSED_GATE_UP_SUFFIX = ".mlp.experts.gate_up_proj"
FUSED_DOWN_SUFFIX = ".mlp.experts.down_proj"


def unfuse_olmoe_experts(weights: Iterable[tuple[str, torch.Tensor]]) -> Iterator[tuple[str, torch.Tensor]]:
    """Pass weights through, expanding fused expert tensors into per-expert ones.

    A no-op for any stream that is already per-expert, so it is safe to apply to every load: the
    initial load from a Hub checkpoint saved by transformers 4.x goes through untouched.

    :raises ValueError: if a fused tensor's shape cannot be split evenly into gate and up halves.
        Raised rather than guessed, because a silent mis-split loads weights that are wrong but
        not obviously wrong.
    """
    for name, tensor in weights:
        if name.endswith(FUSED_GATE_UP_SUFFIX):
            yield from _split_gate_up(name, tensor)
        elif name.endswith(FUSED_DOWN_SUFFIX):
            yield from _split_down(name, tensor)
        else:
            yield name, tensor


def _prefix(name: str, suffix: str) -> str:
    """``model.layers.0.mlp.experts.gate_up_proj`` -> ``model.layers.0.mlp.experts``."""
    return name[: -len(suffix)] + ".mlp.experts"


def _split_gate_up(name: str, tensor: torch.Tensor) -> Iterator[tuple[str, torch.Tensor]]:
    if tensor.ndim != 3:
        raise ValueError(f"{name}: expected a 3D fused expert tensor, got shape {tuple(tensor.shape)}")
    num_experts, two_intermediate, _hidden = tensor.shape
    if two_intermediate % 2 != 0:
        raise ValueError(
            f"{name}: fused dimension {two_intermediate} is odd, so it cannot be a gate/up "
            "concatenation. Refusing to guess a split."
        )
    intermediate = two_intermediate // 2
    prefix = _prefix(name, FUSED_GATE_UP_SUFFIX)
    for expert in range(num_experts):
        # Rows, not columns: see the module docstring on why this follows from OlmoeExperts.forward.
        yield f"{prefix}.{expert}.gate_proj.weight", tensor[expert, :intermediate, :]
        yield f"{prefix}.{expert}.up_proj.weight", tensor[expert, intermediate:, :]


def _split_down(name: str, tensor: torch.Tensor) -> Iterator[tuple[str, torch.Tensor]]:
    if tensor.ndim != 3:
        raise ValueError(f"{name}: expected a 3D fused expert tensor, got shape {tuple(tensor.shape)}")
    num_experts = tensor.shape[0]
    prefix = _prefix(name, FUSED_DOWN_SUFFIX)
    for expert in range(num_experts):
        # Already [hidden, intermediate], which is nn.Linear's (out, in). No transpose.
        yield f"{prefix}.{expert}.down_proj.weight", tensor[expert]


def stream_is_fused(names: Iterable[str]) -> bool:
    """True if any name is a fused expert tensor. For logging and tests, not for control flow."""
    return any(n.endswith(FUSED_GATE_UP_SUFFIX) or n.endswith(FUSED_DOWN_SUFFIX) for n in names)
