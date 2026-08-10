"""Version-tolerant model helpers for dense OLMo and OLMoE.

Transformers 4.x represents OLMoE experts as a ``ModuleList``. Transformers
5.4 returns ``(probabilities, weights, indices)`` from the gate, while 5.14
returns ``(logits, weights, indices)``. These helpers detect both contracts and
recompute true logits from the router input whenever its weight is available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class Routing:
    logits: torch.Tensor
    weights: torch.Tensor
    indices: torch.Tensor


def decoder_layers(model) -> list:
    base = getattr(model, "model", model)
    layers = getattr(base, "layers", None)
    if layers is None:
        raise TypeError(f"cannot find decoder layers on {type(model).__name__}")
    return list(layers)


def tensor_from_output(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(f"expected tensor or tensor-first tuple, got {type(output).__name__}")


def replace_output_tensor(output: Any, value: torch.Tensor) -> Any:
    if torch.is_tensor(output):
        return value
    if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
        return (value, *output[1:])
    raise TypeError(f"expected tensor or tensor-first tuple, got {type(output).__name__}")


def is_moe_block(module) -> bool:
    return (
        hasattr(module, "gate")
        and hasattr(module, "experts")
        and (hasattr(module, "top_k") or hasattr(module.gate, "top_k"))
    )


def moe_top_k(block) -> int:
    if hasattr(block, "top_k"):
        return int(block.top_k)
    return int(block.gate.top_k)


def moe_norm_topk_prob(block) -> bool:
    return bool(getattr(block, "norm_topk_prob", getattr(block.gate, "norm_topk_prob", False)))


def looks_like_probabilities(value: torch.Tensor, atol: float = 1e-4) -> bool:
    if value.ndim != 2 or not torch.isfinite(value).all():
        return False
    float_value = value.float()
    return bool(
        float_value.min() >= -atol
        and float_value.max() <= 1.0 + atol
        and torch.allclose(float_value.sum(dim=-1), torch.ones(value.shape[0], device=value.device), atol=atol)
    )


def normalize_gate_output(block, gate_output: Any) -> Routing:
    """Normalize Transformers 4.x logits and both Transformers 5.x tuple contracts."""
    if isinstance(gate_output, tuple) and len(gate_output) >= 3:
        first, weights, indices = gate_output[:3]
        logits = first.float().clamp_min(torch.finfo(torch.float32).tiny).log() if looks_like_probabilities(first) else first
        return Routing(logits=logits, weights=weights, indices=indices)
    if not torch.is_tensor(gate_output):
        raise TypeError(f"unsupported gate output {type(gate_output).__name__}")
    logits = gate_output
    probabilities = F.softmax(logits, dim=-1, dtype=torch.float)
    weights, indices = torch.topk(probabilities, moe_top_k(block), dim=-1)
    if moe_norm_topk_prob(block):
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)
    return Routing(logits=logits, weights=weights.to(logits.dtype), indices=indices)


def routing_from_gate_output(block, hidden_states: torch.Tensor, gate_output: Any) -> Routing:
    routing = normalize_gate_output(block, gate_output)
    router = block.gate
    weight = getattr(router, "weight", None)
    if weight is not None:
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        logits = F.linear(flat.float(), weight.float())
        return Routing(logits=logits, weights=routing.weights, indices=routing.indices)
    return routing


def route(block, hidden_states: torch.Tensor) -> Routing:
    flat = hidden_states.reshape(-1, hidden_states.shape[-1])
    return routing_from_gate_output(block, flat, block.gate(flat))


def renormalize_topk(logits: torch.Tensor, top_k: int, norm_topk_prob: bool = True) -> Routing:
    probabilities = F.softmax(logits, dim=-1, dtype=torch.float)
    weights, indices = torch.topk(probabilities, top_k, dim=-1)
    if norm_topk_prob:
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)
    return Routing(logits=logits, weights=weights.to(logits.dtype), indices=indices)


def selected_expert_contributions(
    block,
    hidden_states: torch.Tensor,
    routing: Routing | None = None,
) -> torch.Tensor:
    """Weighted output of every selected expert, before their sum.

    Args:
        block: an OLMoE sparse block.
        hidden_states: ``[tokens, hidden]`` or ``[batch, seq, hidden]`` block input.
        routing: optional routing for the same flattened states.

    Returns:
        Tensor ``[tokens, top_k, hidden]``. Summing axis 1 reproduces the sparse
        block output up to normal floating-point accumulation differences.
    """
    flat = hidden_states.reshape(-1, hidden_states.shape[-1])
    routing = routing or route(block, flat)
    tokens, top_k = routing.indices.shape
    out = torch.zeros(tokens, top_k, flat.shape[-1], dtype=flat.dtype, device=flat.device)
    experts = block.experts

    if isinstance(experts, torch.nn.ModuleList):
        for expert_idx, expert in enumerate(experts):
            token_idx, topk_idx = torch.where(routing.indices == expert_idx)
            if token_idx.numel() == 0:
                continue
            value = expert(flat[token_idx])
            out[token_idx, topk_idx] = value * routing.weights[token_idx, topk_idx, None].to(value.dtype)
        return out

    gate_up = getattr(experts, "gate_up_proj", None)
    down = getattr(experts, "down_proj", None)
    activation = getattr(experts, "act_fn", None)
    if gate_up is None or down is None or activation is None:
        raise TypeError(f"unsupported fused experts layout on {type(experts).__name__}")
    for expert_idx in torch.unique(routing.indices).tolist():
        if int(expert_idx) >= int(getattr(experts, "num_experts", gate_up.shape[0])):
            continue
        token_idx, topk_idx = torch.where(routing.indices == expert_idx)
        current = flat[token_idx]
        gate, up = F.linear(current, gate_up[expert_idx]).chunk(2, dim=-1)
        value = F.linear(activation(gate) * up, down[expert_idx])
        out[token_idx, topk_idx] = value * routing.weights[token_idx, topk_idx, None].to(value.dtype)
    return out


def verify_moe_decomposition(
    block,
    hidden_states: torch.Tensor,
    *,
    atol: float = 2e-3,
    rtol: float = 2e-3,
) -> float:
    """Return max error and raise when selected contributions do not reconstruct."""
    block_input = hidden_states.unsqueeze(0) if hidden_states.ndim == 2 else hidden_states
    with torch.no_grad():
        actual = tensor_from_output(block(block_input))
        routing = route(block, block_input)
        reconstructed = selected_expert_contributions(block, block_input, routing).sum(dim=1)
        reconstructed = reconstructed.reshape_as(actual)
    error = float((actual.float() - reconstructed.float()).abs().max().item())
    if not torch.allclose(actual.float(), reconstructed.float(), atol=atol, rtol=rtol):
        raise AssertionError(f"expert decomposition mismatch: max_abs_error={error:.6g}")
    return error
