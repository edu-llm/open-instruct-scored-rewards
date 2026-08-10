"""Loading gpt-oss (and OLMoE) for LoRA fine-tuning, with the routers left alone.

Two things about gpt-oss make this more than ``from_pretrained``:

**It ships in MXFP4 and cannot be trained in it.** The released weights are 4-bit, which is why
the 20B runs inference in ~14GB, but no training stack does a backward pass through MXFP4.
``Mxfp4Config(dequantize=True)`` upcasts to bf16 at load, which is the supported path and takes
the 20B to about 42GB. That number is the reason this project targets 80GB cards: it fits on
one with room for activations, and does not fit on the 40GB A100s at all.

**Its experts are one fused 3-D tensor, not a ModuleList.** ``target_modules`` adapts a module
by replacing its ``forward``, and an ``nn.Parameter`` does not have one, so the feed-forward
half of a MoE is invisible to ordinary LoRA -- a run configured that way trains attention only
and quietly leaves 19 of the 20 billion parameters untouched. PEFT's ``target_parameters``
reaches them and treats dim 0 as the expert dimension, giving each expert its own A and B.

The router is deliberately NOT targeted. Adapting it changes which experts fire rather than
what they compute, which is a different and much less predictable intervention than the one
this project is testing. :func:`assert_routers_frozen` turns that intention into a check,
because the failure is silent: a mis-specified target list trains fine and produces a model
whose routing has drifted.
"""

from __future__ import annotations

import torch

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)

# Attention is a plain Linear on both architectures and is reached by name.
ATTENTION_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# The fused expert tensors, per architecture. PEFT matches these as suffixes. All three
# architectures happen to agree on the names -- transformers v5 stores experts as
# (num_experts, ...) Parameters throughout -- but they are listed per type rather than shared,
# because a model whose names differ must fail the lookup loudly instead of silently adapting
# nothing.
EXPERT_PARAMETERS = {
    "gpt_oss": ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"],
    "olmoe": ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"],
    "qwen3_moe": ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"],
}

# What a trainable router parameter looks like, per architecture. gpt-oss calls its router
# ``mlp.router``; OLMoE and Qwen3-MoE both call the same thing ``mlp.gate``. Getting this wrong
# makes the freeze check vacuous rather than noisy, so it is keyed by model_type not guessed.
ROUTER_MARKERS = {
    "gpt_oss": ("mlp.router.",),
    "olmoe": ("mlp.gate.",),
    "qwen3_moe": ("mlp.gate.",),
}

MOE_TYPES = frozenset(EXPERT_PARAMETERS)


def model_type(model_or_config) -> str:
    config = getattr(model_or_config, "config", model_or_config)
    return getattr(config, "model_type", "")


def is_moe(model_or_config) -> bool:
    return model_type(model_or_config) in MOE_TYPES


def expert_parameters(model_or_config) -> list[str]:
    return list(EXPERT_PARAMETERS.get(model_type(model_or_config), []))


def default_expert_rank(config, lora_r: int) -> int:
    """PEFT's guidance, following "LoRA Without Regret": ``lora_r // num_experts``, at least 1.

    There is one LoRA pair per expert, so rank ``r`` across 32 experts is 32x the parameter
    budget of rank ``r`` on one dense layer. Scaling the rank down by the expert count is what
    keeps the attention and feed-forward budgets comparable.
    """
    experts = getattr(config, "num_local_experts", None) or getattr(config, "num_experts", 1)
    return max(1, lora_r // max(1, experts))


def load_base_model(model_name: str, *, dtype=torch.bfloat16, attn_implementation=None,
                    device_map=None, gradient_checkpointing=False):
    """The frozen base, dequantised where it needs to be."""
    from transformers import AutoModelForCausalLM

    kwargs = {"dtype": dtype, "use_cache": False}

    if "gpt-oss" in model_name.lower() or model_type_of_pretrained(model_name) == "gpt_oss":
        from transformers import Mxfp4Config

        kwargs["quantization_config"] = Mxfp4Config(dequantize=True)
        # gpt-oss ships attention sinks that FlashAttention-2 does not implement; the supported
        # choices are eager and the vllm-flash-attn3 kernel. eager is the one that is always
        # present, so it is the default rather than a fallback discovered at step 0.
        kwargs["attn_implementation"] = attn_implementation or "eager"
    elif attn_implementation:
        kwargs["attn_implementation"] = attn_implementation

    if device_map:
        kwargs["device_map"] = device_map

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    return model


def model_type_of_pretrained(model_name: str) -> str:
    """``model_type`` from the hub config without downloading weights."""
    from transformers import AutoConfig

    try:
        return getattr(AutoConfig.from_pretrained(model_name), "model_type", "")
    except Exception:  # offline, or a local path that is not a hub id
        return ""


def build_lora_config(model, model_config):
    """Delegate to the trainer's LoRA builder, filling in the MoE fields for this architecture.

    ``model_config`` is an ``open_instruct.model_utils.ModelConfig``. Its ``lora_target_parameters``
    /``lora_expert_rank``/``lora_expert_alpha`` fields already exist for exactly this case, so
    the only thing added here is architecture-specific defaults for anything left unset.
    """
    # Imported here, not at module scope: grpo_fast pulls in Ray and vLLM, which is a slow and
    # occasionally hostile import for a process that only wants twenty lines of config.
    from open_instruct.grpo_fast import _build_lora_config

    if is_moe(model):
        if not model_config.lora_target_parameters:
            model_config.lora_target_parameters = expert_parameters(model)
        if model_config.lora_expert_rank is None:
            model_config.lora_expert_rank = default_expert_rank(model.config, model_config.lora_r)
        if model_config.lora_expert_alpha is None:
            # Keep the experts' alpha/r scaling equal to attention's rather than inheriting the
            # global alpha, which at a smaller expert rank would scale them up, not down.
            ratio = model_config.lora_alpha / max(1, model_config.lora_r)
            model_config.lora_expert_alpha = max(1, int(round(ratio * model_config.lora_expert_rank)))
        if model_config.lora_dropout:
            # PEFT adapts a bare Parameter with ParamWrapper, which has nowhere to put a dropout
            # mask. PEFT raises on this itself; saying so here names the field to change.
            raise ValueError(
                f"MoE LoRA needs lora_dropout=0, got {model_config.lora_dropout}. PEFT's ParamWrapper "
                "adapts the fused expert tensors and cannot apply dropout."
            )
        if not model_config.lora_target_modules:
            model_config.lora_target_modules = list(ATTENTION_MODULES)

    return _build_lora_config(model_config)


def assert_routers_frozen(model) -> int:
    """Raise if any routing parameter is trainable. Returns how many were checked.

    A zero count is itself a failure: it means the marker for this architecture is wrong and the
    check has been passing vacuously.
    """
    markers = ROUTER_MARKERS.get(model_type(model), ())
    if not markers:
        return 0
    named = dict(model.named_parameters())
    routers = [n for n in named if any(m in n for m in markers)]
    if not routers:
        raise RuntimeError(
            f"no parameters matched the router markers {markers} for model_type "
            f"{model_type(model)!r} -- the freeze check would pass vacuously"
        )
    hot = [n for n in routers if named[n].requires_grad]
    if hot:
        raise RuntimeError(f"{len(hot)} router parameter(s) are trainable, e.g. {hot[0]}")
    return len(routers)


def wrap_lora(model, model_config):
    """LoRA-wrap ``model`` and verify the routers stayed frozen."""
    from peft import get_peft_model

    moe = is_moe(model)
    model = get_peft_model(model, build_lora_config(model, model_config))
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    if moe:
        n = assert_routers_frozen(model)
        logger.info(
            "MoE LoRA: attention r=%s, per-expert r=%s (alpha %s); %d router tensors frozen",
            model_config.lora_r, model_config.lora_expert_rank, model_config.lora_expert_alpha, n,
        )
    return model
