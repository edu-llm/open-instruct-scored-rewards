"""Does splitting a fused MoE parameter give back the per-expert weights vLLM expects?

The sync path rewrites transformers 5's stacked expert parameter into the per-expert checkpoint
names vLLM's FusedMoE loader understands. Getting the layout wrong there does not raise: it sends
tensors of the right shape holding the wrong numbers, so vLLM generates from a scrambled model
while the trainer's own loss looks perfectly healthy. That failure is invisible in every metric we
log, which is why it is worth a test that checks values rather than shapes.

So this drives the real OlmoeExperts forward and asks whether three plain nn.Linear layers built
from the split reproduce it. If the gate/up halves were swapped, or either needed a transpose, the
numbers diverge and these fail.
"""

import ast
import pathlib

import pytest
import torch

from open_instruct.vllm_utils import _params_to_send, _peft_target_specs, _split_fused_experts


def experts_and_config(num_experts: int = 3, hidden: int = 8, inter: int = 4):
    """A small OlmoeExperts with reproducible non-trivial weights."""
    olmoe = pytest.importorskip("transformers.models.olmoe.modeling_olmoe")
    config = pytest.importorskip("transformers.models.olmoe.configuration_olmoe")
    if not hasattr(olmoe, "OlmoeExperts"):
        pytest.skip("transformers<5 keeps experts as modules, so there is nothing fused to split")
    cfg = config.OlmoeConfig(
        hidden_size=hidden,
        intermediate_size=inter,
        num_local_experts=num_experts,
        num_experts_per_tok=1,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    torch.manual_seed(0)
    experts = olmoe.OlmoeExperts(cfg)
    with torch.no_grad():
        experts.gate_up_proj.normal_(0, 0.5)
        experts.down_proj.normal_(0, 0.5)
    return experts, cfg


def split_of(experts, name: str) -> dict[str, torch.Tensor]:
    """The per-expert tensors the sync would send for one fused parameter."""
    param = getattr(experts, name)
    pieces = _split_fused_experts(f"model.layers.0.mlp.experts.{name}", param.shape)
    assert pieces is not None, f"{name} should be recognised as a fused expert parameter"
    return {n: param[i] for n, _, i in pieces}


def test_split_reproduces_the_expert_forward():
    """Per-expert linears built from the split must match OlmoeExperts on the same tokens."""
    experts, cfg = experts_and_config()
    sent = split_of(experts, "gate_up_proj") | split_of(experts, "down_proj")
    tokens = torch.randn(5, cfg.hidden_size)

    for e in range(cfg.num_local_experts):
        # Route every token to this one expert with weight 1, so the output is that expert alone.
        index = torch.full((5, 1), e)
        weights = torch.ones(5, 1)
        expected = experts(tokens, index, weights)

        stem = f"model.layers.0.mlp.experts.{e}"
        gate = torch.nn.functional.linear(tokens, sent[f"{stem}.gate_proj.weight"])
        up = torch.nn.functional.linear(tokens, sent[f"{stem}.up_proj.weight"])
        got = torch.nn.functional.linear(experts.act_fn(gate) * up, sent[f"{stem}.down_proj.weight"])

        torch.testing.assert_close(got, expected, msg=f"expert {e} does not match after the split")


def test_gate_and_up_are_not_interchangeable():
    """Guards the test above: swapping the halves has to actually break it.

    gate and up have identical shapes, so a swap is invisible to any shape check and survives a
    round trip. If SiLU-on-gate times up happened to equal SiLU-on-up times gate here, the test
    above would pass for a broken split, so confirm the check has teeth.
    """
    experts, cfg = experts_and_config()
    sent = split_of(experts, "gate_up_proj") | split_of(experts, "down_proj")
    tokens = torch.randn(5, cfg.hidden_size)
    stem = "model.layers.0.mlp.experts.0"

    gate = torch.nn.functional.linear(tokens, sent[f"{stem}.gate_proj.weight"])
    up = torch.nn.functional.linear(tokens, sent[f"{stem}.up_proj.weight"])
    right = experts.act_fn(gate) * up
    swapped = experts.act_fn(up) * gate
    assert not torch.allclose(right, swapped, atol=1e-4), "swapping gate and up changed nothing"


def test_every_expert_weight_is_sent_exactly_once():
    """A silent drop is as bad as a wrong layout, and would leave vLLM on stale expert weights."""
    experts, cfg = experts_and_config()
    names = list(split_of(experts, "gate_up_proj")) + list(split_of(experts, "down_proj"))

    assert len(names) == len(set(names)), "an expert weight is sent twice"
    assert len(names) == 3 * cfg.num_local_experts, "expected gate, up and down for every expert"
    for e in range(cfg.num_local_experts):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            assert f"model.layers.0.mlp.experts.{e}.{proj}.weight" in names


def test_send_splits_experts_and_ipc_avoids_copying():
    """The shared send path must split experts, and must not copy when IPC asked it not to."""
    experts, cfg = experts_and_config()
    params = [
        ("model.layers.0.mlp.experts.gate_up_proj", experts.gate_up_proj),
        ("model.layers.0.mlp.experts.down_proj", experts.down_proj),
        ("model.layers.0.self_attn.q_proj.weight", torch.nn.Parameter(torch.randn(8, 8))),
    ]

    for clone in (True, False):
        sent = dict(_params_to_send(params, None, clone=clone))
        assert len(sent) == 3 * cfg.num_local_experts + 1, f"wrong count with clone={clone}"
        assert "model.layers.0.mlp.experts.gate_up_proj" not in sent, "fused name still sent"
        assert "model.layers.0.self_attn.q_proj.weight" in sent, "attention weight dropped"
        got = sent["model.layers.0.mlp.experts.1.down_proj.weight"]
        torch.testing.assert_close(got, experts.down_proj[1])
        shares = got.untyped_storage().data_ptr() == experts.down_proj.untyped_storage().data_ptr()
        assert shares is not clone, f"clone={clone} should {'copy' if clone else 'alias'} storage"


def test_qwen3_moe_sends_every_expert_weight_in_vllm_layout():
    """Qwen3 expert weights reach vLLM under its per-expert checkpoint names."""
    modeling = pytest.importorskip("transformers.models.qwen3_moe.modeling_qwen3_moe")
    configuration = pytest.importorskip("transformers.models.qwen3_moe.configuration_qwen3_moe")
    cfg = configuration.Qwen3MoeConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=24,
        moe_intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        num_experts=3,
        num_experts_per_tok=1,
    )
    model = modeling.Qwen3MoeForCausalLM(cfg)
    experts = [(name, param) for name, param in model.named_parameters() if ".mlp.experts." in name]
    sent = dict(_params_to_send(experts, None, clone=True))

    fused_gate = next(((name, param) for name, param in experts if name.endswith(".gate_up_proj")), None)
    fused_down = next(((name, param) for name, param in experts if name.endswith(".down_proj")), None)
    if fused_gate and fused_down:
        stem = fused_gate[0].removesuffix(".experts.gate_up_proj")
        gate_up, down = fused_gate[1], fused_down[1]
        intermediate = gate_up.shape[1] // 2
        expected = {}
        for expert in range(cfg.num_experts):
            expected[f"{stem}.experts.{expert}.gate_proj.weight"] = gate_up[expert, :intermediate]
            expected[f"{stem}.experts.{expert}.up_proj.weight"] = gate_up[expert, intermediate:]
            expected[f"{stem}.experts.{expert}.down_proj.weight"] = down[expert]
    else:
        expected = dict(experts)

    assert set(sent) == set(expected)
    assert len(sent) == 3 * cfg.num_experts
    for name, parameter in expected.items():
        torch.testing.assert_close(sent[name], parameter)
        assert sent[name].untyped_storage().data_ptr() != parameter.untyped_storage().data_ptr()


def test_zero3_lora_sync_selects_only_reconstructed_base_weights():
    class TinyPeftModel(torch.nn.Module):
        def active_adapters(self):
            return ["default"]

    class TinyLora(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base_layer = torch.nn.Linear(4, 3, bias=False)
            self.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(4, 2, bias=False)})
            self.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(2, 3, bias=False)})
            self.active_adapters = ["default"]
            self.lora_bias = {"default": False}

    model = TinyPeftModel()
    model.q_proj = TinyLora()
    model.untouched = torch.nn.Linear(4, 4, bias=False)
    specs = _peft_target_specs(model, lambda name: name.replace(".base_layer.", "."))

    assert [(name, adapter) for name, _, adapter in specs] == [("q_proj.weight", "default")]
    assert specs[0][1] is model.q_proj


def test_every_sender_goes_through_the_shared_path():
    """No sender may map names itself, which is the bug this file exists to prevent recurring.

    Three places walked named_parameters() and applied the mapper inline: the NCCL send, the IPC
    send, and the metadata pass. Teaching two of them to split expert weights fixed nothing,
    because the third - IPC, used whenever the trainer and vLLM share a GPU - kept sending the
    fused name. The failure was byte-identical before and after, which is what made it costly.

    So this asserts the structural property rather than the behaviour: name_mapper is called in
    exactly the two functions allowed to interpret it. A fourth sender that rolls its own fails here.
    """
    source = pathlib.Path(__file__).with_name("vllm_utils.py").read_text()
    callers = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.FunctionDef):
            continue
        for inner in ast.walk(node):
            called = isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name)
            if called and inner.func.id == "name_mapper":
                callers.add(node.name)

    assert callers == {"_params_to_send", "_collect_weight_metadata", "_peft_target_specs"}, (
        f"name_mapper is applied in {sorted(callers)}; a sender outside the shared path will not "
        "split fused expert weights and will fail the vLLM sync"
    )


def test_ordinary_parameters_are_left_alone():
    """Only the stacked expert parameters are rewritten; everything else must pass through."""
    for name, shape in (
        ("model.layers.0.self_attn.q_proj.weight", (8, 8)),
        ("model.layers.0.mlp.gate.weight", (3, 8)),
        # Right name, wrong rank: a 2D tensor is already per-expert and must not be sliced.
        ("model.layers.0.mlp.experts.down_proj", (8, 4)),
    ):
        assert _split_fused_experts(name, shape) is None, f"{name} should not be split"
