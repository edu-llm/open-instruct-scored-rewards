"""CPU tests for consolidating a DeepSpeed ZeRO state into a loadable adapter.

Nothing here downloads a model or touches a GPU. A toy policy shaped like OLMoE
- attention projections plus one module holding two fused 3-D expert tensors -
is wrapped by the real PEFT, and its trainable parameters are written into a
real DeepSpeed ZeRO-2 layout, so the export is read back by deepspeed's own
``zero_to_fp32`` rather than by a stand-in for it. The keys and tensors the
export produces are then compared against what PEFT itself would have saved,
which is the only comparison that pins the naming.
"""

from __future__ import annotations

import json
import zipfile
from collections import OrderedDict
from pathlib import Path

import pytest
import torch
from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict
from projects.learning_science_semantic_atlas import checkpoint_delta as delta
from projects.learning_science_semantic_atlas import export_checkpoint as export
from safetensors.torch import load_file
from torch import nn

BASE_MODEL = "allenai/OLMoE-1B-7B-0924-Instruct"
EXPERT_TARGETS = ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]
EXPERTS = 5
ATTENTION_RANK = 8
EXPERT_RANK = 2
# A fused expert adapter stacks one LoRA per expert, so this is what its lora_A
# is wide, and it is not the rank. The OLMoE run's is 64 x 2 = 128.
FUSED_WIDTH = EXPERTS * EXPERT_RANK


# --------------------------------------------------------------------------
# a toy policy and a real ZeRO-2 checkpoint of it
# --------------------------------------------------------------------------


class ToyExperts(nn.Module):
    """One module holding two fused per-expert tensors, as OLMoE stores them.

    Five experts rather than a power of two, so ``experts * r`` cannot be
    mistaken for the attention rank in any of the assertions below.
    """

    def __init__(self, experts: int = EXPERTS, hidden: int = 8, ffn: int = 6):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.randn(experts, hidden, 2 * ffn))
        self.down_proj = nn.Parameter(torch.randn(experts, ffn, hidden))


class ToyMlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(8, 4, bias=False)  # the router, frozen during RL
        self.experts = ToyExperts()


class ToyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 8, bias=False)
        self.o_proj = nn.Linear(8, 8, bias=False)
        self.mlp = ToyMlp()


class ToyPolicy(nn.Module):
    def __init__(self, layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([ToyLayer() for _ in range(layers)])
        self.register_buffer("inv_freq", torch.arange(4.0))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden


def build_peft_policy(*, seed: int = 0, **overrides):
    """The toy policy under the LoRA configuration the OLMoE run used."""
    torch.manual_seed(seed)
    settings = {
        "r": ATTENTION_RANK,
        "lora_alpha": 32,
        "lora_dropout": 0.0,
        "target_modules": ["q_proj", "o_proj"],
        "target_parameters": list(EXPERT_TARGETS),
        # The expert-side rank and scaling the OLMoE run set, in the form
        # grpo_fast writes them: keyed on the tail of the targeted parameter.
        "rank_pattern": dict.fromkeys(("experts.gate_up_proj", "experts.down_proj"), EXPERT_RANK),
        "alpha_pattern": dict.fromkeys(("experts.gate_up_proj", "experts.down_proj"), 8),
    }
    settings.update(overrides)
    model = get_peft_model(ToyPolicy(), LoraConfig(**settings))
    # A toy module has no ``config``, so PEFT blanks the base model name it was
    # given. The real run records it, and the export's base-mismatch guard needs it.
    model.peft_config["default"].base_model_name_or_path = BASE_MODEL
    for index, parameter in enumerate(p for p in model.parameters() if p.requires_grad):
        # lora_B initializes to zero, which would make every comparison below
        # pass for the wrong reason.
        parameter.data = torch.full_like(parameter, 0.01 * (index + 1))
    return model


def write_zero_checkpoint(
    root: Path,
    model,
    *,
    tag: str = "global_step200",
    world_size: int = 1,
    bf16: bool = True,
    stage: int = 2,
    write_latest: bool = True,
    training_step: int = 200,
) -> Path:
    """Write ``model``'s trainable parameters in the layout DeepSpeed ZeRO-2 writes.

    The flat partition, its padding, the per-rank optimizer shards and the
    parameter-shape groups are all laid out the way ``zero_to_fp32`` expects,
    because that is the reader the export actually uses.
    """
    step = root / tag
    step.mkdir(parents=True)
    trainable = OrderedDict((name, p) for name, p in model.named_parameters() if p.requires_grad)
    frozen = OrderedDict((name, p) for name, p in model.named_parameters() if not p.requires_grad)

    flat = torch.cat([p.detach().reshape(-1).float() for p in trainable.values()])
    # ZeRO-2 aligns each group to twice the world size, so the reconstruction
    # check compares the padded lengths rather than the exact ones.
    alignment = 2 * world_size
    padding = (-flat.numel()) % alignment
    flat = torch.cat([flat, torch.zeros(padding)])
    partitions = list(flat.chunk(world_size))

    torch.save(
        {
            "module": model.state_dict(),
            "buffer_names": ["base_model.model.inv_freq"],
            "optimizer": None,
            "param_shapes": [OrderedDict((name, p.shape) for name, p in trainable.items())],
            "frozen_param_shapes": OrderedDict((name, p.shape) for name, p in frozen.items()),
            "shared_params": {},
            "frozen_param_fragments": OrderedDict((name, p.detach()) for name, p in frozen.items()),
            "lr_scheduler": None,
            "sparse_tensor_module_names": set(),
            "skipped_steps": 0,
            "global_steps": training_step,
            "global_samples": 0,
            "dp_world_size": world_size,
            "mp_world_size": 1,
            "ds_config": {"zero_optimization": {"stage": stage}, "bf16": {"enabled": bf16}},
            "ds_version": "0.18.4",
            "training_step": training_step,
        },
        step / "mp_rank_00_model_states.pt",
    )
    for rank in range(world_size):
        name = f"{'bf16_' if bf16 else ''}zero_pp_rank_{rank}_mp_rank_00_optim_states.pt"
        torch.save(
            {
                "optimizer_state_dict": {
                    "zero_stage": stage,
                    "partition_count": world_size,
                    "single_partition_of_fp32_groups": [partitions[rank].clone()],
                    "group_paddings": [padding if rank == world_size - 1 else 0],
                    "param_slice_mappings": [],
                    "ds_version": "0.18.4",
                }
            },
            step / name,
        )
    if write_latest:
        (root / "latest").write_text(f"{tag}\n")
    (root / "zero_to_fp32.py").write_text("# written beside the steps by deepspeed\n")
    return step


def write_adapter_config(directory: Path, model, **overrides) -> Path:
    """The adapter_config.json PEFT writes beside an HF-format save of this run."""
    directory.mkdir(parents=True, exist_ok=True)
    body = model.peft_config["default"].to_dict()
    body = json.loads(json.dumps(body, default=lambda value: sorted(value) if isinstance(value, set) else str(value)))
    body["inference_mode"] = True
    body.update(overrides)
    path = directory / "adapter_config.json"
    path.write_text(json.dumps(body, indent=2))
    return path


@pytest.fixture
def olmoe_run(tmp_path):
    """A ZeRO state, and the adapter configuration the same run wrote elsewhere."""
    model = build_peft_policy()
    root = tmp_path / "ckpt" / "pedagogy_olmoe"
    step = write_zero_checkpoint(root, model)
    config = write_adapter_config(tmp_path / "output" / "pedagogy_olmoe_checkpoints" / "step_200", model)
    return model, step, config


# --------------------------------------------------------------------------
# layout inspection: no torch, no deepspeed, no GPU
# --------------------------------------------------------------------------


def test_a_zero_step_is_described_from_its_shards_alone(olmoe_run):
    _, step, _ = olmoe_run
    report = export.inspect_zero_step(step)
    assert report.kind == "deepspeed_zero"
    assert report.consolidatable
    assert report.tag == "global_step200" and report.step == 200
    assert report.optim_prefix == "bf16_"
    assert report.data_parallel_ranks == 1 and report.model_parallel_ranks == 1
    assert report.latest_tag == "global_step200"
    assert report.has_consolidation_script
    # The model-states shard is unpickled whole, so its size is the number that
    # decides the job's memory request.
    assert report.model_state_bytes > 0
    assert report.recommended_mem_gb >= export.MEMORY_FLOOR_GB
    assert report.problems == []


def test_inspection_reads_only_names_and_sizes(olmoe_run, monkeypatch):
    # A login node has no GPU and the shards are tens of gigabytes, so the
    # layout has to be decidable without opening one.
    _, step, _ = olmoe_run
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("inspection loaded a tensor"))
    assert export.inspect_zero_step(step).consolidatable


def test_a_second_model_parallel_rank_is_refused_rather_than_halved(olmoe_run):
    _, step, _ = olmoe_run
    (step / "mp_rank_01_model_states.pt").write_bytes(b"x")
    report = export.inspect_zero_step(step)
    assert not report.consolidatable
    assert "mp_rank_00 only" in report.reason
    with pytest.raises(export.AmbiguousCheckpoint):
        report.require_consolidatable()


def test_a_missing_data_parallel_rank_is_refused(tmp_path):
    model = build_peft_policy()
    step = write_zero_checkpoint(tmp_path / "ckpt", model, world_size=4)
    (step / "bf16_zero_pp_rank_2_mp_rank_00_optim_states.pt").unlink()
    report = export.inspect_zero_step(step)
    assert not report.consolidatable
    assert "not 0..2" in report.reason


def test_two_optimizer_prefixes_in_one_step_are_refused(olmoe_run):
    _, step, _ = olmoe_run
    (step / "zero_pp_rank_0_mp_rank_00_optim_states.pt").write_bytes(b"x")
    report = export.inspect_zero_step(step)
    assert not report.consolidatable
    assert "more than one prefix" in report.reason


def test_an_export_written_on_top_of_a_training_state_is_refused(olmoe_run):
    _, step, config = olmoe_run
    (step / "adapter_config.json").write_text(config.read_text())
    report = export.inspect_zero_step(step)
    assert not report.consolidatable
    assert "cannot be told apart" in report.reason


def test_a_step_whose_write_did_not_finish_is_refused(olmoe_run):
    _, step, _ = olmoe_run
    (step / "bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt").write_bytes(b"")
    report = export.inspect_zero_step(step)
    assert not report.consolidatable
    assert "did not finish" in report.reason


def test_a_step_with_no_optimizer_shards_is_refused_with_the_reason(olmoe_run):
    _, step, _ = olmoe_run
    (step / "bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt").unlink()
    report = export.inspect_zero_step(step)
    assert not report.consolidatable
    assert "optimizer partitions" in report.reason


def test_a_finished_export_is_not_mistaken_for_a_training_state(tmp_path):
    directory = tmp_path / "pedagogy_olmoe_exported" / "global_step200"
    directory.mkdir(parents=True)
    (directory / "adapter_config.json").write_text("{}")
    report = export.inspect_zero_step(directory)
    assert report.kind == "not_a_zero_step"
    assert "needs no export" in report.reason


def test_a_missing_step_is_reported_not_guessed(tmp_path):
    report = export.inspect_zero_step(tmp_path / "global_step999")
    assert report.kind == "missing" and not report.consolidatable


def test_a_step_without_a_latest_pointer_is_a_note_not_a_refusal(tmp_path):
    model = build_peft_policy()
    step = write_zero_checkpoint(tmp_path / "ckpt", model, write_latest=False)
    report = export.inspect_zero_step(step)
    assert report.consolidatable
    assert any("latest" in note for note in report.notes)


def test_a_latest_pointing_elsewhere_is_a_note_because_the_tag_is_explicit(tmp_path):
    model = build_peft_policy()
    root = tmp_path / "ckpt"
    step = write_zero_checkpoint(root, model, tag="global_step190")
    (root / "latest").write_text("global_step200\n")
    report = export.inspect_zero_step(step)
    assert report.consolidatable
    assert any("global_step200" in note for note in report.notes)


# --------------------------------------------------------------------------
# the parameter index, which needs torch and an explicit choice about pickle
# --------------------------------------------------------------------------


def test_the_parameter_index_names_the_lora_tensors_and_counts_the_frozen_base(olmoe_run):
    _, step, _ = olmoe_run
    index = export.read_parameter_index(export.inspect_zero_step(step), allow_pickle=True)
    assert index["adapter_names"] == ["default"]
    assert index["lora_parameters"] == index["trainable_parameters"] > 0
    assert index["frozen_parameters"] > 0
    assert index["zero_stage"] == 2
    assert index["engine_global_steps"] == 200
    assert all(name.startswith(export.PEFT_PREFIX) for name in index["shapes"])


def test_reading_a_training_shard_requires_saying_so(olmoe_run, monkeypatch):
    # A model-states file carries the trainer's client state as well as its
    # tensors, so reading it runs pickle from that file. Refusing by default and
    # naming the flag is the difference between a choice and an accident.
    _, step, _ = olmoe_run
    report = export.inspect_zero_step(step)
    unpatched = torch.load

    def refuse(*args, **kwargs):
        if kwargs.get("weights_only"):
            raise RuntimeError("Weights only load failed")
        return unpatched(*args, **kwargs)

    monkeypatch.setattr(torch, "load", refuse)
    with pytest.raises(export.CheckpointExportError, match="--allow-pickle"):
        export.read_parameter_index(report)
    assert export.read_parameter_index(report, allow_pickle=True)["lora_parameters"] > 0


# --------------------------------------------------------------------------
# adapter metadata is found, never invented
# --------------------------------------------------------------------------


def test_the_adapter_configuration_is_found_beside_the_hf_format_save(olmoe_run, tmp_path):
    _, _, config = olmoe_run
    source = export.find_adapter_config([tmp_path / "output" / "pedagogy_olmoe_checkpoints"], step=200)
    assert Path(source.path) == config
    assert source.config["peft_type"] == "LORA"
    assert source.config["base_model_name_or_path"] == BASE_MODEL


def test_two_configurations_that_disagree_are_an_error_not_a_choice(olmoe_run, tmp_path):
    model, _, _ = olmoe_run
    other = build_peft_policy(seed=1, r=32)
    write_adapter_config(tmp_path / "elsewhere" / "step_200", other)
    with pytest.raises(export.AmbiguousCheckpoint, match="different runs"):
        export.find_adapter_config(
            [tmp_path / "output" / "pedagogy_olmoe_checkpoints", tmp_path / "elsewhere"], step=200
        )
    assert model.peft_config["default"].r == 8


def test_two_copies_of_the_same_configuration_are_one_answer(olmoe_run, tmp_path):
    model, _, config = olmoe_run
    duplicate = tmp_path / "copy" / "step_200" / "adapter_config.json"
    duplicate.parent.mkdir(parents=True)
    duplicate.write_bytes(config.read_bytes())
    source = export.find_adapter_config(
        [tmp_path / "output" / "pedagogy_olmoe_checkpoints", tmp_path / "copy"], step=200
    )
    assert source.config["r"] == model.peft_config["default"].r


def test_no_configuration_anywhere_says_what_to_look_for(tmp_path):
    with pytest.raises(export.CheckpointExportError, match="not recoverable from a ZeRO state"):
        export.find_adapter_config([tmp_path], step=200)


def test_a_save_from_another_step_answers_when_the_two_counters_do_not_line_up(olmoe_run, tmp_path):
    # DeepSpeed counts optimizer steps and the HF-format saves are named for
    # training steps, so global_step180 need not have a step_180 beside it. The
    # rank and target lists belong to the run, so a sibling save answers.
    _, _, config = olmoe_run
    source = export.find_adapter_config([tmp_path / "output" / "pedagogy_olmoe_checkpoints"], step=180)
    assert Path(source.path) == config


def test_two_steps_of_the_same_run_that_disagree_still_refuse(olmoe_run, tmp_path):
    _, _, _ = olmoe_run
    root = tmp_path / "output" / "pedagogy_olmoe_checkpoints"
    write_adapter_config(root / "step_190", build_peft_policy(seed=5, r=16))
    with pytest.raises(export.AmbiguousCheckpoint, match="different runs"):
        export.find_adapter_config([root], step=180)


# --------------------------------------------------------------------------
# validation: does this configuration describe these tensors?
# --------------------------------------------------------------------------


def shapes_of(model) -> dict[str, tuple[int, ...]]:
    return {name: tuple(p.shape) for name, p in model.named_parameters() if p.requires_grad}


def test_the_run_configuration_explains_every_tensor_it_produced(olmoe_run):
    model, _, config = olmoe_run
    validation = export.validate_adapter(json.loads(config.read_text()), shapes_of(model))
    assert validation.ok, validation.problems
    assert validation.adapter_name == "default"
    assert validation.targets_by_kind["target_modules"] == 4  # q_proj and o_proj on two layers
    assert validation.targets_by_kind["target_parameters"] == 4  # two expert tensors on two layers
    assert validation.module_target_ranks == {str(ATTENTION_RANK): 4}
    assert validation.parameter_target_widths == {str(FUSED_WIDTH): 4}
    assert validation.expert_counts == [EXPERTS]


def test_a_fused_expert_adapter_is_experts_times_rank_wide_and_is_not_read_as_a_rank(olmoe_run):
    # PEFT stacks one LoRA per expert behind a fused 3-D weight, so lora_A is
    # experts * r wide. Reading that as the rank would refuse the OLMoE run,
    # whose 64 experts at rank 2 give 128 against a configured r of 8.
    model, _, config = olmoe_run
    body = json.loads(config.read_text())
    fused = [name for name in shapes_of(model) if "experts" in name and "lora_A" in name]
    assert fused and all(shapes_of(model)[name][0] == FUSED_WIDTH for name in fused)
    assert body["r"] != FUSED_WIDTH
    assert export.validate_adapter(body, shapes_of(model)).ok


def test_the_real_olmoe_widths_are_accepted_and_report_both_readings():
    # 64 experts at rank 2 and 16 at rank 8 are the same 128 tensors, and the
    # tensors do not say which. Both are reported rather than one being picked.
    body = {
        "peft_type": "LORA",
        "r": 8,
        "rank_pattern": {"experts.gate_up_proj": 2, "experts.down_proj": 2},
        "target_modules": ["q_proj"],
        "target_parameters": list(EXPERT_TARGETS),
        "base_model_name_or_path": BASE_MODEL,
    }
    shapes = {
        f"{export.PEFT_PREFIX}model.layers.0.self_attn.q_proj.lora_A.default.weight": (8, 2048),
        f"{export.PEFT_PREFIX}model.layers.0.self_attn.q_proj.lora_B.default.weight": (2048, 8),
        f"{export.PEFT_PREFIX}model.layers.0.mlp.experts.lora_A.default.weight": (128, 1024),
        f"{export.PEFT_PREFIX}model.layers.0.mlp.experts.lora_B.default.weight": (2048, 128),
        f"{export.PEFT_PREFIX}model.layers.0.mlp.experts.base_layer.lora_A.default.weight": (128, 2048),
        f"{export.PEFT_PREFIX}model.layers.0.mlp.experts.base_layer.lora_B.default.weight": (2048, 128),
    }
    validation = export.validate_adapter(body, shapes)
    assert validation.ok, validation.problems
    assert validation.expert_counts == [16, 64]
    assert any("do not separate the two factors" in note for note in validation.notes)


def test_a_configuration_from_a_different_run_is_caught_by_the_rank(olmoe_run, tmp_path):
    model, _, _ = olmoe_run
    other = build_peft_policy(seed=4, r=32, rank_pattern={}, alpha_pattern={})
    body = json.loads(write_adapter_config(tmp_path / "other_run", other).read_text())
    validation = export.validate_adapter(body, shapes_of(model))
    assert not validation.ok
    assert any("different run" in problem for problem in validation.problems)


def test_a_module_the_configuration_does_not_reach_is_refused(olmoe_run):
    model, _, config = olmoe_run
    body = json.loads(config.read_text())
    body["target_parameters"] = []
    validation = export.validate_adapter(body, shapes_of(model))
    assert not validation.ok
    assert any("neither target_modules nor target_parameters" in problem for problem in validation.problems)


def test_half_an_adapter_is_refused(olmoe_run):
    model, _, config = olmoe_run
    shapes = shapes_of(model)
    victim = next(name for name in shapes if name.endswith("lora_B.default.weight"))
    del shapes[victim]
    validation = export.validate_adapter(json.loads(config.read_text()), shapes)
    assert not validation.ok
    assert any("half an adapter" in problem for problem in validation.problems)


def test_more_wrappers_than_named_parameters_cannot_be_mapped_and_is_refused(olmoe_run):
    # Two parameters on one module give two nested wrappers whose names are
    # positional. A third wrapper has no parameter to belong to.
    model, _, config = olmoe_run
    shapes = shapes_of(model)
    stem = "base_model.model.layers.0.mlp.experts.base_layer.base_layer"
    shapes[f"{stem}.lora_A.default.weight"] = (FUSED_WIDTH, 8)
    shapes[f"{stem}.lora_B.default.weight"] = (12, FUSED_WIDTH)
    validation = export.validate_adapter(json.loads(config.read_text()), shapes)
    assert not validation.ok
    assert any("not recoverable from the names" in problem for problem in validation.problems)


def test_two_adapter_names_cannot_share_one_configuration(olmoe_run):
    model, _, config = olmoe_run
    shapes = shapes_of(model)
    donor = next(name for name in shapes if name.endswith("lora_A.default.weight"))
    shapes[donor.replace(".default.", ".other.")] = shapes[donor]
    validation = export.validate_adapter(json.loads(config.read_text()), shapes)
    assert not validation.ok
    assert any("one adapter per export" in problem for problem in validation.problems)


def test_a_non_lora_configuration_is_refused(olmoe_run):
    model, _, config = olmoe_run
    body = json.loads(config.read_text()) | {"peft_type": "PROMPT_TUNING"}
    assert any("PROMPT_TUNING" in problem for problem in export.validate_adapter(body, shapes_of(model)).problems)


def test_trained_biases_are_refused_because_they_are_not_reconstructed(olmoe_run):
    model, _, config = olmoe_run
    body = json.loads(config.read_text()) | {"bias": "all"}
    assert any("frozen" in problem for problem in export.validate_adapter(body, shapes_of(model)).problems)


# --------------------------------------------------------------------------
# the export itself
# --------------------------------------------------------------------------


def test_the_exported_keys_and_tensors_are_what_peft_would_have_saved(olmoe_run, tmp_path):
    model, step, config = olmoe_run
    reference = get_peft_model_state_dict(model)
    report = export.export_adapter(
        step, tmp_path / "exported" / "global_step200", adapter_config=export.read_adapter_config(config)
    )
    assert report.written and not report.dry_run
    written = load_file(tmp_path / "exported" / "global_step200" / export.ADAPTER_WEIGHTS_NAME)
    assert sorted(written) == sorted(reference)
    for key, value in reference.items():
        assert torch.allclose(written[key], value.float(), atol=1e-6), key
    assert report.tensors == len(reference)
    assert report.validation["ok"]


def test_the_frozen_base_never_enters_the_export(olmoe_run, tmp_path):
    # The 7B base is what makes these steps 27 GB. Excluding the frozen
    # parameters is why an export of a step costs a few hundred megabytes.
    model, step, config = olmoe_run
    destination = tmp_path / "exported" / "global_step200"
    export.export_adapter(step, destination, adapter_config=export.read_adapter_config(config))
    written = load_file(destination / export.ADAPTER_WEIGHTS_NAME)
    assert written and all("lora_" in key for key in written)
    frozen = {name for name, p in model.named_parameters() if not p.requires_grad}
    assert frozen and not any(name in written for name in frozen)


def test_the_adapter_configuration_is_copied_through_untouched(olmoe_run, tmp_path):
    _, step, config = olmoe_run
    destination = tmp_path / "exported" / "global_step200"
    source = export.read_adapter_config(config)
    export.export_adapter(step, destination, adapter_config=source)
    assert json.loads((destination / export.ADAPTER_CONFIG_NAME).read_text()) == source.config
    provenance = json.loads((destination / export.PROVENANCE_NAME).read_text())
    assert provenance["adapter_config_sha256"] == source.sha256
    assert provenance["source"] == str(step)
    assert provenance["tag"] == "global_step200"


def test_the_export_loads_back_through_peft_and_merges(olmoe_run, tmp_path):
    # The end of the road: PEFT rebuilds the nested parameter wrappers from the
    # copied configuration, finds every exported key, and merges. Nothing else
    # proves that ``mlp.experts.lora_A`` and ``mlp.experts.base_layer.lora_A``
    # went back onto the parameters they came off.
    model, step, config = olmoe_run
    destination = tmp_path / "exported" / "global_step200"
    export.export_adapter(step, destination, adapter_config=export.read_adapter_config(config))
    reloaded = PeftModel.from_pretrained(ToyPolicy(), str(destination), is_trainable=False)
    returned = get_peft_model_state_dict(reloaded)
    reference = get_peft_model_state_dict(model)
    assert sorted(returned) == sorted(reference)
    for key, value in reference.items():
        assert torch.allclose(returned[key].float(), value.float(), atol=1e-6), key
    assert isinstance(reloaded.merge_and_unload(), ToyPolicy)


def test_the_export_is_what_checkpoint_delta_agrees_to_load(olmoe_run, tmp_path):
    # The two modules classify checkpoints independently, and the export exists
    # to satisfy the classifier the tracer actually consults.
    _, step, config = olmoe_run
    destination = tmp_path / "exported" / "global_step200"
    export.export_adapter(step, destination, adapter_config=export.read_adapter_config(config))
    description = delta.describe_checkpoint(destination)
    assert description.kind == "peft_adapter"
    assert description.loadable
    assert description.architecture == "mixture_of_experts"
    assert description.adapts_experts and not description.adapts_router
    description.require_loadable()


def test_the_tokenizer_beside_the_configuration_travels_with_the_adapter(olmoe_run, tmp_path):
    _, step, config = olmoe_run
    (config.parent / "tokenizer_config.json").write_text("{}")
    (config.parent / "chat_template.jinja").write_text("{{ messages }}")
    destination = tmp_path / "exported" / "global_step200"
    report = export.export_adapter(step, destination, adapter_config=export.read_adapter_config(config))
    assert sorted(report.sidecars) == ["chat_template.jinja", "tokenizer_config.json"]
    assert (destination / "chat_template.jinja").read_text() == "{{ messages }}"


def test_bfloat16_is_available_for_reproducing_the_trainer_forward(olmoe_run, tmp_path):
    _, step, config = olmoe_run
    destination = tmp_path / "exported" / "global_step200"
    export.export_adapter(step, destination, adapter_config=export.read_adapter_config(config), dtype="bfloat16")
    written = load_file(destination / export.ADAPTER_WEIGHTS_NAME)
    assert all(value.dtype is torch.bfloat16 for value in written.values())


def test_a_dry_run_writes_nothing_and_still_checks_the_configuration(olmoe_run, tmp_path):
    _, step, config = olmoe_run
    destination = tmp_path / "exported" / "global_step200"
    report = export.export_adapter(
        step, destination, adapter_config=export.read_adapter_config(config), dry_run=True, allow_pickle=True
    )
    assert report.dry_run and not report.written
    assert report.validation["ok"]
    assert report.tensors > 0
    assert not destination.exists()


def test_a_dry_run_without_pickle_says_the_configuration_was_not_checked(olmoe_run, tmp_path):
    _, step, config = olmoe_run
    report = export.export_adapter(
        step, tmp_path / "exported", adapter_config=export.read_adapter_config(config), dry_run=True
    )
    assert report.validation["ok"] is None
    assert "--allow-pickle" in report.validation["reason"]


def test_a_dry_run_refuses_a_configuration_that_does_not_match_before_any_work(olmoe_run, tmp_path):
    _, step, _ = olmoe_run
    wrong = write_adapter_config(tmp_path / "wrong", build_peft_policy(seed=2, r=32))
    with pytest.raises(export.AmbiguousCheckpoint):
        export.export_adapter(
            step,
            tmp_path / "exported",
            adapter_config=export.read_adapter_config(wrong),
            dry_run=True,
            allow_pickle=True,
        )


def test_exporting_twice_is_accepted_and_writes_nothing_the_second_time(olmoe_run, tmp_path):
    _, step, config = olmoe_run
    destination = tmp_path / "exported" / "global_step200"
    source = export.read_adapter_config(config)
    first = export.export_adapter(step, destination, adapter_config=source)
    stamp = (destination / export.ADAPTER_WEIGHTS_NAME).stat().st_mtime_ns
    second = export.export_adapter(step, destination, adapter_config=source)
    assert first.written and not second.written
    assert "already" in second.reason
    assert (destination / export.ADAPTER_WEIGHTS_NAME).stat().st_mtime_ns == stamp


def test_a_destination_holding_a_different_export_is_refused_not_replaced(olmoe_run, tmp_path):
    model, step, config = olmoe_run
    destination = tmp_path / "exported" / "global_step200"
    export.export_adapter(step, destination, adapter_config=export.read_adapter_config(config))
    before = load_file(destination / export.ADAPTER_WEIGHTS_NAME)
    other_step = write_zero_checkpoint(tmp_path / "other", model, tag="global_step190")
    with pytest.raises(export.CheckpointExportError, match="move it aside"):
        export.export_adapter(other_step, destination, adapter_config=export.read_adapter_config(config))
    after = load_file(destination / export.ADAPTER_WEIGHTS_NAME)
    assert sorted(before) == sorted(after)


def test_a_destination_that_is_not_an_export_is_left_alone(olmoe_run, tmp_path):
    _, step, config = olmoe_run
    destination = tmp_path / "occupied"
    destination.mkdir()
    (destination / "someone_elses_work.txt").write_text("keep me")
    with pytest.raises(export.CheckpointExportError, match="move it aside"):
        export.export_adapter(step, destination, adapter_config=export.read_adapter_config(config))
    assert (destination / "someone_elses_work.txt").read_text() == "keep me"


def test_a_failed_export_leaves_no_half_written_directory(olmoe_run, tmp_path, monkeypatch):
    _, step, config = olmoe_run
    destination = tmp_path / "exported" / "global_step200"

    def explode(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("safetensors.torch.save_file", explode)
    with pytest.raises(OSError, match="disk full"):
        export.export_adapter(step, destination, adapter_config=export.read_adapter_config(config))
    assert not destination.exists()
    assert list(destination.parent.glob("*.partial-*")) == []


def test_an_unconsolidatable_step_is_refused_before_the_configuration_is_read(olmoe_run, tmp_path):
    _, step, config = olmoe_run
    (step / "mp_rank_01_model_states.pt").write_bytes(b"x")
    with pytest.raises(export.AmbiguousCheckpoint, match="mp_rank_00 only"):
        export.export_adapter(step, tmp_path / "exported", adapter_config=export.read_adapter_config(config))


def test_a_four_rank_checkpoint_reassembles_the_same_tensors(tmp_path):
    model = build_peft_policy(seed=3)
    reference = get_peft_model_state_dict(model)
    step = write_zero_checkpoint(tmp_path / "ckpt", model, world_size=4)
    config = write_adapter_config(tmp_path / "output", model)
    destination = tmp_path / "exported" / "global_step200"
    export.export_adapter(step, destination, adapter_config=export.read_adapter_config(config))
    written = load_file(destination / export.ADAPTER_WEIGHTS_NAME)
    for key, value in reference.items():
        assert torch.allclose(written[key], value.float(), atol=1e-6), key


# --------------------------------------------------------------------------
# names
# --------------------------------------------------------------------------


def test_the_adapter_name_is_dropped_exactly_where_peft_drops_it(olmoe_run):
    model, _, _ = olmoe_run
    reference = get_peft_model_state_dict(model)
    live = [name for name, p in model.named_parameters() if p.requires_grad]
    assert sorted(export.peft_state_dict_key(name, "default") for name in live) == sorted(reference)


def test_a_module_that_happens_to_be_called_default_survives():
    name = "base_model.model.layers.0.default.q_proj.lora_A.default.weight"
    assert export.peft_state_dict_key(name, "default") == "base_model.model.layers.0.default.q_proj.lora_A.weight"


def test_a_parameter_shaped_lora_key_has_no_weight_suffix_to_stitch_back():
    assert export.peft_state_dict_key("a.b.lora_embedding_A.default", "default") == "a.b.lora_embedding_A"


def test_the_nesting_depth_is_what_separates_two_parameters_on_one_module():
    assert export.split_target("layers.0.mlp.experts") == ("layers.0.mlp.experts", 0)
    assert export.split_target("layers.0.mlp.experts.base_layer") == ("layers.0.mlp.experts", 1)
    assert export.parse_lora_name("layers.0.self_attn.q_proj.weight") is None
    parsed = export.parse_lora_name("base_model.model.layers.0.mlp.experts.lora_A.default.weight")
    assert parsed is not None
    assert parsed.module == "layers.0.mlp.experts" and parsed.kind == "lora_A" and parsed.adapter == "default"


# --------------------------------------------------------------------------
# the dense arm, which arrives as a zip
# --------------------------------------------------------------------------


def write_adapter_zip(path: Path, *, root: str = "pedagogy-tutor-armF/", extra: dict[str, str] | None = None) -> Path:
    body = {
        "peft_type": "LORA",
        "base_model_name_or_path": "allenai/OLMo-2-1124-7B-Instruct",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "r": 32,
        "lora_alpha": 64,
    }
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{root}adapter_config.json", json.dumps(body))
        archive.writestr(f"{root}adapter_model.safetensors", b"not really safetensors")
        archive.writestr(f"{root}tokenizer_config.json", "{}")
        for name, content in (extra or {}).items():
            archive.writestr(name, content)
    return path


def test_the_dense_tutor_zip_is_read_without_extracting_it(tmp_path):
    report = export.inspect_adapter_zip(write_adapter_zip(tmp_path / "pedagogy-tutor-armF.zip"))
    assert report.kind == "peft_adapter_zip" and report.consolidatable
    assert report.base_model == "allenai/OLMo-2-1124-7B-Instruct"
    assert report.root == "pedagogy-tutor-armF/"
    assert report.weight_file == "adapter_model.safetensors"


def test_staging_the_dense_zip_produces_a_directory_checkpoint_delta_loads(tmp_path):
    archive = write_adapter_zip(tmp_path / "pedagogy-tutor-armF.zip")
    destination = tmp_path / "exported" / "armF"
    report = export.stage_adapter_zip(archive, destination)
    assert report.written
    assert report.files == ["adapter_config.json", "adapter_model.safetensors", "tokenizer_config.json"]
    assert (destination / "adapter_config.json").is_file()
    assert (destination / "tokenizer_config.json").is_file()
    description = delta.describe_checkpoint(destination)
    assert description.kind == "peft_adapter" and description.loadable
    assert description.architecture == "dense"


def test_a_zip_member_that_would_escape_the_destination_is_refused(tmp_path):
    archive = write_adapter_zip(tmp_path / "evil.zip", extra={"../escaped.txt": "no"})
    report = export.inspect_adapter_zip(archive)
    assert not report.consolidatable
    assert "outside the destination" in report.reason
    with pytest.raises(export.AmbiguousCheckpoint):
        export.stage_adapter_zip(archive, tmp_path / "exported")
    assert not (tmp_path / "escaped.txt").exists()


def test_two_adapters_in_one_zip_are_refused(tmp_path):
    archive = write_adapter_zip(tmp_path / "two.zip", extra={"second/adapter_config.json": "{}"})
    report = export.inspect_adapter_zip(archive)
    assert not report.consolidatable
    assert "2 adapters" in report.reason


def test_a_zip_with_a_configuration_but_no_weights_is_refused(tmp_path):
    archive = tmp_path / "empty.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("armF/adapter_config.json", json.dumps({"peft_type": "LORA"}))
    report = export.inspect_adapter_zip(archive)
    assert not report.consolidatable
    assert "no adapter_model weights" in report.reason


def test_staging_the_same_zip_twice_writes_nothing_the_second_time(tmp_path):
    archive = write_adapter_zip(tmp_path / "armF.zip")
    destination = tmp_path / "exported" / "armF"
    assert export.stage_adapter_zip(archive, destination).written
    assert not export.stage_adapter_zip(archive, destination).written


def test_a_zip_dry_run_writes_nothing(tmp_path):
    archive = write_adapter_zip(tmp_path / "armF.zip")
    destination = tmp_path / "exported" / "armF"
    report = export.stage_adapter_zip(archive, destination, dry_run=True)
    assert report.dry_run and not report.written
    assert not destination.exists()


# --------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------


def test_inspect_prints_the_layout_and_can_be_asked_to_fail_on_a_refusal(olmoe_run, capsys):
    _, step, _ = olmoe_run
    export.main(["inspect", str(step), "--require-consolidatable"])
    assert "deepspeed_zero" in capsys.readouterr().out
    (step / "mp_rank_01_model_states.pt").write_bytes(b"x")
    export.main(["inspect", str(step)])  # classifying a refused layout still succeeds
    with pytest.raises(SystemExit, match="cannot be consolidated"):
        export.main(["inspect", str(step), "--require-consolidatable"])


def test_the_command_line_finds_the_configuration_it_is_pointed_at(olmoe_run, tmp_path, capsys):
    _, step, _ = olmoe_run
    export.main(
        [
            "export",
            "--step",
            str(step),
            "--out",
            str(tmp_path / "exported" / "global_step200"),
            "--adapter-config-search",
            str(tmp_path / "output" / "pedagogy_olmoe_checkpoints"),
        ]
    )
    report = json.loads(capsys.readouterr().out)
    assert report["written"] and report["schema"] == export.SCHEMA
    assert report["validation"]["ok"]
