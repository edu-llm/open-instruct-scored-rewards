"""Turn a training state into something the tracer will load, or say why it cannot.

``checkpoint_delta`` refuses a DeepSpeed ZeRO directory on purpose. A step under
``ckpt/pedagogy_olmoe/global_step{180,190,200}`` is optimizer partitions beside a
frozen base, and nothing inside it records which parameters its LoRA tensors
adapt. This module is the other half of that refusal: it consolidates one step
into the PEFT adapter directory ``compare_rl_checkpoints.sbatch`` looks for, and
it declines to guess exactly the things the refusal was about.

Four things it will not do.

It will not invent an ``adapter_config.json``. Rank, alpha, target modules and
target parameters are training-time facts; PEFT wrote them beside the HF-format
saves under ``output/<exp>_checkpoints/step_<N>``, and one of those files has to
be named. Two candidates whose bytes differ are an error rather than a choice.

It will not rename a LoRA tensor. Per-expert LoRA reaches OLMoE's fused 3-D
expert weights through PEFT's parameter wrapper, and those module names are
positional: ``mlp.experts.lora_A`` and ``mlp.experts.base_layer.lora_A`` are two
parameters on one module, and neither name says which parameter it adapts. They
round-trip only because the same config rebuilds the same nesting in the same
order, so the names are copied through untouched and the config is copied with
them.

It will not accept a layout that admits two readings. A second model-parallel
rank, a gap in the data-parallel ranks, both a bf16 and a plain optimizer
prefix, or a ``config.json`` sitting inside a ZeRO step each stop the export
with the reason attached.

It will not overwrite or delete. An existing destination is compared against its
recorded provenance and is either accepted as already done or refused; every
write lands in a sibling ``.partial-<pid>`` directory that is renamed into place
at the end.

Inspection is separate from export and needs neither torch nor deepspeed nor a
GPU, which is what makes ``inspect`` and ``--dry-run`` usable on a login node.
Reading the parameter index does need torch, and unpickling a training shard is
opt-in behind ``--allow-pickle`` because a model-states file carries the
trainer's client state as well as its tensors.

The dense arm goes through the same door. ``ckpt/pedagogy-tutor-armF.zip`` is
already a PEFT adapter, so ``stage-zip`` extracts it into the same export layout
after checking that it holds exactly one adapter and no path that escapes the
destination.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
import zipfile
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCHEMA = "semantic-atlas-checkpoint-export/v1"

MODEL_STATE_SUFFIX = "_model_states.pt"
OPTIM_STATE_SUFFIX = "_optim_states.pt"
ADAPTER_CONFIG_NAME = "adapter_config.json"
ADAPTER_WEIGHTS_NAME = "adapter_model.safetensors"
PROVENANCE_NAME = "export_provenance.json"
# PeftModel wraps the causal LM twice, so every tensor DeepSpeed saved for the
# policy starts here. A name that does not is not from the model we think it is.
PEFT_PREFIX = "base_model.model."
# Copied beside the adapter when they sit beside the config PEFT wrote, so the
# export says which tokenizer produced the text the adapter was trained on.
SIDECAR_NAMES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja")
DTYPES = ("float32", "bfloat16", "float16")
# fp32 master weights are what ZeRO reconstructs, and keeping them costs a few
# hundred MB on an adapter. bfloat16 reproduces the trainer's forward exactly.
DEFAULT_DTYPE = "float32"
# The model-states shard is unpickled whole by deepspeed, and under ZeRO-2 it
# holds the module state dict and the frozen fragments, so peak resident memory
# tracks its size rather than the size of the adapter being written.
MEMORY_HEADROOM = 1.4
MEMORY_FLOOR_GB = 8


class CheckpointExportError(RuntimeError):
    """A checkpoint could not be inspected or exported."""


class AmbiguousCheckpoint(CheckpointExportError):
    """The layout admits more than one reading, so nothing is written."""


# --------------------------------------------------------------------------
# LoRA tensor names
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LoraTensor:
    """One ``...lora_A.default.weight`` name, split where PEFT splits it."""

    name: str
    qualified_module: str
    kind: str
    adapter: str
    suffix: str

    @property
    def module(self) -> str:
        """The module path with the PeftModel prefix removed."""
        return self.qualified_module.removeprefix(PEFT_PREFIX)


def split_target(module: str) -> tuple[str, int]:
    """The wrapped module and how deeply the wrapper is nested.

    Targeting two parameters on one module makes PEFT wrap the wrapper, so the
    second adapter appears one ``base_layer`` deeper than the first. That depth
    is the only thing separating them, and it is positional rather than named.
    """
    parts = module.split(".")
    depth = 0
    while parts and parts[-1] == "base_layer":
        parts.pop()
        depth += 1
    return ".".join(parts), depth


def parse_lora_name(name: str) -> LoraTensor | None:
    """Split a checkpoint parameter name, or return None if it is not a LoRA tensor."""
    parts = name.split(".")
    positions = [index for index, part in enumerate(parts) if part.startswith("lora_")]
    if not positions:
        return None
    index = positions[-1]
    return LoraTensor(
        name=name,
        qualified_module=".".join(parts[:index]),
        kind=parts[index],
        adapter=parts[index + 1] if index + 1 < len(parts) else "",
        suffix=".".join(parts[index + 2 :]),
    )


def peft_state_dict_key(name: str, adapter_name: str) -> str:
    """Drop the adapter name exactly where ``get_peft_model_state_dict`` drops it.

    PEFT removes the name only at the end of the key or immediately before the
    trailing ``weight``/``bias``, never in the middle, so a module that happens
    to be called ``default`` survives. ``test_the_exported_keys_match_peft``
    pins this against PEFT itself rather than against this docstring.
    """
    if "." not in name:
        return name
    if name.endswith(f".{adapter_name}"):
        return name.removesuffix(f".{adapter_name}")
    head, _, suffix = name.rpartition(".")
    head = re.sub(re.escape(f".{adapter_name}") + r"$", "", head)
    return f"{head}.{suffix}"


# --------------------------------------------------------------------------
# ZeRO step inspection: filesystem only, no torch, no GPU
# --------------------------------------------------------------------------


@dataclass
class ZeroStepReport:
    """Everything decidable about one ``global_step<N>`` without reading a tensor."""

    path: str
    kind: str
    consolidatable: bool
    reason: str = ""
    checkpoint_root: str = ""
    tag: str = ""
    step: int | None = None
    model_state_files: list[str] = field(default_factory=list)
    optim_state_files: list[str] = field(default_factory=list)
    optim_prefix: str = ""
    data_parallel_ranks: int = 0
    model_parallel_ranks: int = 0
    latest_tag: str | None = None
    has_consolidation_script: bool = False
    model_state_bytes: int = 0
    optim_state_bytes: int = 0
    recommended_mem_gb: int = 0
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    parameter_index: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def require_consolidatable(self) -> ZeroStepReport:
        if not self.consolidatable:
            raise AmbiguousCheckpoint(f"{self.path}: {self.reason}")
        return self


def _rank_ids(names: Iterable[str], pattern: re.Pattern[str], group: str) -> list[int]:
    found = set()
    for name in names:
        match = pattern.search(name)
        if match and match.group(group) is not None:
            found.add(int(match.group(group)))
    return sorted(found)


_DP_RANK = re.compile(r"zero_pp_rank_(?P<dp>\d+)")
_MP_RANK = re.compile(r"mp_rank_(?P<mp>\d+)")
_OPTIM_PREFIX = re.compile(r"^(?P<prefix>.*?)zero_pp_rank_\d+")


def _recommended_mem_gb(model_state_bytes: int) -> int:
    gigabytes = model_state_bytes / float(1 << 30)
    return max(MEMORY_FLOOR_GB, int(gigabytes * MEMORY_HEADROOM) + MEMORY_FLOOR_GB)


def inspect_zero_step(path: Path | str) -> ZeroStepReport:
    """Classify one DeepSpeed step directory and decide whether it can be consolidated."""
    path = Path(path)
    if not path.exists():
        return ZeroStepReport(path=str(path), kind="missing", consolidatable=False, reason="path does not exist")
    if not path.is_dir():
        return ZeroStepReport(
            path=str(path), kind="not_a_zero_step", consolidatable=False, reason="a ZeRO step is a directory"
        )

    entries = sorted(item.name for item in path.iterdir())
    model_files = [name for name in entries if name.endswith(MODEL_STATE_SUFFIX)]
    optim_files = [name for name in entries if name.endswith(OPTIM_STATE_SUFFIX)]
    if not model_files and not optim_files:
        reason = "no *_model_states.pt and no *_optim_states.pt, so this is not a DeepSpeed step"
        if ADAPTER_CONFIG_NAME in entries or "config.json" in entries:
            reason = "this directory holds a model or an adapter, not a training state; it needs no export"
        return ZeroStepReport(path=str(path), kind="not_a_zero_step", consolidatable=False, reason=reason)

    root = path.parent
    siblings = sorted(item.name for item in root.iterdir()) if root.exists() else []
    latest_tag = None
    latest_file = root / "latest"
    if latest_file.is_file():
        latest_tag = latest_file.read_text().strip() or None

    step = int(path.name.removeprefix("global_step")) if path.name.removeprefix("global_step").isdigit() else None
    dp_ranks = _rank_ids(optim_files, _DP_RANK, "dp")
    mp_ranks = _rank_ids(model_files + optim_files, _MP_RANK, "mp")
    prefixes = {match.group("prefix") for name in optim_files if (match := _OPTIM_PREFIX.match(name))}

    problems: list[str] = []
    notes: list[str] = []
    if not optim_files:
        problems.append(
            "no *_optim_states.pt shards: ZeRO rebuilds the trained weights from the optimizer partitions, "
            "so a step without them holds no trainable parameters to export"
        )
    if not model_files:
        problems.append(
            "no *_model_states.pt shards: the parameter names and their order live there, and without them "
            "the flat optimizer partitions cannot be split back into named tensors"
        )
    if len(mp_ranks) > 1:
        problems.append(
            f"model-parallel ranks {mp_ranks} are present; this export reads mp_rank_00 only and will not "
            "silently drop the rest"
        )
    if optim_files and not dp_ranks:
        problems.append(
            f"the optimizer shards {optim_files[:3]} carry no zero_pp_rank, so this is not a ZeRO checkpoint "
            "and there are no partitions to concatenate"
        )
    elif dp_ranks != list(range(len(dp_ranks))):
        problems.append(
            f"data-parallel ranks {dp_ranks} are not 0..{len(dp_ranks) - 1}; a rank failed to write and the "
            "partitions cannot be concatenated"
        )
    elif len(dp_ranks) != len(optim_files):
        problems.append(
            f"{len(optim_files)} optimizer shards carry {len(dp_ranks)} distinct data-parallel ranks; "
            "deepspeed reconstructs one partition per rank and would refuse this too"
        )
    if len(prefixes) > 1:
        problems.append(
            f"optimizer shards carry more than one prefix {sorted(prefixes)}; a bf16 and a non-bf16 optimizer "
            "wrote into the same step and the pair cannot be read as one"
        )
    stage_three_model_states = [name for name in model_files if name.startswith("zero_pp_rank_")]
    if stage_three_model_states and "mp_rank_00_model_states.pt" in model_files:
        problems.append("both ZeRO-2 and ZeRO-3 model-state names are present, so the stage of this step is ambiguous")
    if ADAPTER_CONFIG_NAME in entries or "config.json" in entries:
        problems.append(
            f"a {ADAPTER_CONFIG_NAME if ADAPTER_CONFIG_NAME in entries else 'config.json'} sits inside this "
            "training state; an export has been written on top of it and the two cannot be told apart"
        )
    empty = [name for name in model_files + optim_files if (path / name).stat().st_size == 0]
    if empty:
        problems.append(f"these shards are empty and the write did not finish: {empty}")

    if latest_tag is None:
        notes.append("no 'latest' pointer beside this step, so the tag is passed explicitly")
    elif latest_tag != path.name:
        notes.append(f"'latest' names {latest_tag!r} rather than {path.name!r}; the tag is passed explicitly")
    if "zero_to_fp32.py" not in siblings:
        notes.append("no zero_to_fp32.py beside the steps; consolidation uses the installed deepspeed instead")

    model_bytes = sum((path / name).stat().st_size for name in model_files)
    optim_bytes = sum((path / name).stat().st_size for name in optim_files)
    return ZeroStepReport(
        path=str(path),
        kind="deepspeed_zero",
        consolidatable=not problems,
        reason="; ".join(problems),
        checkpoint_root=str(root),
        tag=path.name,
        step=step,
        model_state_files=model_files,
        optim_state_files=optim_files,
        optim_prefix=next(iter(prefixes), ""),
        data_parallel_ranks=len(dp_ranks),
        model_parallel_ranks=max(len(mp_ranks), 1),
        latest_tag=latest_tag,
        has_consolidation_script="zero_to_fp32.py" in siblings,
        model_state_bytes=model_bytes,
        optim_state_bytes=optim_bytes,
        recommended_mem_gb=_recommended_mem_gb(model_bytes),
        problems=problems,
        notes=notes,
    )


def read_parameter_index(report: ZeroStepReport, *, allow_pickle: bool = False) -> dict[str, Any]:
    """Read parameter names and shapes from the model-states shard. Needs torch, not a GPU.

    A DeepSpeed model-states file carries the trainer's client state - dataloader
    position, RNG state, data-prep actor state - alongside its tensors, so
    ``weights_only=True`` usually refuses it. That refusal is doing its job, and
    the way past it is an explicit ``allow_pickle``, not a silent fallback.
    """
    import torch  # noqa: PLC0415

    report.require_consolidatable()
    shard = Path(report.path) / "mp_rank_00_model_states.pt"
    if not shard.is_file():
        candidates = [name for name in report.model_state_files if name.endswith(MODEL_STATE_SUFFIX)]
        if not candidates:
            raise CheckpointExportError(f"{report.path}: no model-states shard to read")
        shard = Path(report.path) / candidates[0]
    try:
        state = torch.load(shard, map_location="cpu", weights_only=True, mmap=True)
    except Exception as exc:
        if not allow_pickle:
            raise CheckpointExportError(
                f"{shard} cannot be read with weights_only=True ({type(exc).__name__}: {exc}). A DeepSpeed "
                "model-states file stores the trainer's client state beside its tensors, so reading it "
                "executes pickle from that file; pass --allow-pickle to accept that, and run it from the "
                "repository whose classes the state refers to."
            ) from exc
        state = torch.load(shard, map_location="cpu", weights_only=False, mmap=True)

    trainable: dict[str, list[int]] = {}
    for group in state.get("param_shapes") or []:
        for name, shape in group.items():
            trainable[str(name)] = [int(value) for value in shape]
    frozen = state.get("frozen_param_shapes") or {}
    lora = {name: shape for name, shape in trainable.items() if parse_lora_name(name)}
    adapters = sorted({parsed.adapter for name in lora if (parsed := parse_lora_name(name)) and parsed.adapter})
    configured = state.get("ds_config") or {}
    return {
        "shard": str(shard),
        "trainable_parameters": len(trainable),
        "frozen_parameters": len(frozen),
        "lora_parameters": len(lora),
        "adapter_names": adapters,
        "buffers": len(state.get("buffer_names") or []),
        "ds_version": state.get("ds_version"),
        "zero_stage": (configured.get("zero_optimization") or {}).get("stage"),
        "dp_world_size": state.get("dp_world_size"),
        "mp_world_size": state.get("mp_world_size"),
        "engine_global_steps": state.get("global_steps"),
        "trainer_training_step": state.get("training_step"),
        "shapes": trainable,
    }


# --------------------------------------------------------------------------
# adapter metadata: found, never invented
# --------------------------------------------------------------------------


@dataclass
class AdapterConfigSource:
    """One ``adapter_config.json``, with the hash that decides whether two are the same."""

    path: str
    sha256: str
    config: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {"path": self.path, "sha256": self.sha256, "config": self.config}


def read_adapter_config(path: Path | str) -> AdapterConfigSource:
    path = Path(path)
    if path.is_dir():
        path = path / ADAPTER_CONFIG_NAME
    if not path.is_file():
        raise CheckpointExportError(f"{path} is not an adapter_config.json")
    raw = path.read_bytes()
    body = json.loads(raw)
    if not isinstance(body, dict):
        raise CheckpointExportError(f"{path} does not hold a json object")
    return AdapterConfigSource(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), config=body)


def find_adapter_config(roots: Sequence[Path | str], *, step: int | None = None) -> AdapterConfigSource:
    """Find the config PEFT wrote for this run, and refuse if the candidates disagree.

    Every candidate is read rather than the first one taken. Two configs with the
    same bytes are one answer; two that differ are two runs, and picking either
    would attach the wrong rank and target list to these tensors.

    A step's own save is preferred when there is one. DeepSpeed's ``global_step``
    counts optimizer steps and the HF-format saves are named for training steps,
    so the two numbers need not line up; when they do not, any save from the same
    run answers, because the rank and target lists are a property of the run
    rather than of the step. That fallback is only safe because it still refuses
    the moment two of those saves disagree.
    """
    exact: list[Path] = []
    anywhere: list[Path] = []
    for root in roots:
        root = Path(root)
        for name in (root / ADAPTER_CONFIG_NAME, *sorted(root.glob(f"*step*/{ADAPTER_CONFIG_NAME}"))):
            if not name.is_file():
                continue
            anywhere.append(name)
            if step is not None and name.parent.name in {f"step_{step}", f"global_step{step}"}:
                exact.append(name)
    candidates = exact or anywhere
    if not candidates:
        searched = ", ".join(str(Path(root)) for root in roots) or "nothing"
        raise CheckpointExportError(
            f"no {ADAPTER_CONFIG_NAME} under {searched}. The rank, alpha and target lists are training-time "
            "facts and are not recoverable from a ZeRO state; point --adapter-config at the one PEFT wrote "
            "beside this run's HF-format save."
        )
    sources = [read_adapter_config(path) for path in candidates]
    distinct = {source.sha256: source for source in sources}
    if len(distinct) > 1:
        listing = "; ".join(f"{source.path} ({digest[:12]})" for digest, source in sorted(distinct.items()))
        raise AmbiguousCheckpoint(
            f"{len(distinct)} different adapter configurations are in scope and they describe different runs: "
            f"{listing}. Name one with --adapter-config."
        )
    return sources[0]


# --------------------------------------------------------------------------
# validation: does this configuration describe these tensors?
# --------------------------------------------------------------------------


@dataclass
class AdapterValidation:
    ok: bool
    adapter_name: str = ""
    tensors: int = 0
    parameters: int = 0
    targets: int = 0
    module_target_ranks: dict[str, int] = field(default_factory=dict)
    parameter_target_widths: dict[str, int] = field(default_factory=dict)
    expert_counts: list[int] = field(default_factory=list)
    targets_by_kind: dict[str, int] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def require_ok(self, source: str) -> AdapterValidation:
        if not self.ok:
            raise AmbiguousCheckpoint(f"{source}: " + "; ".join(self.problems))
        return self


def _parameter_parents(config: dict[str, Any]) -> tuple[list[str], bool]:
    """The module paths ``target_parameters`` reaches, and whether any is bare."""
    parents, bare = [], False
    for target in config.get("target_parameters") or []:
        text = str(target)
        if "." in text:
            parents.append(text.rsplit(".", 1)[0])
        else:
            bare = True
    return parents, bare


def _explains(config: dict[str, Any], module: str, parents: Sequence[str], bare: bool) -> str:
    modules = config.get("target_modules")
    if isinstance(modules, str):
        if re.fullmatch(modules, module):
            return "target_modules"
    elif modules:
        targets = [str(value) for value in modules]
        if module.rsplit(".", 1)[-1] in targets or any(module.endswith(f".{target}") for target in targets):
            return "target_modules"
    if any(module == parent or module.endswith(f".{parent}") for parent in parents):
        return "target_parameters"
    if bare and (config.get("target_parameters") or []):
        return "target_parameters"
    return ""


def _numel(shape: Sequence[int]) -> int:
    total = 1
    for value in shape:
        total *= int(value)
    return total


def _expert_counts(widths: Iterable[int], allowed_ranks: Iterable[int]) -> list[int]:
    """Expert counts under which every fused width is an allowed rank times that count.

    A fused expert adapter stacks one LoRA per expert, so its inner dimension is
    ``experts * r`` rather than ``r``. The two factors are not separable from the
    tensors alone: 128 is 64 experts at rank 2 and equally 16 at rank 8. What is
    checkable is that one expert count explains every fused tensor at once.
    """
    widths = sorted(set(widths))
    ranks = {rank for rank in allowed_ranks if rank > 0}
    if not widths or not ranks:
        return []
    return [
        count
        for count in range(1, widths[0] + 1)
        if all(width % count == 0 and width // count in ranks for width in widths)
    ]


def validate_adapter(config: dict[str, Any], shapes: dict[str, Sequence[int]]) -> AdapterValidation:
    """Check that this configuration could have produced these tensors.

    The two kinds of target are checked differently because their shapes mean
    different things. A LoRA on a ``Linear`` has ``lora_A`` of shape ``(r, in)``,
    so the rank is read straight off it and has to be one the configuration can
    produce. A LoRA reaching a fused 3-D expert weight through
    ``target_parameters`` has ``lora_A`` of shape ``(experts * r, in)``, so the
    same read gives 128 for the OLMoE run's rank of 2 across 64 experts. Testing
    that against ``r`` would refuse the only checkpoint there is; the fused
    tensors are instead required to share one expert count that makes every one
    of them an allowed rank.

    The membership test is against ``r`` together with every ``rank_pattern``
    value, rather than against the value the pattern is supposed to select.
    Resolving the pattern is PEFT's business - it matches these keys against the
    key of the module it is about to wrap, and for a parameter target that key
    carries the parameter name - and reimplementing that resolution here would
    make this refuse checkpoints whenever PEFT's matching changed.
    """
    problems: list[str] = []
    notes: list[str] = []
    peft_type = str(config.get("peft_type", "")).upper()
    if peft_type != "LORA":
        problems.append(f"peft_type {peft_type or 'unknown'} is not supported here")
    bias = str(config.get("bias", "none"))
    if bias != "none":
        problems.append(
            f"bias={bias!r} means the export would also need the base biases, which are frozen and are "
            "therefore not reconstructed here"
        )
    if not config.get("base_model_name_or_path"):
        notes.append(
            "base_model_name_or_path is empty, so checkpoint_delta cannot check the adapter against the base "
            "it is applied to"
        )

    parsed = {name: parse_lora_name(name) for name in shapes}
    lora = {name: tensor for name, tensor in parsed.items() if tensor is not None}
    if not lora:
        problems.append("the checkpoint holds no LoRA tensors")
        return AdapterValidation(ok=False, problems=problems, notes=notes)

    adapters = sorted({tensor.adapter for tensor in lora.values()})
    if len(adapters) != 1:
        problems.append(f"tensors carry {len(adapters)} adapter names {adapters}; one adapter per export")
    unprefixed = sorted(name for name in lora if not name.startswith(PEFT_PREFIX))[:3]
    if unprefixed:
        problems.append(
            f"these LoRA tensors do not start with {PEFT_PREFIX!r} and so are not from a PeftModel: {unprefixed}"
        )

    by_module: dict[str, dict[str, Sequence[int]]] = {}
    for name, tensor in lora.items():
        by_module.setdefault(tensor.module, {})[tensor.kind] = shapes[name]

    parents, bare = _parameter_parents(config)
    allowed_ranks = {int(config.get("r", 0))} | {int(value) for value in (config.get("rank_pattern") or {}).values()}
    module_ranks: list[int] = []
    fused_widths: list[int] = []
    kinds: dict[str, int] = {}
    depths: dict[str, set[int]] = {}
    targets = 0
    for module, tensors in sorted(by_module.items()):
        missing = [kind for kind in ("lora_A", "lora_B") if kind not in tensors]
        if missing:
            problems.append(f"{module} has no {missing}; the export would be half an adapter")
            continue
        left, right = list(tensors["lora_A"]), list(tensors["lora_B"])
        if not left or not right or left[0] != right[-1]:
            problems.append(f"{module}: lora_A is {left} and lora_B is {right}, which share no inner dimension")
            continue
        width = int(left[0])
        targets += 1
        target, depth = split_target(module)
        explained = _explains(config, target, parents, bare)
        if not explained:
            problems.append(
                f"{target} is adapted in the checkpoint but neither target_modules nor target_parameters reaches it"
            )
            continue
        if explained == "target_modules":
            module_ranks.append(width)
            if allowed_ranks and width not in allowed_ranks:
                problems.append(
                    f"{module} holds rank {width} but this configuration can only produce "
                    f"{sorted(allowed_ranks)}; the configuration belongs to a different run"
                )
        else:
            fused_widths.append(width)
        kinds[explained] = kinds.get(explained, 0) + 1
        depths.setdefault(target, set()).add(depth)

    expert_counts = _expert_counts(fused_widths, allowed_ranks)
    if fused_widths and not expert_counts:
        problems.append(
            f"the fused expert adapters are {sorted(set(fused_widths))} wide, and no single expert count makes "
            f"all of them a rank this configuration can produce {sorted(allowed_ranks)}; the configuration "
            "belongs to a different run"
        )
    elif len(expert_counts) > 1:
        notes.append(
            f"a fused expert adapter's width is experts times rank, and {expert_counts} all fit; the tensors "
            "alone do not separate the two factors, so the rank here is checked only up to that product"
        )
    if fused_widths and not module_ranks:
        notes.append(
            "every target is a fused expert weight, so no tensor pins the rank exactly and the check above is "
            "the weaker one"
        )

    parameters_per_module: dict[str, int] = {}
    for target in depths:
        matching = sum(1 for parent in parents if target == parent or target.endswith(f".{parent}"))
        parameters_per_module[target] = matching or (len(config.get("target_parameters") or []) if bare else 1)
    for target, seen in sorted(depths.items()):
        if len(seen) > parameters_per_module[target] and parameters_per_module[target] > 1:
            problems.append(
                f"{target} carries {len(seen)} nested wrappers but the configuration names "
                f"{parameters_per_module[target]} parameters on it; which tensor adapts which parameter is "
                "not recoverable from the names"
            )
    if any(len(seen) > 1 for seen in depths.values()):
        notes.append(
            "more than one parameter is adapted on a single module, so the exported names are positional and "
            "only mean the same thing under this exact target_parameters order"
        )

    return AdapterValidation(
        ok=not problems,
        adapter_name=adapters[0] if len(adapters) == 1 else "",
        tensors=len(lora),
        parameters=sum(_numel(shapes[name]) for name in lora),
        targets=targets,
        module_target_ranks=_histogram(module_ranks),
        parameter_target_widths=_histogram(fused_widths),
        expert_counts=expert_counts,
        targets_by_kind=kinds,
        problems=problems,
        notes=notes,
    )


def _histogram(values: Iterable[int]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return counts


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------


@dataclass
class ExportReport:
    schema: str
    action: str
    source: str
    destination: str
    written: bool
    dry_run: bool
    reason: str = ""
    tag: str = ""
    dtype: str = ""
    tensors: int = 0
    parameters: int = 0
    dropped: list[str] = field(default_factory=list)
    adapter_config: str = ""
    adapter_config_sha256: str = ""
    deepspeed_version: str | None = None
    sidecars: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    validation: dict[str, Any] = field(default_factory=dict)
    layout: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _provenance(report: ExportReport) -> dict[str, Any]:
    body = report.as_dict()
    body.pop("dry_run", None)
    body["written_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    return body


IDENTITY_FIELDS = ("action", "source", "tag", "adapter_config_sha256", "dtype")


def _already_exported(destination: Path, report: ExportReport) -> bool:
    """Accept a finished export of the same thing; refuse anything else in the way.

    Nothing here removes a file. A destination that holds a different export, or
    holds something that is not an export at all, is reported and left alone.
    """
    if not destination.exists():
        return False
    if not destination.is_dir():
        raise CheckpointExportError(f"{destination} exists and is not a directory")
    if not any(destination.iterdir()):
        return False
    marker = destination / PROVENANCE_NAME
    if not marker.is_file():
        raise CheckpointExportError(
            f"{destination} is not empty and holds no {PROVENANCE_NAME}, so it was not written here; "
            "move it aside before exporting into it"
        )
    previous = json.loads(marker.read_text())
    differing = [key for key in IDENTITY_FIELDS if previous.get(key) != getattr(report, key)]
    if differing:
        raise CheckpointExportError(
            f"{destination} already holds an export that differs in {differing} "
            f"(recorded {[previous.get(key) for key in differing]}); move it aside before replacing it"
        )
    return True


def _write_directory(destination: Path, write: Callable[[Path], None]) -> None:
    """Build the export beside its destination and rename it into place.

    A half-written adapter directory that a later run would have to distinguish
    from a finished one is the failure mode worth spending a rename on.
    """
    partial = destination.parent / f"{destination.name}.partial-{os.getpid()}"
    if partial.exists():
        raise CheckpointExportError(f"{partial} already exists; a previous export may still be running")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial.mkdir()
    try:
        write(partial)
        os.rename(partial, destination)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise


def _resolve_dtype(name: str):
    import torch  # noqa: PLC0415

    if name not in DTYPES:
        raise CheckpointExportError(f"dtype {name!r} is not one of {DTYPES}")
    return getattr(torch, name)


def consolidate_lora_tensors(report: ZeroStepReport) -> tuple[dict[str, Any], list[str], str | None]:
    """Rebuild the trained tensors from the optimizer partitions and keep the LoRA ones.

    ``exclude_frozen_parameters`` is what makes this cheap: under LoRA the only
    trainable parameters are the adapters, so the 7B frozen base is never
    reassembled. Peak memory is still set by the model-states shard, which
    deepspeed unpickles whole to find the parameter order.
    """
    import contextlib  # noqa: PLC0415

    import deepspeed  # noqa: PLC0415
    from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint  # noqa: PLC0415

    report.require_consolidatable()
    try:
        # deepspeed narrates the reconstruction on stdout. That belongs beside
        # the job's other progress rather than in the middle of the report a
        # calling script parses.
        with contextlib.redirect_stdout(sys.stderr):
            state = get_fp32_state_dict_from_zero_checkpoint(
                report.checkpoint_root, tag=report.tag, exclude_frozen_parameters=True
            )
    except ModuleNotFoundError as exc:
        raise CheckpointExportError(
            f"{report.path}: reading this checkpoint needs {exc.name!r}, which the training run's client state "
            "refers to; run this from the repository root with the training environment"
        ) from exc
    tensors = {name: value for name, value in state.items() if parse_lora_name(name)}
    dropped = sorted(name for name in state if name not in tensors)
    return tensors, dropped, getattr(deepspeed, "__version__", None)


def export_adapter(
    step: Path | str,
    destination: Path | str,
    *,
    adapter_config: AdapterConfigSource,
    dtype: str = DEFAULT_DTYPE,
    dry_run: bool = False,
    allow_pickle: bool = False,
    copy_sidecars: bool = True,
) -> ExportReport:
    """Consolidate one ZeRO step into a PEFT adapter directory the tracer can load."""
    step = Path(step)
    destination = Path(destination)
    layout = inspect_zero_step(step).require_consolidatable()
    report = ExportReport(
        schema=SCHEMA,
        action="export_adapter",
        source=str(step),
        destination=str(destination),
        written=False,
        dry_run=dry_run,
        tag=layout.tag,
        dtype=dtype,
        adapter_config=adapter_config.path,
        adapter_config_sha256=adapter_config.sha256,
        layout={key: value for key, value in layout.as_dict().items() if key not in {"parameter_index"}},
    )

    if dry_run:
        report.reason = "dry run: nothing was written"
        if allow_pickle:
            index = read_parameter_index(layout, allow_pickle=True)
            validation = validate_adapter(adapter_config.config, index["shapes"])
            report.validation = validation.as_dict()
            report.tensors = validation.tensors
            report.parameters = validation.parameters
            validation.require_ok(str(step))
        else:
            report.validation = {
                "ok": None,
                "reason": (
                    "the parameter index was not read, so the configuration was not checked against the "
                    "tensors; pass --allow-pickle to read it"
                ),
            }
        return report

    if _already_exported(destination, report):
        report.reason = "an export of this step with this configuration is already here"
        return report

    tensors, dropped, version = consolidate_lora_tensors(layout)
    shapes = {name: tuple(value.shape) for name, value in tensors.items()}
    validation = validate_adapter(adapter_config.config, shapes).require_ok(str(step))
    torch_dtype = _resolve_dtype(dtype)
    payload = {
        peft_state_dict_key(name, validation.adapter_name): value.detach().to(torch_dtype).contiguous().clone()
        for name, value in sorted(tensors.items())
    }
    if len(payload) != len(tensors):
        raise AmbiguousCheckpoint(
            f"{step}: dropping the adapter name collapsed {len(tensors)} tensors into {len(payload)} keys"
        )

    sidecars = _sidecar_paths(Path(adapter_config.path).parent) if copy_sidecars else []
    report.tensors = len(payload)
    report.parameters = validation.parameters
    report.dropped = dropped
    report.deepspeed_version = version
    report.sidecars = [path.name for path in sidecars]
    report.validation = validation.as_dict()

    def write(target: Path) -> None:
        from safetensors.torch import save_file  # noqa: PLC0415

        save_file(payload, str(target / ADAPTER_WEIGHTS_NAME), metadata={"format": "pt"})
        (target / ADAPTER_CONFIG_NAME).write_text(json.dumps(adapter_config.config, indent=2) + "\n")
        for path in sidecars:
            shutil.copyfile(path, target / path.name)
        (target / PROVENANCE_NAME).write_text(json.dumps(_provenance(report), indent=2, default=str) + "\n")

    _write_directory(destination, write)
    report.written = True
    return report


def _sidecar_paths(directory: Path) -> list[Path]:
    return [directory / name for name in SIDECAR_NAMES if (directory / name).is_file()]


# --------------------------------------------------------------------------
# the dense arm: a PEFT adapter that arrived as a zip
# --------------------------------------------------------------------------


@dataclass
class AdapterZipReport:
    path: str
    kind: str
    consolidatable: bool
    reason: str = ""
    root: str = ""
    members: list[str] = field(default_factory=list)
    base_model: str | None = None
    weight_file: str = ""
    problems: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def require_consolidatable(self) -> AdapterZipReport:
        if not self.consolidatable:
            raise AmbiguousCheckpoint(f"{self.path}: {self.reason}")
        return self


def _escapes(member: str) -> bool:
    path = Path(member)
    return path.is_absolute() or ".." in path.parts or member.startswith(("/", "\\"))


def inspect_adapter_zip(path: Path | str) -> AdapterZipReport:
    """Read a zipped PEFT adapter without extracting it."""
    path = Path(path)
    if not path.is_file():
        return AdapterZipReport(path=str(path), kind="missing", consolidatable=False, reason="path does not exist")
    if not zipfile.is_zipfile(path):
        return AdapterZipReport(path=str(path), kind="not_a_zip", consolidatable=False, reason="not a zip archive")
    with zipfile.ZipFile(path) as archive:
        members = archive.namelist()
        configs = [name for name in members if name.endswith(ADAPTER_CONFIG_NAME)]
        problems: list[str] = []
        escaping = sorted(name for name in members if _escapes(name))[:3]
        if escaping:
            problems.append(f"these members would be written outside the destination: {escaping}")
        if not configs:
            problems.append(f"no {ADAPTER_CONFIG_NAME} inside, so this is not a PEFT adapter")
        elif len(configs) > 1:
            problems.append(f"{len(configs)} adapters inside {configs[:4]}; extract the one you want yourself")
        root = configs[0].rsplit(ADAPTER_CONFIG_NAME, 1)[0] if len(configs) == 1 else ""
        weights = [
            name
            for name in members
            if name.startswith(root) and Path(name).name in {ADAPTER_WEIGHTS_NAME, "adapter_model.bin"}
        ]
        if configs and not weights:
            problems.append("the adapter has a configuration but no adapter_model weights beside it")
        base_model = None
        if len(configs) == 1:
            body = json.loads(archive.read(configs[0]))
            base_model = body.get("base_model_name_or_path")
            if str(body.get("peft_type", "")).upper() != "LORA":
                problems.append(f"peft_type {body.get('peft_type')!r} is not supported here")
        return AdapterZipReport(
            path=str(path),
            kind="peft_adapter_zip",
            consolidatable=not problems,
            reason="; ".join(problems),
            root=root,
            members=members[:40],
            base_model=base_model,
            weight_file=Path(weights[0]).name if weights else "",
            problems=problems,
        )


def stage_adapter_zip(archive_path: Path | str, destination: Path | str, *, dry_run: bool = False) -> ExportReport:
    """Extract a zipped adapter into the export layout, flattening its single root."""
    archive_path = Path(archive_path)
    destination = Path(destination)
    layout = inspect_adapter_zip(archive_path).require_consolidatable()
    with zipfile.ZipFile(archive_path) as archive:
        wanted = [
            name
            for name in archive.namelist()
            if name.startswith(layout.root) and not name.endswith("/") and name != layout.root
        ]
        source = read_adapter_config_from_zip(archive, layout.root)
    report = ExportReport(
        schema=SCHEMA,
        action="stage_adapter_zip",
        source=str(archive_path),
        destination=str(destination),
        written=False,
        dry_run=dry_run,
        tag=layout.root.rstrip("/"),
        dtype="unchanged",
        adapter_config=f"{archive_path}!{layout.root}{ADAPTER_CONFIG_NAME}",
        adapter_config_sha256=source.sha256,
        files=sorted(Path(name).name for name in wanted),
        validation={
            "ok": True,
            "base_model": layout.base_model,
            "note": "an adapter that was already exported is copied rather than rebuilt",
        },
    )
    if dry_run:
        report.reason = "dry run: nothing was written"
        return report
    if _already_exported(destination, report):
        report.reason = "this adapter is already staged here"
        return report

    def write(target: Path) -> None:
        with zipfile.ZipFile(archive_path) as handle:
            for name in wanted:
                relative = Path(name[len(layout.root) :])
                out = target / relative
                out.parent.mkdir(parents=True, exist_ok=True)
                with handle.open(name) as reader, out.open("wb") as writer:
                    shutil.copyfileobj(reader, writer)
        (target / PROVENANCE_NAME).write_text(json.dumps(_provenance(report), indent=2, default=str) + "\n")

    _write_directory(destination, write)
    report.written = True
    return report


def read_adapter_config_from_zip(archive: zipfile.ZipFile, root: str) -> AdapterConfigSource:
    name = f"{root}{ADAPTER_CONFIG_NAME}"
    raw = archive.read(name)
    return AdapterConfigSource(path=name, sha256=hashlib.sha256(raw).hexdigest(), config=json.loads(raw))


# --------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------


def inspect_checkpoint(path: Path | str, *, allow_pickle: bool = False, read_index: bool = False):
    """Classify whatever is at ``path``: a ZeRO step, a zipped adapter, or neither."""
    path = Path(path)
    if path.is_file() and zipfile.is_zipfile(path):
        return inspect_adapter_zip(path)
    report = inspect_zero_step(path)
    if read_index and report.consolidatable:
        index = read_parameter_index(report, allow_pickle=allow_pickle)
        index.pop("shapes", None)
        report.parameter_index = index
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    inspect = commands.add_parser("inspect", help="classify checkpoints without reading a tensor")
    inspect.add_argument("paths", type=Path, nargs="+")
    inspect.add_argument(
        "--read-parameter-index", action="store_true", help="also read names and shapes (needs torch)"
    )
    inspect.add_argument("--allow-pickle", action="store_true", help="permit unpickling a training shard")
    inspect.add_argument("--require-consolidatable", action="store_true", help="exit non-zero on a refused layout")

    export = commands.add_parser("export", help="consolidate one ZeRO step into a PEFT adapter directory")
    export.add_argument("--step", type=Path, required=True)
    export.add_argument("--out", type=Path, required=True)
    export.add_argument("--adapter-config", type=Path, help="the adapter_config.json PEFT wrote for this run")
    export.add_argument("--adapter-config-search", type=Path, nargs="*", default=[], help="directories to search")
    export.add_argument("--dtype", default=DEFAULT_DTYPE, choices=DTYPES)
    export.add_argument("--dry-run", action="store_true")
    export.add_argument("--allow-pickle", action="store_true")
    export.add_argument("--no-sidecars", action="store_true", help="do not copy the tokenizer files")

    stage = commands.add_parser("stage-zip", help="extract a zipped PEFT adapter into the export layout")
    stage.add_argument("--archive", type=Path, required=True)
    stage.add_argument("--out", type=Path, required=True)
    stage.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "inspect":
        reports = [
            inspect_checkpoint(path, allow_pickle=args.allow_pickle, read_index=args.read_parameter_index)
            for path in args.paths
        ]
        for report in reports:
            print(json.dumps(report.as_dict(), indent=2, default=str))
        refused = [report for report in reports if not report.consolidatable]
        if refused and args.require_consolidatable:
            raise SystemExit(
                f"{len(refused)} of {len(reports)} checkpoints cannot be consolidated: "
                + "; ".join(f"{report.path}: {report.reason}" for report in refused)
            )
        return

    if args.command == "stage-zip":
        report = stage_adapter_zip(args.archive, args.out, dry_run=args.dry_run)
    else:
        step = inspect_zero_step(args.step).require_consolidatable()
        if args.adapter_config:
            source = read_adapter_config(args.adapter_config)
        else:
            source = find_adapter_config(args.adapter_config_search, step=step.step)
        report = export_adapter(
            args.step,
            args.out,
            adapter_config=source,
            dtype=args.dtype,
            dry_run=args.dry_run,
            allow_pickle=args.allow_pickle,
            copy_sidecars=not args.no_sidecars,
        )
    print(json.dumps(report.as_dict(), indent=2, default=str))


if __name__ == "__main__":
    try:
        main()
    except CheckpointExportError as error:
        # Exit 3, distinct from argparse's 2, so a job script can tell a refusal
        # from a typo without parsing the message.
        print(f"refused: {error}", file=sys.stderr)
        raise SystemExit(3) from error
