"""Check every --flag in .edullm/run.yaml against the fields the trainers actually parse.

Reads the dataclasses with ast rather than importing them, so this runs on a laptop with
no torch. Catches the one failure this smoke test is most likely to hit: a flag that looks
right, is spelled wrong, and is only discovered after a machine has been paid for.
"""

from __future__ import annotations

import ast
import pathlib
import re
import shlex
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _module(path: str) -> ast.Module:
    return ast.parse((ROOT / path).read_text())


def _classes(tree: ast.Module) -> dict[str, ast.ClassDef]:
    return {n.name: n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}


def _own_fields(cls: ast.ClassDef) -> set[str]:
    return {
        stmt.target.id for stmt in cls.body if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
    }


def fields_of(entry: str, class_name: str, extra_modules: list[str]) -> set[str]:
    """Every field on a dataclass and on the bases it inherits, across the given modules."""
    pool: dict[str, ast.ClassDef] = {}
    for path in [entry, *extra_modules]:
        pool.update(_classes(_module(path)))

    seen: set[str] = set()
    out: set[str] = set()

    def walk(name: str) -> None:
        if name in seen or name not in pool:
            return
        seen.add(name)
        cls = pool[name]
        out.update(_own_fields(cls))
        for base in cls.bases:
            if isinstance(base, ast.Name):
                walk(base.id)

    walk(class_name)
    return out


def argparse_flags(path: str) -> set[str]:
    tree = _module(path)
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            for arg in node.args:
                if isinstance(arg, ast.Constant) and str(arg.value).startswith("--"):
                    out.add(str(arg.value)[2:])
    return out


SHARED = ["open_instruct/utils.py", "open_instruct/dataset_transformation.py", "open_instruct/model_utils.py"]

# dpo_utils imports its bases from olmo_core_utils, so that module goes last and wins the
# names it shares with model_utils and dataset_transformation.
DPO_MODULES = [*SHARED, "open_instruct/dpo_utils.py", "open_instruct/olmo_core_utils.py"]

KNOWN = {
    "open_instruct/finetune.py": fields_of("open_instruct/finetune.py", "FlatArguments", SHARED)
    | fields_of("open_instruct/utils.py", "TokenizerConfig", SHARED),
    "open_instruct/dpo_tune_cache.py": fields_of("open_instruct/dpo_utils.py", "DPOExperimentConfig", DPO_MODULES)
    | fields_of("open_instruct/utils.py", "TokenizerConfig", SHARED),
}

# accelerate launch's own flags, consumed before the program sees them.
LAUNCHER = {"mixed_precision", "num_processes", "config_file", "use_deepspeed", "deepspeed_multinode_launcher"}


def semantic_errors(spec: dict) -> list[str]:
    """Check cross-flag invariants that field-name validation cannot prove."""
    script = shlex.split(spec["command"])[-1]
    prep, rest = script.split('echo "=== STAGE 1/2', maxsplit=1)
    sft, dpo = rest.split('echo "=== STAGE 2/2', maxsplit=1)

    no_side_effects = {
        "--push_to_hub false": "must not make an unauthenticated Hugging Face write",
        "--try_launch_beaker_eval_jobs false": "must not launch AI2 Beaker jobs",
        "--try_auto_save_to_beaker false": "must not write to an AI2 Beaker dataset",
    }
    errors = [
        f"{stage_name} {detail}"
        for stage_name, stage in (("SFT", sft), ("DPO", dpo))
        for text, detail in no_side_effects.items()
        if text not in stage
    ]

    # A full fine-tune of 6.9B needs ~110GB of weights, gradients and Adam state. The
    # cards are 40GB, so every rank must shard all three or the first step runs out.
    for stage_name, stage in (("SFT", sft), ("DPO", dpo)):
        if "--use_lora" in stage:
            errors.append(f"{stage_name} must be a full fine-tune; this spec no longer trains LoRA adapters")
        if "--num_processes 8" not in stage:
            errors.append(f"{stage_name} must use all eight ranks to hold a sharded full fine-tune")
        if "--mixed_precision bf16" not in stage:
            errors.append(f"{stage_name} must name bf16 so the platform precision guard can see it")
    if "deepspeed_zero3.yaml" not in sft:
        errors.append("SFT must shard with ZeRO-3; 110GB of optimizer state does not fit a 40GB card")
    if "--zero_stage 3" not in dpo:
        errors.append("DPO builds its own DeepSpeed config and must be told to shard with ZeRO-3")

    if 't.bos_token=t.eos_token; t.save_pretrained(\\"$OUT/tokenizer\\")' not in prep:
        errors.append("the base OLMoE tokenizer must materialize its intended BOS alias before SFT")
    if '--tokenizer_name_or_path "$OUT/tokenizer"' not in sft:
        errors.append("SFT must load the tokenizer with the materialized BOS alias")
    if "--dataset_mixer_list allenai/tulu-3-sft-personas-algebra 64" not in sft:
        errors.append("the SFT mixer count, not the dead --max_train_samples field, must bound the smoke")
    if "--max_train_samples" in sft:
        errors.append("SFT's --max_train_samples field is dead code; set the mixer count instead")

    # A full fine-tune writes a dense checkpoint, so DPO reads SFT's output directly and
    # inherits the tokenizer saved beside it. There is no adapter left to merge.
    if '--model_name_or_path "$OUT/sft"' not in dpo:
        errors.append("DPO must start from the SFT checkpoint that stage one wrote")
    if "merge_lora" in script:
        errors.append("a full fine-tune writes dense weights; there is no adapter to merge")

    # dpo_norm needs reference logprobs, cached under a path that defaults to AI2 Weka.
    if 'REFERENCE_LOGPROBS_CACHE_PATH="$OUT/ref_cache"' not in prep:
        errors.append("the reference-logprob cache must be redirected off its AI2 Weka default")

    # The platform shows the last fifty lines only, and torchrun's eight-rank epilogue
    # fills them by itself, so each launcher's own error has to be replayed after it.
    for stage_name, stage, log in (("SFT", sft, "sft"), ("DPO", dpo, "dpo")):
        if f'> "$OUT/{log}.log" 2>&1' not in stage:
            errors.append(f"{stage_name} must capture its own output so a traceback survives torchrun's epilogue")
        if f'tail -40 "$OUT/{log}.log"' not in stage:
            errors.append(f"{stage_name} must replay its last lines when it fails")

    # The platform reads the launcher out of the shlex-split command, where `;` only ends a
    # command when it is its own word. `echo "..."; accelerate launch` folds to one token
    # ending in `;`, which leaves `accelerate` an argument to echo and the run refused as
    # process_per_device -- no launcher found, so one process asked for on eight GPUs.
    words = shlex.split(script)
    operators = {";", "&&", "||", "|", "&", "(", ")"}
    launched = [i for i, w in enumerate(words) if w == "accelerate"]
    if len(launched) != 2:
        errors.append("expected exactly two accelerate invocations, one per stage")
    for i in launched:
        if i != 0 and words[i - 1] not in operators:
            errors.append(
                f"`accelerate` at word {i} follows {words[i - 1]!r} rather than an operator, "
                "so the launch guard cannot see it; put a space before the preceding `;`"
            )

    if spec.get("suggested_compute") != "gpu-8xa100":
        errors.append("a full fine-tune of 6.9B needs the eight-card A100 node and its 500GiB root disk")
    return errors


def main() -> int:
    spec = yaml.safe_load((ROOT / ".edullm/run.yaml").read_text())
    words = shlex.split(spec["command"])
    # The command is `bash -lc '<script>'`; the script is the last word.
    words = shlex.split(words[-1].replace("&&", " ").replace(";", " "))

    program = None
    bad: list[tuple[str, str]] = []
    checked = 0
    for word in words:
        if re.fullmatch(r"open_instruct/\w+\.py", word):
            program = word
            continue
        if not word.startswith("--"):
            continue
        flag = word[2:]
        if flag in LAUNCHER:
            continue
        if program is None:
            continue
        checked += 1
        if flag not in KNOWN[program]:
            bad.append((program, flag))

    print(f"checked {checked} flags across {len(KNOWN)} programs")
    for program, flag in bad:
        near = sorted(f for f in KNOWN[program] if flag.split("_")[0] in f)[:4]
        print(f"  UNKNOWN  {program}  --{flag}" + (f"   near: {near}" if near else ""))
    semantic = semantic_errors(spec)
    for detail in semantic:
        print(f"  UNSAFE   {detail}")
    if not bad and not semantic:
        print("every flag resolves to a parsed field")
        print("every smoke-test safety invariant holds")
    return 1 if bad or semantic else 0


if __name__ == "__main__":
    sys.exit(main())
