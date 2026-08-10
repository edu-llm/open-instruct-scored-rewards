#!/bin/bash
#SBATCH -J atlas_ckpt_export
# NO GPU AND NOT PREEMPTABLE, WHICH IS THE WHOLE POINT OF SPLITTING THIS OUT.
# Consolidating a ZeRO state is a CPU job: it reads optimizer partitions, adds
# them up and writes a few hundred megabytes of adapter. Asking for an H100 to
# do it puts the job behind every training run in the queue for no benefit, and
# mit_preemptable would restart it halfway through a 27 GB read.
#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --time=02:00:00
# THE MEMORY IS SET BY THE SHARD, NOT BY THE EXPORT. deepspeed unpickles
# mp_rank_00_model_states.pt whole to recover the parameter order, and under
# ZeRO-2 that file holds the bf16 module state dict *and* the frozen fragments -
# about twice the model. `STAGE=inspect` prints the exact bytes and a
# recommended_mem_gb for each step; raise this if it says more.
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --open-mode=append
#SBATCH -o logs/atlas_ckpt_export_%A.out
#SBATCH -e logs/atlas_ckpt_export_%A.err
#
# Consolidate the pedagogy-RL checkpoints into what compare_rl_checkpoints.sbatch
# looks for, and refuse anything ambiguous rather than exporting a guess.
#
#   STAGE=inspect   classify every checkpoint and stop  (no torch, no GPU)
#   STAGE=export    inspect, then write the exports     (default)
#   DRY_RUN=1       print what would be written and write nothing
#
# EXACT COMMANDS ON ORCD, in the order they are useful:
#
#   ssh orcd-login
#   cd /orcd/scratch/orcd/013/zsophia/open-instruct
#
#   # 1. look at the checkpoints. Runs on the login node: no GPU, no torch, and
#   #    it opens no shard, so it costs a directory listing.
#   STAGE=inspect bash projects/learning_science_semantic_atlas/scripts/export_rl_checkpoints.sbatch
#
#   # 2. rehearse the export. Resolves the adapter configuration and checks it
#   #    against the tensors in the shard; still writes nothing.
#   DRY_RUN=1 bash projects/learning_science_semantic_atlas/scripts/export_rl_checkpoints.sbatch
#
#   # 3. run it.
#   sbatch projects/learning_science_semantic_atlas/scripts/export_rl_checkpoints.sbatch
#
#   # 4. then the comparison, which now finds a loadable export for each step.
#   SPLIT=<your split> sbatch projects/learning_science_semantic_atlas/scripts/compare_rl_checkpoints.sbatch
#
# WHERE THE ADAPTER CONFIGURATION COMES FROM, AND WHY IT IS NOT OPTIONAL. A ZeRO
# state records tensors and their names; it does not record that
# mlp.experts.gate_up_proj was adapted at rank 2 behind a frozen router. PEFT
# wrote that beside the HF-format saves this run also produced, under
# output/pedagogy_olmoe_checkpoints/step_<N>. ADAPTER_CONFIG_SEARCH lists where
# to look; two configurations whose bytes differ stop the export instead of
# being averaged into a coin flip. If the search finds nothing, pass
# ADAPTER_CONFIG=/path/to/adapter_config.json explicitly.
#
# NOTHING HERE DELETES OR OVERWRITES. Each export is built in a sibling
# .partial-<pid> directory and renamed into place, an export that is already
# there and matches is left alone, and one that differs stops the job.
set -euo pipefail

SCRATCH=${SCRATCH:-/orcd/scratch/orcd/013/zsophia}
REPO=${REPO:-$SCRATCH/open-instruct}
VENV=${VENV:-$SCRATCH/venv_oi}
PROJECT=$REPO/projects/learning_science_semantic_atlas

CKPT_ROOT=${CKPT_ROOT:-$SCRATCH/ckpt/pedagogy_olmoe}
EXPORT_ROOT=${EXPORT_ROOT:-$SCRATCH/ckpt/pedagogy_olmoe_exported}
STEPS=${STEPS:-180 190 200}
EXP=${EXP:-pedagogy_olmoe}
ADAPTER_CONFIG=${ADAPTER_CONFIG:-}
ADAPTER_CONFIG_SEARCH=${ADAPTER_CONFIG_SEARCH:-"$REPO/output/${EXP}_checkpoints $REPO/output/$EXP $CKPT_ROOT"}

# The dense arm needs no consolidation - it is already a PEFT adapter - but the
# comparison wants a directory, so it is unpacked through the same door.
DENSE_ZIP=${DENSE_ZIP:-$REPO/ckpt/pedagogy-tutor-armF.zip}
DENSE_EXPORT=${DENSE_EXPORT:-$SCRATCH/ckpt/pedagogy_tutor_armF_exported}

# fp32 keeps the master weights ZeRO reconstructs; bfloat16 reproduces the
# trainer's forward exactly. The difference is far below the effects being
# measured, and fp32 is the one that loses nothing.
DTYPE=${DTYPE:-float32}
STAGE=${STAGE:-export}
DRY_RUN=${DRY_RUN:-0}
# Reading parameter names out of a shard executes pickle from that file, because
# DeepSpeed stores the trainer's client state beside its tensors. It is on for
# the dry run, where checking the configuration against the tensors is the
# entire point, and it is a deliberate flag rather than a silent fallback.
ALLOW_PICKLE=${ALLOW_PICKLE:-1}

export HF_HOME=${HF_HOME:-$SCRATCH/hf-cache}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=false

PYTHON=${PYTHON:-$VENV/bin/python}
if [ ! -x "$PYTHON" ]; then
    echo "python $PYTHON is not executable; set VENV or PYTHON" >&2
    exit 2
fi

echo "started $(date -Is) on $(hostname), job ${SLURM_JOB_ID:-none}"
echo "ckpt_root=$CKPT_ROOT export_root=$EXPORT_ROOT steps='$STEPS' stage=$STAGE dry_run=$DRY_RUN dtype=$DTYPE"

cd "$REPO"
mkdir -p "$REPO/logs"

export_checkpoint=("$PYTHON" -m projects.learning_science_semantic_atlas.export_checkpoint)
describe=("$PYTHON" -m projects.learning_science_semantic_atlas.checkpoint_delta --describe)

# ---------------------------------------------------------------- inspect
# Every step is classified first, and the classification is a directory listing:
# no torch, no deepspeed, no shard opened. A layout this refuses is reported with
# its reason and does not consume the hour a consolidation costs.
inspectable=()
for step in $STEPS; do
    raw="$CKPT_ROOT/global_step$step"
    echo "--- global_step$step ---"
    if [ ! -d "$raw" ]; then
        echo "no checkpoint at $raw" >&2
        continue
    fi
    if "${export_checkpoint[@]}" inspect "$raw" --require-consolidatable; then
        inspectable+=("$step")
    else
        echo "global_step$step cannot be consolidated; see the reason above" >&2
    fi
done

# The dense arm is independent of the OLMoE steps, so a problem with the zip
# reports itself and leaves the rest of the job alone.
dense=0
if [ -f "$DENSE_ZIP" ]; then
    echo "--- $(basename "$DENSE_ZIP") ---"
    if "${export_checkpoint[@]}" inspect "$DENSE_ZIP" --require-consolidatable; then
        dense=1
    else
        echo "$DENSE_ZIP cannot be staged; see the reason above" >&2
    fi
fi

if [ ${#inspectable[@]} -eq 0 ] && [ "$dense" = "0" ]; then
    echo "nothing to export" >&2
    exit 3
fi
if [ "$STAGE" = "inspect" ]; then
    echo "inspection only; rerun without STAGE=inspect to write the exports"
    exit 0
fi

# ----------------------------------------------------------------- export
search=()
for directory in $ADAPTER_CONFIG_SEARCH; do
    [ -d "$directory" ] && search+=("$directory")
done
if [ -z "$ADAPTER_CONFIG" ] && [ ${#search[@]} -eq 0 ]; then
    echo "no directory in ADAPTER_CONFIG_SEARCH exists ($ADAPTER_CONFIG_SEARCH)." >&2
    echo "point ADAPTER_CONFIG at the adapter_config.json PEFT wrote for $EXP." >&2
    exit 2
fi

exported=()
for step in ${inspectable[@]+"${inspectable[@]}"}; do
    args=(
        export
        --step "$CKPT_ROOT/global_step$step"
        --out "$EXPORT_ROOT/global_step$step"
        --dtype "$DTYPE"
    )
    if [ -n "$ADAPTER_CONFIG" ]; then
        args+=(--adapter-config "$ADAPTER_CONFIG")
    else
        args+=(--adapter-config-search "${search[@]}")
    fi
    [ "$ALLOW_PICKLE" = "1" ] && args+=(--allow-pickle)
    [ "$DRY_RUN" = "1" ] && args+=(--dry-run)
    echo "--- exporting global_step$step ---"
    "${export_checkpoint[@]}" "${args[@]}"
    exported+=("$EXPORT_ROOT/global_step$step")
done

if [ "$dense" = "1" ]; then
    echo "--- staging $(basename "$DENSE_ZIP") ---"
    dense_args=(stage-zip --archive "$DENSE_ZIP" --out "$DENSE_EXPORT")
    [ "$DRY_RUN" = "1" ] && dense_args+=(--dry-run)
    "${export_checkpoint[@]}" "${dense_args[@]}"
    exported+=("$DENSE_EXPORT")
fi

if [ "$DRY_RUN" = "1" ]; then
    echo "dry run complete: nothing was written"
    exit 0
fi

# ------------------------------------------------------------------ accept
# The acceptance test is the classifier the tracer itself consults. An export
# this rejects would fail inside compare_rl_checkpoints.sbatch instead, after
# that job had already queued for a GPU.
echo "--- checking every export against the tracer's own classifier ---"
"${describe[@]}" ${exported[@]+"${exported[@]}"} --require-loadable

echo "finished $(date -Is); exports are in $EXPORT_ROOT"
echo "next: SPLIT=<split> sbatch $PROJECT/scripts/compare_rl_checkpoints.sbatch"
