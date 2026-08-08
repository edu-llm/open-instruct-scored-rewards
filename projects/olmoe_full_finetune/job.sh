#!/usr/bin/env bash
set -euo pipefail

SOURCE_PROJECT_DIR="${SOURCE_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
SCRATCH="${SCRATCH:-/orcd/scratch/orcd/013/zsophia}"
REPO="${REPO:-$SCRATCH/open-instruct}"
VENV="${VENV:-$SCRATCH/venv_oi}"
export SOURCE_PROJECT_DIR SCRATCH REPO VENV

# Protect running jobs from edits or checkouts that change a script while bash is
# still reading it by byte offset.
if [[ -z "${OLMOE_JOB_SNAPSHOT:-}" ]]; then
    OLMOE_JOB_SNAPSHOT="${TMPDIR:-/tmp}/olmoe_full_finetune_${SLURM_JOB_ID:-$$}"
    mkdir -p "$OLMOE_JOB_SNAPSHOT"
    cp "$SOURCE_PROJECT_DIR/job.sh" "$SOURCE_PROJECT_DIR/train.sh" "$SOURCE_PROJECT_DIR/midtrain.py" \
        "$SOURCE_PROJECT_DIR/stage3_midtrain_accelerate.conf" \
        "$SOURCE_PROJECT_DIR/stage3_offloading_accelerate.conf" "$OLMOE_JOB_SNAPSHOT/"
    export OLMOE_JOB_SNAPSHOT MIDTRAIN_SCRIPT="$OLMOE_JOB_SNAPSHOT/midtrain.py"
    exec bash "$OLMOE_JOB_SNAPSHOT/job.sh"
fi

if [[ "${DEEPSPEED_CONFIG_FILE:-}" == projects/olmoe_full_finetune/* ]]; then
    DEEPSPEED_CONFIG_FILE="$OLMOE_JOB_SNAPSHOT/${DEEPSPEED_CONFIG_FILE##*/}"
fi
export DEEPSPEED_CONFIG_FILE="${DEEPSPEED_CONFIG_FILE:-$OLMOE_JOB_SNAPSHOT/stage3_midtrain_accelerate.conf}"

echo "started $(date -Is) on $(hostname), job ${SLURM_JOB_ID:-none}"

set +u
command -v module >/dev/null 2>&1 || source /usr/share/lmod/lmod/init/bash
module load cuda/13.0.1
source "$VENV/bin/activate"
set -u

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$SCRATCH/hf-cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$SCRATCH/xdg-cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$SCRATCH/triton-cache/$(hostname -s)}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$SCRATCH/inductor-cache/$(hostname -s)}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TOKENIZERS_PARALLELISM=false
export NCCL_CUMEM_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

PY_INCLUDE="${PY_INCLUDE:-$HOME/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/include/python3.12}"
if [[ -f "$PY_INCLUDE/Python.h" ]]; then
    export CPATH="$PY_INCLUDE${CPATH:+:$CPATH}"
fi

export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DIR="${WANDB_DIR:-$SCRATCH/wandb}"
export WANDB_PROJECT="${WANDB_PROJECT:-olmoe-midtraining}"
export WANDB_ENTITY="${WANDB_ENTITY:-eduLLM}"
mkdir -p "$WANDB_DIR"

commit="$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)"
dirty="$(git -C "$REPO" status --porcelain 2>/dev/null | grep -cvE '^\?\? projects/olmoe_full_finetune/runs/' || true)"
export WANDB_TAGS="commit-$commit,midtraining,olmoe,cosine,lr-${LR:-unset},seed-${SEED:-42}"
export WANDB_NOTES="code $commit (uncommitted files: $dirty) | model 6d84c48581ece | dolmino a319f19eef1e | global batch ${GLOBAL_BATCH_SIZE:-1024} | $(hostname)"
echo "provenance: $WANDB_NOTES"
if [[ "$dirty" != "0" ]]; then
    echo "WARNING: $dirty files differ from commit $commit; the run is not reproducible from that commit" >&2
fi

exec bash "$OLMOE_JOB_SNAPSHOT/train.sh"
