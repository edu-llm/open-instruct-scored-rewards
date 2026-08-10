#!/bin/bash
#SBATCH -J atlas_trace
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --time=04:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o logs/atlas_trace_%A.out
#SBATCH -e logs/atlas_trace_%A.err
#
# Teacher-forced replay tracing for one checkpoint over one split.
#
# Runs three ways:
#   DRY_RUN=1 bash trace_atlas.sbatch   tokenizer only, no GPU, prints the slot plan
#   bash trace_atlas.sbatch             foreground on an interactive node
#   sbatch trace_atlas.sbatch           batch
#
# Dense OLMo-2 and OLMoE both work. Routing arrays only appear for OLMoE; the
# dense arm still writes activations, so the analysis is the same command.
set -euo pipefail

SCRATCH=${SCRATCH:-/orcd/scratch/orcd/013/zsophia}
REPO=${REPO:-$SCRATCH/open-instruct}
VENV=${VENV:-$SCRATCH/venv_oi}
PROJECT=$REPO/projects/learning_science_semantic_atlas

# A hub id, or a local directory holding a whole model such as a consolidated
# training export. A local path is classified below before any weights are read.
MODEL=${MODEL:-allenai/OLMoE-1B-7B-0924-Instruct}
MODEL_TAG=${MODEL_TAG:-olmoe}
# A PEFT adapter directory or zip applied on top of MODEL. ckpt/pedagogy-tutor-armF.zip
# is a LoRA over allenai/OLMo-2-1124-7B-Instruct, so it needs MODEL set to that base.
ADAPTER=${ADAPTER:-}
SPLIT=${SPLIT:?set SPLIT to the input jsonl stem under data/}
INPUT=${INPUT:-$PROJECT/data/$SPLIT.jsonl}
OUTPUT=${OUTPUT:-$PROJECT/results/$MODEL_TAG/$SPLIT.npz}
ROUTER_WEIGHTS=${ROUTER_WEIGHTS:-$PROJECT/results/$MODEL_TAG/${SPLIT}_router_weights.npz}

LAYERS=${LAYERS:-}
SITES=${SITES:-}
FIRST_K=${FIRST_K:-8}
MAX_BOUNDARIES=${MAX_BOUNDARIES:-6}
MAX_LEN=${MAX_LEN:-2048}
LIMIT=${LIMIT:-0}
DEVICE_MAP=${DEVICE_MAP:-}
# Weighted per-expert outputs are [layers, slots, top_k, hidden] per record. Useful
# for the observational expert-change metric, expensive to store; off by default.
STORE_EXPERT_OUTPUTS=${STORE_EXPERT_OUTPUTS:-0}
VERIFY_DECOMPOSITION=${VERIFY_DECOMPOSITION:-1}
REQUIRE_CLEAN_BOUNDARY=${REQUIRE_CLEAN_BOUNDARY:-0}
DRY_RUN=${DRY_RUN:-0}

export HF_HOME=${HF_HOME:-$SCRATCH/hf-cache}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=false

if [ ! -f "$INPUT" ]; then
    echo "input $INPUT does not exist; set INPUT or stage the split first" >&2
    exit 2
fi

PYTHON=${PYTHON:-$VENV/bin/python}
if [ ! -x "$PYTHON" ]; then
    echo "python $PYTHON is not executable; set VENV or PYTHON" >&2
    exit 2
fi

args=(
    --input "$INPUT"
    --output "$OUTPUT"
    --model "$MODEL"
    --max-len "$MAX_LEN"
    --first-k "$FIRST_K"
    --max-boundaries "$MAX_BOUNDARIES"
    --limit "$LIMIT"
)
[ -n "$ADAPTER" ] && args+=(--adapter "$ADAPTER")
[ -n "$LAYERS" ] && args+=(--layers "$LAYERS")
[ -n "$SITES" ] && args+=(--sites "$SITES")
[ -n "$DEVICE_MAP" ] && args+=(--device-map "$DEVICE_MAP")
[ "$STORE_EXPERT_OUTPUTS" = "1" ] && args+=(--store-expert-outputs)
[ "$VERIFY_DECOMPOSITION" = "1" ] && args+=(--verify-decomposition)
[ "$REQUIRE_CLEAN_BOUNDARY" = "1" ] && args+=(--require-clean-boundary)
[ -n "$ROUTER_WEIGHTS" ] && args+=(--router-weights "$ROUTER_WEIGHTS")

echo "started $(date -Is) on $(hostname), job ${SLURM_JOB_ID:-none}"
echo "model=$MODEL adapter=${ADAPTER:-none} split=$SPLIT dry_run=$DRY_RUN"

cd "$REPO"
mkdir -p "$REPO/logs" "$(dirname "$OUTPUT")"

# Classify before loading anything, and stop on anything that cannot be loaded: a
# DeepSpeed ZeRO directory is refused here with its reason rather than halfway
# through a 7B load. A hub id is not a local path and has nothing to classify.
if [ -d "$MODEL" ]; then
    "$PYTHON" -m projects.learning_science_semantic_atlas.checkpoint_delta --describe "$MODEL" --require-loadable
fi
if [ -n "$ADAPTER" ]; then
    "$PYTHON" -m projects.learning_science_semantic_atlas.checkpoint_delta --describe "$ADAPTER" --require-loadable
fi

if [ "$DRY_RUN" = "1" ]; then
    echo "dry run: tokenizing only, no model weights are loaded"
    exec "$PYTHON" -m projects.learning_science_semantic_atlas.trace "${args[@]}" --dry-run
fi

exec "$PYTHON" -m projects.learning_science_semantic_atlas.trace "${args[@]}"
