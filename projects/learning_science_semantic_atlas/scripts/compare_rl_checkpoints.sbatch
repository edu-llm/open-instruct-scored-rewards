#!/bin/bash
#SBATCH -J atlas_ckpt_delta
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --time=06:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o logs/atlas_ckpt_delta_%A.out
#SBATCH -e logs/atlas_ckpt_delta_%A.err
#
# What the RL run changed, measured on one fixed set of texts.
#
# The base and every tuned checkpoint replay the same split, then the traces are
# compared. Nothing is inferred from the training configuration: the pedagogy_olmoe
# run froze mlp.gate, and this script still measures routing, because a frozen
# router only fixes the parameters, not the input the router reads.
#
#   STAGE=describe   classify every checkpoint and stop      (no GPU)
#   STAGE=trace      describe, then trace base and tuned
#   STAGE=compare    describe, trace, then write the deltas  (default)
#   DRY_RUN=1        print the plan and the tokenizer report, load no weights
#
# The checkpoints under CKPT_ROOT are DeepSpeed ZeRO training states. They cannot
# be traced as they stand. Consolidate each one into EXPORT_ROOT/global_step<N>
# first, as either a PEFT adapter or a whole merged model; an adapter is applied on
# top of BASE_MODEL and a merged model is traced in its own right. This script
# reports exactly which steps are missing an export rather than skipping them
# silently, and refuses to trace an export that cannot be loaded.
set -euo pipefail

SCRATCH=${SCRATCH:-/orcd/scratch/orcd/013/zsophia}
REPO=${REPO:-$SCRATCH/open-instruct}
VENV=${VENV:-$SCRATCH/venv_oi}
PROJECT=$REPO/projects/learning_science_semantic_atlas

BASE_MODEL=${BASE_MODEL:-allenai/OLMoE-1B-7B-0924-Instruct}
MODEL_TAG=${MODEL_TAG:-olmoe_rl}
CKPT_ROOT=${CKPT_ROOT:-$SCRATCH/ckpt/pedagogy_olmoe}
EXPORT_ROOT=${EXPORT_ROOT:-$SCRATCH/ckpt/pedagogy_olmoe_exported}
STEPS=${STEPS:-180 190 200}

SPLIT=${SPLIT:?set SPLIT to the input jsonl stem under data/}
INPUT=${INPUT:-$PROJECT/data/$SPLIT.jsonl}
RESULTS=${RESULTS:-$PROJECT/results/$MODEL_TAG}
ROWS=${ROWS:-$INPUT}

SLOTS=${SLOTS:-}
SELECTIVITY_SITE=${SELECTIVITY_SITE:-mlp_in}
# Siblings share a prompt, so prompt_last cannot separate them; it is still traced
# and still enters the routing and activation deltas, which compare checkpoints
# rather than siblings.
SELECTIVITY_ROLE=${SELECTIVITY_ROLE:-content_last}
FOLDS=${FOLDS:-5}
SEED=${SEED:-1701}
STAGE=${STAGE:-compare}
DRY_RUN=${DRY_RUN:-0}

export HF_HOME=${HF_HOME:-$SCRATCH/hf-cache}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=false

PYTHON=${PYTHON:-$VENV/bin/python}
if [ ! -x "$PYTHON" ]; then
    echo "python $PYTHON is not executable; set VENV or PYTHON" >&2
    exit 2
fi
if [ ! -f "$INPUT" ]; then
    echo "input $INPUT does not exist; set INPUT or stage the split first" >&2
    exit 2
fi

echo "started $(date -Is) on $(hostname), job ${SLURM_JOB_ID:-none}"
echo "base=$BASE_MODEL steps='$STEPS' stage=$STAGE dry_run=$DRY_RUN"

cd "$REPO"
mkdir -p "$REPO/logs" "$RESULTS"

# ---------------------------------------------------------------- describe
usable_steps=()
missing_steps=()
describe=("$PYTHON" -m projects.learning_science_semantic_atlas.checkpoint_delta --describe)
for step in $STEPS; do
    raw="$CKPT_ROOT/global_step$step"
    exported="$EXPORT_ROOT/global_step$step"
    echo "--- global_step$step ---"
    # An export directory that exists is not the same as one that can be loaded, so
    # --require-loadable decides, and a step it rejects is reported rather than traced.
    if [ -d "$exported" ] && "${describe[@]}" "$exported" --require-loadable; then
        usable_steps+=("$step")
    else
        [ -e "$raw" ] && "${describe[@]}" "$raw"
        echo "no loadable export at $exported" >&2
        missing_steps+=("$step")
    fi
done

if [ ${#missing_steps[@]} -gt 0 ]; then
    echo "steps without a loadable export: ${missing_steps[*]}" >&2
fi
if [ "$STAGE" = "describe" ]; then
    exit 0
fi
if [ ${#usable_steps[@]} -eq 0 ]; then
    # A dry run is still worth finishing: it checks the split, the slot plan and
    # every path, which is exactly what you want before spending a GPU hour.
    if [ "$DRY_RUN" != "1" ]; then
        echo "no tuned checkpoint can be loaded; consolidate the ZeRO states into $EXPORT_ROOT first" >&2
        exit 3
    fi
    echo "no tuned checkpoint is loadable; dry running the base arm only" >&2
fi

# ------------------------------------------------------------------- trace
trace_one() {
    local tag=$1 model=$2 adapter=$3
    # REPO and VENV are forwarded so the inner script runs this checkout and this
    # interpreter rather than falling back to its own defaults.
    REPO="$REPO" \
    VENV="$VENV" \
    PYTHON="$PYTHON" \
    MODEL="$model" \
    MODEL_TAG="$MODEL_TAG" \
    ADAPTER="$adapter" \
    SPLIT="$SPLIT" \
    INPUT="$INPUT" \
    OUTPUT="$RESULTS/$tag.npz" \
    ROUTER_WEIGHTS="$RESULTS/${tag}_router_weights.npz" \
    STORE_EXPERT_OUTPUTS=${STORE_EXPERT_OUTPUTS:-1} \
    DRY_RUN="$DRY_RUN" \
    bash "$PROJECT/scripts/trace_atlas.sbatch"
}

trace_one base "$BASE_MODEL" ""
for step in ${usable_steps[@]+"${usable_steps[@]}"}; do
    exported="$EXPORT_ROOT/global_step$step"
    if [ -f "$exported/adapter_config.json" ]; then
        trace_one "step$step" "$BASE_MODEL" "$exported"
    else
        trace_one "step$step" "$exported" ""
    fi
done

if [ "$DRY_RUN" = "1" ]; then
    echo "dry run complete: no traces were written, so no deltas are computed"
    exit 0
fi
if [ "$STAGE" = "trace" ]; then
    exit 0
fi

# ----------------------------------------------------------------- compare
for step in ${usable_steps[@]+"${usable_steps[@]}"}; do
    args=(
        --base-trace "$RESULTS/base.npz"
        --tuned-trace "$RESULTS/step$step.npz"
        --out "$RESULTS/delta_step$step.json"
        --selectivity-site "$SELECTIVITY_SITE"
        --selectivity-role "$SELECTIVITY_ROLE"
        --rows "$ROWS"
        --folds "$FOLDS"
        --seed "$SEED"
    )
    # A dense base writes no router sidecar. Without both, the delta report says
    # the routing decomposition is unavailable instead of failing on a missing file.
    base_router="$RESULTS/base_router_weights.npz"
    tuned_router="$RESULTS/step${step}_router_weights.npz"
    if [ -f "$base_router" ] && [ -f "$tuned_router" ]; then
        args+=(--base-router-weights "$base_router" --tuned-router-weights "$tuned_router")
    else
        echo "no router sidecar pair for step $step; the routing decomposition will be skipped" >&2
    fi
    [ -n "$SLOTS" ] && args+=(--slots "$SLOTS")
    "$PYTHON" -m projects.learning_science_semantic_atlas.checkpoint_delta "${args[@]}"
done

echo "finished $(date -Is); reports are in $RESULTS"
