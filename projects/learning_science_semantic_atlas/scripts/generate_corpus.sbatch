#!/bin/bash
#SBATCH -J atlas_corpus
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --time=03:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o logs/atlas_corpus_%A.out
#SBATCH -e logs/atlas_corpus_%A.err
#
# Build the 24-item calibration set and the 160-item round-1 pool, and write the
# artifact texts with one model on one GPU.
#
#   MODE=plan       the design, the briefs and every count check   (no GPU)
#   MODE=generate   the artifact texts only, resuming
#   MODE=assemble   corpus, key and audit from what is on disk     (no GPU)
#   MODE=all        all three, in order                            (default)
#   NO_MODEL=1      MODE=all without weights: a marked scaffold corpus (no GPU)
#   DRY_RUN=1       print the effective settings and run MODE=plan (no GPU)
#
# RESUMPTION IS THE DEFAULT AND IS WHY THIS RUNS ON A PREEMPTABLE PARTITION.
# withheld/generations.jsonl is append-only and flushed per item, the plan is
# rebuilt deterministically from the ontology and SALT, and generate skips any
# item whose recorded prompt digest still matches its brief. A requeued job
# therefore picks up where it stopped, and because each item's sampling seed is
# derived from SEED and its unit id, an item regenerated after a preemption comes
# back as the same text. RESUME=0 discards the file and starts again.
#
# The 184 briefs are withheld: each one states its item's target construct, role
# and anchor. They live under OUT/withheld together with the key, the plan and the
# raw generations. Only corpus.jsonl, calibration_corpus.jsonl, items.jsonl and
# manifest.json are blinded, and the audit re-checks that before writing them.
set -euo pipefail

SCRATCH=${SCRATCH:-/orcd/scratch/orcd/013/zsophia}
REPO=${REPO:-$SCRATCH/open-instruct}
VENV=${VENV:-$SCRATCH/venv_oi}
PROJECT=$REPO/projects/learning_science_semantic_atlas

# A hub id or a local directory holding a whole model. The default is the MoE arm;
# OLMo-2 is the dense replication and needs nothing else changed here.
MODEL=${MODEL:-allenai/OLMoE-1B-7B-0924-Instruct}
MODEL_TAG=${MODEL_TAG:-olmoe}
ONTOLOGY=${ONTOLOGY:-$PROJECT/ontology.yaml}
OUT=${OUT:-$PROJECT/data/corpus/$MODEL_TAG}

# The salt names the pool. A new salt is a new, non-colliding set of item ids, so
# change it for a second round rather than overwriting the first.
SALT=${SALT:-atlas-v1}
SEED=${SEED:-1701}
TEMPERATURE=${TEMPERATURE:-0.7}
TOP_P=${TOP_P:-0.9}
# Resamples before a generation is recorded as refused. A refusal is a model that
# echoed its brief or used vocabulary the item forbids; it is counted in the
# manifest either way.
MAX_ATTEMPTS=${MAX_ATTEMPTS:-3}
LIMIT=${LIMIT:-0}

MODE=${MODE:-all}
NO_MODEL=${NO_MODEL:-0}
RESUME=${RESUME:-1}
# Fill absent or refused generations with a marked scaffold instead of failing.
# Off for the GPU path: a corpus quietly completed with placeholders is worse than
# a job that stops and says which items are missing.
SCAFFOLD_MISSING=${SCAFFOLD_MISSING:-0}
DRY_RUN=${DRY_RUN:-0}

export HF_HOME=${HF_HOME:-$SCRATCH/hf-cache}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=false

PYTHON=${PYTHON:-$VENV/bin/python}
if [ ! -x "$PYTHON" ]; then
    echo "python $PYTHON is not executable; set VENV or PYTHON" >&2
    exit 2
fi
if [ ! -f "$ONTOLOGY" ]; then
    echo "ontology $ONTOLOGY does not exist; set ONTOLOGY" >&2
    exit 2
fi

if [ "$NO_MODEL" = "1" ]; then
    MODE=all
fi
case "$MODE" in
    plan|generate|assemble|all) ;;
    *) echo "MODE must be one of plan, generate, assemble, all; got $MODE" >&2; exit 2 ;;
esac

args=(
    --ontology "$ONTOLOGY"
    --out "$OUT"
    --salt "$SALT"
    --seed "$SEED"
    --temperature "$TEMPERATURE"
    --top-p "$TOP_P"
    --max-attempts "$MAX_ATTEMPTS"
    --limit "$LIMIT"
    --model "$MODEL"
)
[ "$NO_MODEL" = "1" ] && args+=(--no-model)
[ "$SCAFFOLD_MISSING" = "1" ] && args+=(--scaffold-missing)
[ "$RESUME" = "0" ] && args+=(--no-resume)

GENERATIONS=$OUT/withheld/generations.jsonl
echo "started $(date -Is) on $(hostname), job ${SLURM_JOB_ID:-none}"
echo "mode=$MODE no_model=$NO_MODEL dry_run=$DRY_RUN resume=$RESUME"
echo "model=$MODEL salt=$SALT seed=$SEED temperature=$TEMPERATURE"
echo "ontology=$ONTOLOGY out=$OUT"
if [ -f "$GENERATIONS" ]; then
    echo "resuming: $(wc -l < "$GENERATIONS") generation(s) already on disk at $GENERATIONS"
fi

cd "$REPO"
mkdir -p "$REPO/logs" "$OUT/withheld"

if [ "$DRY_RUN" = "1" ]; then
    echo "dry run: planning only, no model weights are loaded"
    exec "$PYTHON" -m projects.learning_science_semantic_atlas.build_corpus "${args[@]}" --mode plan
fi

exec "$PYTHON" -m projects.learning_science_semantic_atlas.build_corpus "${args[@]}" --mode "$MODE"
