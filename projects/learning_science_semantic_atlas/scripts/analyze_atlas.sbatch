#!/bin/bash
#SBATCH -J atlas_analysis
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --open-mode=append
#SBATCH -o logs/atlas_analysis_%A.out
#SBATCH -e logs/atlas_analysis_%A.err
#
# Prepare exact replay rows and run both representation report families over an
# existing NPZ trace. MODEL_TAG is required: model identity must never be inferred
# from a checkpoint basename.
#
#   MODEL_TAG=olmoe-base DRY_RUN=1 bash analyze_atlas.sbatch
#   MODEL_TAG=olmoe-base sbatch analyze_atlas.sbatch
set -euo pipefail

SCRATCH=${SCRATCH:-/orcd/scratch/orcd/013/zsophia}
REPO=${REPO:-$SCRATCH/open-instruct}
VENV=${VENV:-$SCRATCH/venv_oi}
PROJECT=$REPO/projects/learning_science_semantic_atlas

MODEL_TAG=${MODEL_TAG:?set MODEL_TAG explicitly, for example olmoe-base or olmo2-armF}
SPLIT=${SPLIT:-round1}
CORPUS_DIR=${CORPUS_DIR:-$PROJECT/data/corpus/$MODEL_TAG}
PROMPTS=${PROMPTS:-$CORPUS_DIR/withheld/prompts.jsonl}
GENERATIONS=${GENERATIONS:-$CORPUS_DIR/withheld/generations.jsonl}
KEY=${KEY:-$CORPUS_DIR/withheld/key.jsonl}
ROWS=${ROWS:-$PROJECT/data/trace_rows/$MODEL_TAG/$SPLIT.jsonl}
TRACE=${TRACE:-$PROJECT/results/$MODEL_TAG/$SPLIT.npz}
REPRESENTATION_OUT=${REPRESENTATION_OUT:-$PROJECT/results/$MODEL_TAG/${SPLIT}_representations.json}
GEOMETRY_OUT=${GEOMETRY_OUT:-$PROJECT/results/$MODEL_TAG/${SPLIT}_geometry.json}
ONTOLOGY=${ONTOLOGY:-$PROJECT/ontology.yaml}

SITE=${SITE:-mlp_in}
ROLE=${ROLE:-content_last}
LAYER=${LAYER:--1}
FOLDS=${FOLDS:-5}
BOOTSTRAP_SAMPLES=${BOOTSTRAP_SAMPLES:-2000}
PERMUTATIONS=${PERMUTATIONS:-1000}
SEED=${SEED:-1701}
SWEEP=${SWEEP:-1}
PREPARE_ROWS=${PREPARE_ROWS:-1}
DRY_RUN=${DRY_RUN:-0}
PYTHON=${PYTHON:-$VENV/bin/python}

prepare_args=(
    -m projects.learning_science_semantic_atlas.prepare_trace_rows
    --prompts "$PROMPTS"
    --generations "$GENERATIONS"
    --key "$KEY"
    --out "$ROWS"
    --model-tag "$MODEL_TAG"
)
representation_args=(
    -m projects.learning_science_semantic_atlas.analyze_representations
    --trace "$TRACE"
    --rows "$ROWS"
    --out "$REPRESENTATION_OUT"
    --site "$SITE"
    --role "$ROLE"
    --layer "$LAYER"
    --folds "$FOLDS"
    --bootstrap-samples "$BOOTSTRAP_SAMPLES"
    --seed "$SEED"
)
[ "$SWEEP" = "1" ] && representation_args+=(--sweep)
geometry_args=(
    -m projects.learning_science_semantic_atlas.semantic_geometry
    --trace "$TRACE"
    --model-tag "$MODEL_TAG"
    --rows "$ROWS"
    --ontology "$ONTOLOGY"
    --out "$GEOMETRY_OUT"
    --site "$SITE"
    --role "$ROLE"
    --layer "$LAYER"
    --permutations "$PERMUTATIONS"
    --seed "$SEED"
)

print_command() {
    printf '  %q' "$@"
    printf '\n'
}

echo "model_tag=$MODEL_TAG split=$SPLIT dry_run=$DRY_RUN prepare_rows=$PREPARE_ROWS"
echo "trace=$TRACE rows=$ROWS site=$SITE role=$ROLE layer=$LAYER"
if [ "$DRY_RUN" = "1" ]; then
    echo "dry run: commands only; no rows or reports will be written"
    [ "$PREPARE_ROWS" = "1" ] && print_command "$PYTHON" "${prepare_args[@]}"
    print_command "$PYTHON" "${representation_args[@]}"
    print_command "$PYTHON" "${geometry_args[@]}"
    exit 0
fi

if [ ! -x "$PYTHON" ]; then
    echo "python $PYTHON is not executable; set VENV or PYTHON" >&2
    exit 2
fi
if [ ! -f "$TRACE" ]; then
    echo "trace $TRACE does not exist; set TRACE to an existing NPZ" >&2
    exit 2
fi
if [ ! -f "$ONTOLOGY" ]; then
    echo "ontology $ONTOLOGY does not exist" >&2
    exit 2
fi
if [ "$PREPARE_ROWS" = "1" ]; then
    for input in "$PROMPTS" "$GENERATIONS" "$KEY"; do
        if [ ! -f "$input" ]; then
            echo "join input $input does not exist" >&2
            exit 2
        fi
    done
elif [ ! -f "$ROWS" ]; then
    echo "rows $ROWS do not exist and PREPARE_ROWS=0" >&2
    exit 2
fi

cd "$REPO"
mkdir -p "$REPO/logs" "$(dirname "$ROWS")" "$(dirname "$REPRESENTATION_OUT")" "$(dirname "$GEOMETRY_OUT")"

if [ "$PREPARE_ROWS" = "1" ]; then
    "$PYTHON" "${prepare_args[@]}"
fi
"$PYTHON" "${representation_args[@]}"
"$PYTHON" "${geometry_args[@]}"
