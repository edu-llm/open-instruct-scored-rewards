#!/bin/bash
#SBATCH -J atlas_behavior
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o logs/atlas_behavior_%A.out
#SBATCH -e logs/atlas_behavior_%A.err
#
# Score the declarative-knowledge and enacted-strategy batteries for one model.
#
# Runs three ways:
#   DRY_RUN=1 bash run_behavior.sbatch   build and write the probes, no GPU, no weights
#   bash run_behavior.sbatch             foreground on an interactive node
#   sbatch run_behavior.sbatch           batch
#
# Dense OLMo-2 and OLMoE both work unchanged. Every probe is a forced choice
# scored by teacher-forced log-probability, so nothing is sampled unless
# GENERATE=1, no router is touched, and no judge is called.
#
# RUN THE DRY RUN FIRST. It writes probes.jsonl and a report whose design audit
# says whether this corpus actually built the contrasts the enacted batteries
# claim - how many of its says-only artifacts carry their construct's vocabulary,
# and which item role supplied each pair's enacting artifact. A corpus that
# cannot support the contrast should be found there and not after a GPU hour.
#
# RESUMPTION IS THE DEFAULT AND IS WHY THIS RUNS ON A PREEMPTABLE PARTITION.
# responses.jsonl is append-only and flushed per probe, the batteries are rebuilt
# deterministically from the ontology, the corpus and SALT, and a stored response
# is reused only while its probe digest still matches - so an edited prompt or
# option is rescored instead of being mixed into the same report. A requeued job
# picks up where it stopped. RESUME=0 discards the file and starts again.
#
# THIS STAGE READS THE WITHHELD KEY AND BRIEFS, so it belongs after labelling.
# Correctness here is keyed to the corpus design, which no rater has confirmed;
# the report records that as agent-grounded and exploratory, and LABELS=... is
# how a completed labelling round narrows the batteries to verified units.
set -euo pipefail

SCRATCH=${SCRATCH:-/orcd/scratch/orcd/013/zsophia}
REPO=${REPO:-$SCRATCH/open-instruct}
VENV=${VENV:-$SCRATCH/venv_oi}
PROJECT=$REPO/projects/learning_science_semantic_atlas

# A hub id, or a local directory holding a whole model. The MoE arm is the
# default; the dense replication is allenai/OLMo-2-1124-7B-Instruct, and it needs
# MODEL_TAG set with it so the two arms do not write to one directory.
MODEL=${MODEL:-allenai/OLMoE-1B-7B-0924-Instruct}
MODEL_TAG=${MODEL_TAG:-olmoe}
# A PEFT adapter directory or zip applied on top of MODEL. ckpt/pedagogy-tutor-armF.zip
# is a LoRA over allenai/OLMo-2-1124-7B-Instruct, so it needs MODEL set to that base.
ADAPTER=${ADAPTER:-}
REVISION=${REVISION:-}
DEVICE_MAP=${DEVICE_MAP:-}

ONTOLOGY=${ONTOLOGY:-$PROJECT/ontology.yaml}
# The corpus is deliberately NOT keyed to MODEL_TAG. Scoring both arms over one
# corpus is the comparable measurement: the option texts are byte-identical, so a
# difference between arms is a difference in the model. Pointing CORPUS_TAG at
# each model's own corpus measures something else and is confounded by what the
# two models happened to generate.
CORPUS_TAG=${CORPUS_TAG:-olmoe}
CORPUS=${CORPUS:-$PROJECT/data/corpus/$CORPUS_TAG}
OUT=${OUT:-$PROJECT/results/behavior/$MODEL_TAG}

# Consensus or per-rater label files. Absent by default because no labelling round
# for this corpus has been completed, and the report says which keying it used.
LABELS=${LABELS:-}
# Space-separated subset of: declarative_knowledge boundary_conditions enacted_strategy
BATTERIES=${BATTERIES:-}

SALT=${SALT:-behavior-v1}
SEED=${SEED:-1701}
NORMALIZATION=${NORMALIZATION:-mean}
BOOTSTRAP_SAMPLES=${BOOTSTRAP_SAMPLES:-5000}
CONDITIONS_PER_PAIR=${CONDITIONS_PER_PAIR:-2}
LIMIT=${LIMIT:-0}

# The naming arm writes text and reports only whether the output uses the
# construct's vocabulary. Off by default: naming is not enactment, and the arm
# costs generation to measure the weaker of the two. Greedy unless TEMPERATURE is
# raised, and seeded per probe either way, so a requeued job regenerates the same
# text instead of quietly changing a rate it already reported.
GENERATE=${GENERATE:-0}
TEMPERATURE=${TEMPERATURE:-0.0}
TOP_P=${TOP_P:-0.9}
RESUME=${RESUME:-1}
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
if [ ! -f "$CORPUS/corpus.jsonl" ]; then
    echo "corpus $CORPUS has no corpus.jsonl; build it with generate_corpus.sbatch or set CORPUS" >&2
    exit 2
fi
if [ ! -f "$CORPUS/withheld/key.jsonl" ]; then
    echo "corpus $CORPUS has no withheld/key.jsonl; the enacted batteries need the design key" >&2
    exit 2
fi
case "$NORMALIZATION" in
    mean|total) ;;
    *) echo "NORMALIZATION must be mean or total; got $NORMALIZATION" >&2; exit 2 ;;
esac

args=(
    --ontology "$ONTOLOGY"
    --corpus "$CORPUS"
    --out "$OUT"
    --model "$MODEL"
    --salt "$SALT"
    --seed "$SEED"
    --normalization "$NORMALIZATION"
    --bootstrap-samples "$BOOTSTRAP_SAMPLES"
    --conditions-per-pair "$CONDITIONS_PER_PAIR"
    --limit "$LIMIT"
    --temperature "$TEMPERATURE"
    --top-p "$TOP_P"
)
[ -n "$ADAPTER" ] && args+=(--adapter "$ADAPTER")
[ -n "$REVISION" ] && args+=(--revision "$REVISION")
[ -n "$DEVICE_MAP" ] && args+=(--device-map "$DEVICE_MAP")
[ "$GENERATE" = "1" ] && args+=(--generate)
[ "$RESUME" = "0" ] && args+=(--no-resume)
# Both take several values. Split on whitespace into an array rather than relying
# on an unquoted expansion, so a path with a space stays one argument.
if [ -n "$BATTERIES" ]; then
    read -ra battery_list <<< "$BATTERIES"
    args+=(--batteries "${battery_list[@]}")
fi
if [ -n "$LABELS" ]; then
    read -ra label_list <<< "$LABELS"
    args+=(--labels "${label_list[@]}")
fi

RESPONSES=$OUT/responses.jsonl
echo "started $(date -Is) on $(hostname), job ${SLURM_JOB_ID:-none}"
echo "model=$MODEL adapter=${ADAPTER:-none} tag=$MODEL_TAG dry_run=$DRY_RUN resume=$RESUME generate=$GENERATE"
echo "corpus=$CORPUS salt=$SALT seed=$SEED normalization=$NORMALIZATION"
echo "ontology=$ONTOLOGY out=$OUT labels=${LABELS:-none} batteries=${BATTERIES:-all}"
if [ -f "$RESPONSES" ]; then
    echo "resuming: $(wc -l < "$RESPONSES") response(s) already on disk at $RESPONSES"
fi

cd "$REPO"
mkdir -p "$REPO/logs" "$OUT"

# Classify before loading anything. A DeepSpeed ZeRO directory is refused here
# with its reason rather than halfway through a 7B load, and a hub id is not a
# local path and has nothing to classify.
if [ -d "$MODEL" ]; then
    "$PYTHON" -m projects.learning_science_semantic_atlas.checkpoint_delta --describe "$MODEL" --require-loadable
fi
if [ -n "$ADAPTER" ]; then
    "$PYTHON" -m projects.learning_science_semantic_atlas.checkpoint_delta --describe "$ADAPTER" --require-loadable
fi

if [ "$DRY_RUN" = "1" ]; then
    echo "dry run: building the probes only, no model weights are loaded"
    exec "$PYTHON" -m projects.learning_science_semantic_atlas.behavior "${args[@]}" --dry-run
fi

exec "$PYTHON" -m projects.learning_science_semantic_atlas.behavior "${args[@]}"
