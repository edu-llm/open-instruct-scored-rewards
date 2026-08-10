#!/bin/bash
#SBATCH -J pedagogy_intervene
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --time=12:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o logs/pedagogy_mech_intervene_%A.out
#SBATCH -e logs/pedagogy_mech_intervene_%A.err
set -euo pipefail

SCRATCH=${SCRATCH:-/orcd/scratch/orcd/013/zsophia}
REPO=${REPO:-$SCRATCH/open-instruct}
VENV=${VENV:-$SCRATCH/venv_oi}
PROJECT=$REPO/projects/pedagogy_mech_interp

MODEL=${MODEL:-allenai/OLMoE-1B-7B-0924-Instruct}
MODEL_TAG=${MODEL_TAG:-olmoe}
LIMIT_PAIRS=${LIMIT_PAIRS:-0}
FORCE_FAILED_GATES=${FORCE_FAILED_GATES:-0}

export HF_HOME=${HF_HOME:-$SCRATCH/hf-cache}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=false

extra=()
if [ "$FORCE_FAILED_GATES" = 1 ]; then
    extra+=(--force-failed-gates)
fi

cd "$REPO"
"$VENV/bin/python" -m projects.pedagogy_mech_interp.intervene \
    --rows "$PROJECT/data/pairs/confirmatory.jsonl" \
    --key "$PROJECT/data/pairs/confirmatory_key.jsonl" \
    --freeze "$PROJECT/results/$MODEL_TAG/discovery/candidate_freeze.json" \
    --directions "$PROJECT/results/$MODEL_TAG/discovery/directions.npz" \
    --out "$PROJECT/results/$MODEL_TAG/interventions.jsonl" \
    --model "$MODEL" \
    --limit-pairs "$LIMIT_PAIRS" \
    "${extra[@]}"
