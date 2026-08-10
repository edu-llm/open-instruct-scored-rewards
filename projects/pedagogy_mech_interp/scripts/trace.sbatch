#!/bin/bash
#SBATCH -J pedagogy_trace
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --time=04:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o logs/pedagogy_mech_trace_%A.out
#SBATCH -e logs/pedagogy_mech_trace_%A.err
set -euo pipefail

SCRATCH=${SCRATCH:-/orcd/scratch/orcd/013/zsophia}
REPO=${REPO:-$SCRATCH/open-instruct}
VENV=${VENV:-$SCRATCH/venv_oi}
PROJECT=$REPO/projects/pedagogy_mech_interp

MODEL=${MODEL:-allenai/OLMoE-1B-7B-0924-Instruct}
MODEL_TAG=${MODEL_TAG:-olmoe}
SPLIT=${SPLIT:?set SPLIT to discovery_train, discovery_validation, or natural_test}
LIMIT=${LIMIT:-0}

export HF_HOME=${HF_HOME:-$SCRATCH/hf-cache}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=false

mkdir -p "$PROJECT/results/$MODEL_TAG"
cd "$REPO"
"$VENV/bin/python" -m projects.pedagogy_mech_interp.trace \
    --input "$PROJECT/data/pairs/$SPLIT.jsonl" \
    --output "$PROJECT/results/$MODEL_TAG/$SPLIT.npz" \
    --model "$MODEL" \
    --limit "$LIMIT" \
    --verify-decomposition
