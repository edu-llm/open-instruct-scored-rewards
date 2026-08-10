#!/bin/bash
#SBATCH -J pedagogy_probe
#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --open-mode=append
#SBATCH -o logs/pedagogy_mech_probe_%A.out
#SBATCH -e logs/pedagogy_mech_probe_%A.err
set -euo pipefail

SCRATCH=${SCRATCH:-/orcd/scratch/orcd/013/zsophia}
REPO=${REPO:-$SCRATCH/open-instruct}
VENV=${VENV:-$SCRATCH/venv_oi}
PROJECT=$REPO/projects/pedagogy_mech_interp
MODEL_TAG=${MODEL_TAG:-olmoe}
PERMUTATIONS=${PERMUTATIONS:-50}
BOOTSTRAP_SAMPLES=${BOOTSTRAP_SAMPLES:-5000}

cd "$REPO"
"$VENV/bin/python" -m projects.pedagogy_mech_interp.probe \
    --train-trace "$PROJECT/results/$MODEL_TAG/discovery_train.npz" \
    --validation-trace "$PROJECT/results/$MODEL_TAG/discovery_validation.npz" \
    --test-trace "$PROJECT/results/$MODEL_TAG/natural_test.npz" \
    --train-rows "$PROJECT/data/pairs/discovery_train.jsonl" \
    --validation-rows "$PROJECT/data/pairs/discovery_validation.jsonl" \
    --test-rows "$PROJECT/data/pairs/natural_test.jsonl" \
    --out "$PROJECT/results/$MODEL_TAG/discovery" \
    --permutations "$PERMUTATIONS" \
    --bootstrap-samples "$BOOTSTRAP_SAMPLES"
