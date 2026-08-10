#!/bin/bash
#SBATCH -J pedagogy_analyze
#SBATCH --partition=mit_quicktest
#SBATCH --nodes=1
#SBATCH --time=00:15:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --open-mode=append
#SBATCH -o logs/pedagogy_mech_analyze_%A.out
#SBATCH -e logs/pedagogy_mech_analyze_%A.err
set -euo pipefail

SCRATCH=${SCRATCH:-/orcd/scratch/orcd/013/zsophia}
REPO=${REPO:-$SCRATCH/open-instruct}
VENV=${VENV:-$SCRATCH/venv_oi}
PROJECT=$REPO/projects/pedagogy_mech_interp
MODEL_TAG=${MODEL_TAG:-olmoe}

cd "$REPO"
"$VENV/bin/python" -m projects.pedagogy_mech_interp.analyze \
    --interventions "$PROJECT/results/$MODEL_TAG/interventions.jsonl" \
    --discovery-report "$PROJECT/results/$MODEL_TAG/discovery/discovery_report.json" \
    --pair-rows "$PROJECT/data/pairs/confirmatory.jsonl" \
    --out "$PROJECT/results/$MODEL_TAG/final_report.json"
