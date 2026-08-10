#!/bin/bash
set -euo pipefail

SCRATCH=${SCRATCH:-/orcd/scratch/orcd/013/zsophia}
REPO=${REPO:-$SCRATCH/open-instruct}
PROJECT=$REPO/projects/pedagogy_mech_interp

MODEL=${MODEL:-allenai/OLMoE-1B-7B-0924-Instruct}
MODEL_TAG=${MODEL_TAG:-olmoe}
LIMIT=${LIMIT:-0}
LIMIT_PAIRS=${LIMIT_PAIRS:-0}

mkdir -p "$REPO/logs" "$PROJECT/results/$MODEL_TAG"
cd "$REPO"

trace_jobs=()
for split in discovery_train discovery_validation natural_test; do
    job=$(sbatch --parsable \
        --export=ALL,MODEL="$MODEL",MODEL_TAG="$MODEL_TAG",SPLIT="$split",LIMIT="$LIMIT" \
        "$PROJECT/scripts/trace.sbatch")
    trace_jobs+=("$job")
done
dependency=$(IFS=:; echo "${trace_jobs[*]}")

probe_job=$(sbatch --parsable --dependency="afterok:$dependency" \
    --export=ALL,MODEL_TAG="$MODEL_TAG" "$PROJECT/scripts/probe.sbatch")
intervention_job=$(sbatch --parsable --dependency="afterok:$probe_job" \
    --export=ALL,MODEL="$MODEL",MODEL_TAG="$MODEL_TAG",LIMIT_PAIRS="$LIMIT_PAIRS" \
    "$PROJECT/scripts/intervene.sbatch")
analysis_job=$(sbatch --parsable --dependency="afterok:$intervention_job" \
    --export=ALL,MODEL_TAG="$MODEL_TAG" "$PROJECT/scripts/analyze.sbatch")

printf 'trace=%s\nprobe=%s\nintervene=%s\nanalyze=%s\n' \
    "$dependency" "$probe_job" "$intervention_job" "$analysis_job"
