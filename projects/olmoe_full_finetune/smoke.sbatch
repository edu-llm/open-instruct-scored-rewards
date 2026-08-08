#!/usr/bin/env bash
#SBATCH --job-name=olmoe_ft_smoke
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:4
#SBATCH --time=02:00:00
#SBATCH --mem=400G
#SBATCH --cpus-per-task=32
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/orcd/scratch/orcd/013/zsophia/open-instruct/projects/olmoe_full_finetune/runs/logs/%x_%j.out
#SBATCH --error=/orcd/scratch/orcd/013/zsophia/open-instruct/projects/olmoe_full_finetune/runs/logs/%x_%j.err

set -euo pipefail

SCRATCH="${SCRATCH:-/orcd/scratch/orcd/013/zsophia}"
REPO="${REPO:-$SCRATCH/open-instruct}"
SOURCE_PROJECT_DIR="$REPO/projects/olmoe_full_finetune"
export SCRATCH REPO SOURCE_PROJECT_DIR

export LR="${LR:-2e-5}"
export SEED="${SEED:-8}"
export RUN_NAME="${RUN_NAME:-smoke}"
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-3}"
export MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-512}"
export CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-1}"

cd "$REPO"
exec bash "$SOURCE_PROJECT_DIR/job.sh"
