#!/usr/bin/env bash
#SBATCH --job-name=olmoe_ft_sweep
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:4
#SBATCH --time=12:00:00
#SBATCH --mem=400G
#SBATCH --cpus-per-task=32
#SBATCH --array=0-2%1
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/orcd/scratch/orcd/013/zsophia/open-instruct/projects/olmoe_full_finetune/runs/logs/%x_%A_%a.out
#SBATCH --error=/orcd/scratch/orcd/013/zsophia/open-instruct/projects/olmoe_full_finetune/runs/logs/%x_%A_%a.err

set -euo pipefail

learning_rates=(4e-5 6.1499e-5 8e-5)
index="${SLURM_ARRAY_TASK_ID:?sweep.sbatch must run as a Slurm array}"
if (( index < 0 || index >= ${#learning_rates[@]} )); then
    echo "array index $index is outside the learning-rate grid" >&2
    exit 2
fi

SCRATCH="${SCRATCH:-/orcd/scratch/orcd/013/zsophia}"
REPO="${REPO:-$SCRATCH/open-instruct}"
SOURCE_PROJECT_DIR="$REPO/projects/olmoe_full_finetune"
export SCRATCH REPO SOURCE_PROJECT_DIR

export LR="${learning_rates[$index]}"
export SEED="${SEED:-42}"
export RUN_NAME="dolmino_lr_${LR}_seed_${SEED}"

cd "$REPO"
exec bash "$SOURCE_PROJECT_DIR/job.sh"
