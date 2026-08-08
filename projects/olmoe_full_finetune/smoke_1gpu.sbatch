#!/usr/bin/env bash
#SBATCH --job-name=olmoe_ft_1gpu
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
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

# One GPU cannot hold full Adam state for all 6.9B parameters. Keep every
# parameter trainable but offload parameters and optimizer state to host RAM.
# A single global sample and optimizer step make this a mechanics/memory smoke,
# not a batch-equivalent quality comparison with the four-GPU recipe.
export GPUS=1
export GLOBAL_BATCH_SIZE=1
export LR="${LR:-2e-5}"
export SEED="${SEED:-8}"
export RUN_NAME="${RUN_NAME:-smoke_1gpu_offload}"
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-1}"
export MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-8}"
export CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-1}"
export DEEPSPEED_CONFIG_FILE="${DEEPSPEED_CONFIG_FILE:-projects/olmoe_full_finetune/stage3_offloading_accelerate.conf}"

cd "$REPO"
exec bash "$SOURCE_PROJECT_DIR/job.sh"
