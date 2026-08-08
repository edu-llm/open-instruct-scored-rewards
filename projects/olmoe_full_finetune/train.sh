#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${SOURCE_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
REPO="${REPO:-$(cd "$PROJECT_DIR/../.." && pwd)}"
cd "$REPO"

MODEL="${MODEL:-allenai/OLMoE-1B-7B-0924}"
MODEL_REVISION="${MODEL_REVISION:-6d84c48581ece794365f2b8e9cfb043c68ade9c5}"
DATASET="${DATASET:-allenai/dolmino-mix-1124}"
DATASET_REVISION="${DATASET_REVISION:-a319f19eef1e257417b11ea8c30da266ae175557}"
read -r -a DATASET_CONFIG_ARRAY <<<"${DATASET_CONFIGS:-dclm flan pes2o wiki stackexchange math}"
read -r -a MIX_PROBABILITY_ARRAY <<<"${MIX_PROBABILITIES:-0.472 0.166 0.0585 0.0711 0.0245 0.208}"

LR="${LR:-6.1499e-5}"
SEED="${SEED:-42}"
GPUS="${GPUS:-4}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-1024}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-4096}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-11931}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
WARMUP_STEPS="${WARMUP_STEPS:-0}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-20}"
DEEPSPEED_CONFIG_FILE="${DEEPSPEED_CONFIG_FILE:-projects/olmoe_full_finetune/stage3_midtrain_accelerate.conf}"

for value_name in GPUS MICRO_BATCH_SIZE GLOBAL_BATCH_SIZE MAX_SEQ_LENGTH MAX_TRAIN_STEPS CHECKPOINTING_STEPS; do
    value="${!value_name}"
    if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "$value_name must be a positive integer, got '$value'" >&2
        exit 2
    fi
done

data_parallel_batch=$((GPUS * MICRO_BATCH_SIZE))
if (( GLOBAL_BATCH_SIZE % data_parallel_batch != 0 )); then
    echo "GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE must be divisible by GPUS*MICRO_BATCH_SIZE=$data_parallel_batch" >&2
    exit 2
fi
GRADIENT_ACCUMULATION_STEPS=$((GLOBAL_BATCH_SIZE / data_parallel_batch))

lr_slug="${LR//./p}"
RUN_NAME="${RUN_NAME:-lr_${lr_slug}_seed_${SEED}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/runs}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/$RUN_NAME}"
mkdir -p "$OUTPUT_DIR"

train_args=(
    "${MIDTRAIN_SCRIPT:-projects/olmoe_full_finetune/midtrain.py}"
    --run_name "$RUN_NAME"
    --model_name_or_path "$MODEL"
    --model_revision "$MODEL_REVISION"
    --dataset_name "$DATASET"
    --dataset_revision "$DATASET_REVISION"
    --dataset_configs "${DATASET_CONFIG_ARRAY[@]}"
    --mix_probabilities "${MIX_PROBABILITY_ARRAY[@]}"
    --shuffle_buffer_size "${SHUFFLE_BUFFER_SIZE:-10000}"
    --sequence_length "$MAX_SEQ_LENGTH"
    --per_device_train_batch_size "$MICRO_BATCH_SIZE"
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
    --max_train_steps "$MAX_TRAIN_STEPS"
    --learning_rate "$LR"
    --lr_scheduler_type "$LR_SCHEDULER_TYPE"
    --warmup_steps "$WARMUP_STEPS"
    --weight_decay "${WEIGHT_DECAY:-0.1}"
    --router_aux_loss_coef "${ROUTER_AUX_LOSS_COEF:-0.01}"
    --router_z_loss_coef "${ROUTER_Z_LOSS_COEF:-0.001}"
    --output_dir "$OUTPUT_DIR"
    --checkpointing_steps "$CHECKPOINTING_STEPS"
    --keep_last_n_checkpoints 1
    --empty_cache_steps "${EMPTY_CACHE_STEPS:-1}"
    --seed "$SEED"
)

if [[ -n "${REMOTE_CHECKPOINT_DIR:-}" ]]; then
    train_args+=(--remote_checkpoint_dir "$REMOTE_CHECKPOINT_DIR")
fi
if [[ -n "${REMOTE_OUTPUT_DIR:-}" ]]; then
    train_args+=(--remote_output_dir "$REMOTE_OUTPUT_DIR")
fi

if [[ "${WITH_TRACKING:-1}" == "1" ]]; then
    train_args+=(
        --with_tracking
        --wandb_project "${WANDB_PROJECT:-olmoe-midtraining}"
        --wandb_entity "${WANDB_ENTITY:-eduLLM}"
    )
fi

launch=(
    accelerate launch
    --mixed_precision bf16
    --num_processes "$GPUS"
    --use_deepspeed
    --deepspeed_config_file "$DEEPSPEED_CONFIG_FILE"
    "${train_args[@]}"
)

echo "run=$RUN_NAME model=$MODEL@$MODEL_REVISION"
echo "dataset=$DATASET@$DATASET_REVISION"
echo "dataset_configs=${DATASET_CONFIG_ARRAY[*]} mix_probabilities=${MIX_PROBABILITY_ARRAY[*]}"
echo "full_parameters=true scheduler=$LR_SCHEDULER_TYPE peak_lr=$LR warmup_steps=$WARMUP_STEPS"
echo "gpus=$GPUS micro_batch=$MICRO_BATCH_SIZE grad_accum=$GRADIENT_ACCUMULATION_STEPS global_batch=$GLOBAL_BATCH_SIZE"
echo "sequence_length=$MAX_SEQ_LENGTH max_steps=$MAX_TRAIN_STEPS token_budget=$((GLOBAL_BATCH_SIZE * MAX_SEQ_LENGTH * MAX_TRAIN_STEPS))"
echo "output=$OUTPUT_DIR"
if [[ -n "${REMOTE_CHECKPOINT_DIR:-}" ]]; then
    echo "remote_checkpoint_dir=$REMOTE_CHECKPOINT_DIR"
fi
if [[ -n "${REMOTE_OUTPUT_DIR:-}" ]]; then
    echo "remote_output_dir=$REMOTE_OUTPUT_DIR/final/"
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '%q ' "${launch[@]}"
    printf '\n'
    exit 0
fi

exec "${launch[@]}"
