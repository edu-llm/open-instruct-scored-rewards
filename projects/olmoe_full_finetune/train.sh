#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${SOURCE_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
REPO="${REPO:-$(cd "$PROJECT_DIR/../.." && pwd)}"
cd "$REPO"

MODEL="${MODEL:-allenai/OLMoE-1B-7B-0924}"
MODEL_REVISION="${MODEL_REVISION:-6d84c48581ece794365f2b8e9cfb043c68ade9c5}"
DATASET="${DATASET:-allenai/tulu-3-sft-olmo-2-mixture-0225}"
DATASET_REVISION="${DATASET_REVISION:-d91a0785ade02942520280fb484866fce41e448f}"

LR="${LR:?set LR to the peak learning rate, for example 2e-5}"
SEED="${SEED:-8}"
GPUS="${GPUS:-4}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-4096}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-2}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-20}"

for value_name in GPUS MICRO_BATCH_SIZE GLOBAL_BATCH_SIZE MAX_SEQ_LENGTH CHECKPOINTING_STEPS; do
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
DATASET_CACHE="${DATASET_CACHE:-${SCRATCH:-$REPO}/open-instruct-data-cache}"
mkdir -p "$OUTPUT_DIR" "$DATASET_CACHE"

train_args=(
    open_instruct/finetune.py
    --exp_name "$RUN_NAME"
    --model_name_or_path "$MODEL"
    --model_revision "$MODEL_REVISION"
    --tokenizer_name "$MODEL"
    --tokenizer_revision "$MODEL_REVISION"
    --use_slow_tokenizer False
    --add_bos
    --dataset_mixer_list "$DATASET" 1.0
    --dataset_mixer_list_splits train
    --dataset_local_cache_dir "$DATASET_CACHE"
    --max_seq_length "$MAX_SEQ_LENGTH"
    --preprocessing_num_workers "${PREPROCESSING_NUM_WORKERS:-32}"
    --per_device_train_batch_size "$MICRO_BATCH_SIZE"
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
    --learning_rate "$LR"
    --lr_scheduler_type cosine
    --warmup_ratio "$WARMUP_RATIO"
    --weight_decay 0.0
    --num_train_epochs "$NUM_TRAIN_EPOCHS"
    --output_dir "$OUTPUT_DIR"
    --do_not_randomize_output_dir
    --checkpointing_steps "$CHECKPOINTING_STEPS"
    --keep_last_n_checkpoints 1
    --logging_steps 1
    --load_balancing_loss
    --gradient_checkpointing
    --seed "$SEED"
    --timeout "${PROCESS_GROUP_TIMEOUT:-7200}"
    --push_to_hub False
    --verbose
)

if [[ -n "${MAX_TRAIN_STEPS:-}" ]]; then
    train_args+=(--max_train_steps "$MAX_TRAIN_STEPS")
fi
if [[ -n "${MAX_TRAIN_SAMPLES:-}" ]]; then
    train_args+=(--max_train_samples "$MAX_TRAIN_SAMPLES")
fi
if [[ "${WITH_TRACKING:-1}" == "1" ]]; then
    train_args+=(
        --with_tracking
        --report_to wandb
        --wandb_project_name "${WANDB_PROJECT:-olmoe-full-finetune}"
        --wandb_entity "${WANDB_ENTITY:-eduLLM}"
    )
fi

launch=(
    accelerate launch
    --mixed_precision bf16
    --num_processes "$GPUS"
    --use_deepspeed
    --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf
    "${train_args[@]}"
)

echo "run=$RUN_NAME model=$MODEL@$MODEL_REVISION"
echo "dataset=$DATASET@$DATASET_REVISION"
echo "full_parameters=true scheduler=cosine peak_lr=$LR warmup_ratio=$WARMUP_RATIO"
echo "gpus=$GPUS micro_batch=$MICRO_BATCH_SIZE grad_accum=$GRADIENT_ACCUMULATION_STEPS global_batch=$GLOBAL_BATCH_SIZE"
echo "output=$OUTPUT_DIR"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '%q ' "${launch[@]}"
    printf '\n'
    exit 0
fi

exec "${launch[@]}"
