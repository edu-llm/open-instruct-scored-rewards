#!/usr/bin/env bash
# Stable first RL arm for Qwen3-30B-A3B on one 8xA100-40GB node.
#
# Topology: four ZeRO-3 LoRA learners plus one vLLM engine at TP=4. ZeRO-3
# keeps the 61 GB base from being replicated on a 40 GB learner. Weight sync
# reconstructs only the attention matrices changed by LoRA, one at a time.
#
# The router and experts are intentionally frozen in this first arm. That
# removes router drift and guarantees that every policy update is shared across
# tokens instead of depending on which experts happened to be selected. Add
# per-expert adapters only after the routing audit shows broad coverage and the
# vLLM-vs-local logprob metrics are stable.
set -euo pipefail

export POLICY=${POLICY:-Qwen/Qwen3-30B-A3B-Instruct-2507}
export POLICY_REVISION=${POLICY_REVISION:-0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe}
export EXP=${EXP:-qwen3_30b_a3b_pedagogy_rl}

export MODE=lora
export ZERO_STAGE=${ZERO_STAGE:-3}
export GPUS=${GPUS:-8}
export LEARNERS=${LEARNERS:-4}
# hpZ duplicates the ZeRO shard when its group equals this four-rank learner
# world, leaving too little of a 40 GiB A100 for activations and gathers.
export DEEPSPEED_ZPG=${DEEPSPEED_ZPG:-1}
export ENGINES=${ENGINES:-1}
export TP=${TP:-4}
export VLLM_EP=${VLLM_EP:-1}
export VLLM_UTIL=${VLLM_UTIL:-0.55}

export LORA_R=${LORA_R:-16}
export LORA_ALPHA=${LORA_ALPHA:-32}
export LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-"q_proj k_proj v_proj o_proj"}
export LR=${LR:-1e-5}
export MICRO_BATCH=${MICRO_BATCH:-1}
export GRAD_CKPT=${GRAD_CKPT:-1}

export TRAIN_DATA=${TRAIN_DATA:-data/rl/qwen30_train.jsonl}
export EVAL_DATA=${EVAL_DATA:-data/rl/qwen30_eval.jsonl}
export HEAD=${HEAD:-data/tutor_metrics_head.npz}
export REWARD_ENCODER=${REWARD_ENCODER:-Qwen/Qwen2.5-14B-Instruct}
export REWARD_ENCODER_REVISION=${REWARD_ENCODER_REVISION:-cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8}
export SCORER=${SCORER:-tutor_metrics}
export REWARD_PLUGINS=${REWARD_PLUGINS:-projects/tutor_metrics/plugin.py}
# The reward encoder is sharded over the same four visible rollout GPUs as the
# TP=4 vLLM engine. Qwen2.5-14B costs roughly 8 GiB/card at its fullest shard;
# a 55% vLLM pool leaves measured A100 headroom for batch-two activations.
export SCORER_ARGS=${SCORER_ARGS:-"device_map=balanced,max_memory_gib=8,batch_size=2"}
# Tutor turns are not final-answer completions, so a math-answer verifier would
# punish correct answer-withholding. Correctness is represented by the probe's
# reference-conflict penalty; the JSONL's passthrough verifier is intentionally
# zero and must not be advertised as an additive reward.
export GROUP_REWARD_MODE=${GROUP_REWARD_MODE:-replace}
export APPLY_VERIFIABLE_REWARD=${APPLY_VERIFIABLE_REWARD:-0}
export GROUP_SCORER_STRICT=${GROUP_SCORER_STRICT:-1}
export REWARD_WEIGHT=${REWARD_WEIGHT:-1.0}
export LENGTH_BAND=${LENGTH_BAND:-}
export LENGTH_RAMP=${LENGTH_RAMP:-}
export LENGTH_WEIGHT=${LENGTH_WEIGHT:-0}

export PROMPTS=${PROMPTS:-32}
export SAMPLES=${SAMPLES:-8}
# The dense 14B reward forward dominates this A100 run. Measure 10k first;
# raise it only when observed step time proves a larger arm fits under 24 h.
export EPISODES=${EPISODES:-10000}
export MAX_PROMPT=${MAX_PROMPT:-1024}
export RESPONSE_LEN=${RESPONSE_LEN:-512}
export PACK_LEN=${PACK_LEN:-2048}
export BETA=${BETA:-0.02}

# Keep rollouts on the just-synced policy for the first MoE run. The rho
# correction still protects against HF/vLLM numerical and routing differences.
export INFLIGHT_UPDATES=${INFLIGHT_UPDATES:-0}
# The generic default pre-fills eight policy steps. That lag is too aggressive
# before Qwen MoE train/inference routing parity has been measured.
export ASYNC_STEPS=${ASYNC_STEPS:-1}
export RHO_CLAMP_LOWER=${RHO_CLAMP_LOWER:-0.5}
export RHO_CLAMP_UPPER=${RHO_CLAMP_UPPER:-2.0}

export EVAL_EVERY=${EVAL_EVERY:-20}
# Resume state is adapter-only under LoRA and is mirrored to S3 every minute,
# so persist every optimizer step. Full adapter exports remain every 20 steps.
export STATE_SAVE_FREQ=${STATE_SAVE_FREQ:-1}
export SAVE_FREQ=${SAVE_FREQ:-20}
export KEEP_CKPTS=${KEEP_CKPTS:-4}
export WANDB_PROJECT=${WANDB_PROJECT:-pedagogy-rm-qwen30}
export WANDB_RESPONSE_SAMPLES=${WANDB_RESPONSE_SAMPLES:-4}
export WANDB_RESPONSE_EVERY=${WANDB_RESPONSE_EVERY:-5}

if [ -n "${EDULLM_RUN_ID:-}" ]; then
    # eduLLM injects a team-scoped service key and joins W&B to the platform run
    # with WANDB_RUN_ID. Never bake a personal ~/.netrc or .wandb directory into
    # the image, and fail before model downloads if platform injection regresses.
    for variable in WANDB_API_KEY WANDB_ENTITY WANDB_PROJECT WANDB_RUN_ID; do
        if [ -z "${!variable:-}" ]; then
            echo "eduLLM W&B injection is missing $variable" >&2
            exit 2
        fi
    done
    export WANDB_MODE=online
    export WANDB_RESUME=allow
fi

for path in "$TRAIN_DATA" "$EVAL_DATA" "$HEAD"; do
    if [ ! -f "$path" ]; then
        echo "required Qwen30 RL artifact is missing: $path" >&2
        echo "override TRAIN_DATA, EVAL_DATA, or HEAD only with a validated replacement" >&2
        exit 2
    fi
done

python -u -m projects.tutor_metrics.preflight_rl \
    --head "$HEAD" \
    --train "$TRAIN_DATA" \
    --eval "$EVAL_DATA" \
    --expected-encoder "$REWARD_ENCODER" \
    --expected-revision "$REWARD_ENCODER_REVISION" \
    --minimum-train-rows "$((PROMPTS * ASYNC_STEPS))"

exec bash projects/pedagogy_rm/scripts/train.sh
