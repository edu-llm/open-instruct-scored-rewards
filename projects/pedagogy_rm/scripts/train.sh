#!/bin/bash
#
# GRPO on OLMo-2-7B-Instruct, rewarded by the hidden-state probe.
#
# The invocation lives here rather than in the sbatch file because two very different
# things have to run it: a Slurm allocation on the MIT cluster, and an AWS Batch
# container on the eduLLM platform, which execs a command string and knows nothing
# about modules or venvs. Everything environment-specific is the caller's job; this
# file is the run itself.
#
# TWO TUNING MODES, AND THE CHOICE IS ABOUT MEMORY RATHER THAN ABOUT METHOD.
#
#   MODE=full  full-weight, DeepSpeed ZeRO-3 sharded across LEARNERS cards.
#   MODE=lora  low-rank adapters on a frozen base, ZeRO-2, fits on one card.
#
# What does not fit on one card is the optimizer, not the model. Adam holds fp32
# momentum, variance and master weights, so 7.3B parameters cost about 88GB of state
# before a single activation - against 14.6GB for the bf16 weights themselves. Stage 3
# divides that 88GB by the number of learners, and OFFLOAD=1 moves it to host RAM
# instead, which is what lets four 48GB cards do full-weight training at all.
#
# WHY FULL WEIGHT IS SAFE HERE, given that the reward is a linear head on one
# checkpoint's activations. The head only keeps its meaning while those activations
# hold still, and full-weight training moves the policy's. It stays valid because the
# scorer does not read the policy: it loads its own frozen copy of the encoder inside
# the vLLM actor, from the checkpoint named in data/head.npz. So the policy cannot
# raise its reward by drifting its representations - the space it is read in is not
# one it owns. What remains is a distribution risk rather than a definitional one: a
# policy trained far enough can write text unlike anything the probe was validated on,
# where its predictions mean less. That is what --beta and the held-out anchor eval
# are for, and it is the reason LoRA is still the better choice on one GPU rather than
# merely the affordable one.
#
# NO --chat_template_name AND NO --add_bos, both deliberate. Leaving the template unset
# makes open-instruct use the tokenizer's own, which is the one the head was fitted
# through; the `olmo` template in dataset_transformation.py injects function-calling
# boilerplate into the system message, which would prompt the policy in one context and
# score it in another. That template already begins with {{ bos_token }}, so --add_bos
# raises rather than being merely redundant.
set -euo pipefail

MODE=${MODE:-lora}
EXP=${EXP:-pedagogy_olmo7b}

# The encoder is deliberately not configurable: it is pinned inside data/head.npz, so
# every run - including a smoke run behind a 1B policy - scores through the same 7B the
# head was fitted on.
POLICY=${POLICY:-allenai/OLMo-2-1124-7B-Instruct}
POLICY_REVISION=${POLICY_REVISION:-}

GPUS=${GPUS:-1}
LEARNERS=${LEARNERS:-1}
ENGINES=${ENGINES:-1}
TP=${TP:-1}
VLLM_EP=${VLLM_EP:-0}
VLLM_ENFORCE_EAGER=${VLLM_ENFORCE_EAGER:-0}
OFFLOAD=${OFFLOAD:-0}

EPISODES=${EPISODES:-100000}
PROMPTS=${PROMPTS:-32}
SAMPLES=${SAMPLES:-8}
MICRO_BATCH=${MICRO_BATCH:-1}
# Recomputing activations in the backward instead of holding them. On by default because
# it is what makes a 7B fit beside vLLM and the scorer's encoder on one 80GB card; a card
# with room to spare should turn it off, since the memory it saves costs a second forward
# pass through every checkpointed block. It changes speed and memory, not the update.
GRAD_CKPT=${GRAD_CKPT:-1}
EVAL_EVERY=${EVAL_EVERY:-10}
SAVE_FREQ=${SAVE_FREQ:-50}
STATE_SAVE_FREQ=${STATE_SAVE_FREQ:-$SAVE_FREQ}
KEEP_CKPTS=${KEEP_CKPTS:-1}
WANDB_RESPONSE_SAMPLES=${WANDB_RESPONSE_SAMPLES:-0}
WANDB_RESPONSE_EVERY=${WANDB_RESPONSE_EVERY:-5}
BETA=${BETA:-0.02}
INFLIGHT_UPDATES=${INFLIGHT_UPDATES:-1}
ASYNC_STEPS=${ASYNC_STEPS:-8}
RHO_CLAMP_LOWER=${RHO_CLAMP_LOWER:-0.0}
RHO_CLAMP_UPPER=${RHO_CLAMP_UPPER:-2.0}
RHO_MASK_LOWER=${RHO_MASK_LOWER:-0.0}
RHO_MASK_UPPER=${RHO_MASK_UPPER:-0.0}
SEED=${SEED:-1}
OUTPUT_DIR=${OUTPUT_DIR:-output/$EXP}
TRAIN_DATA=${TRAIN_DATA:-data/rl/train.jsonl}
EVAL_DATA=${EVAL_DATA:-data/rl/eval.jsonl}
MAX_PROMPT=${MAX_PROMPT:-1024}
RESPONSE_LEN=${RESPONSE_LEN:-512}
PACK_LEN=${PACK_LEN:-2048}

LORA_R=${LORA_R:-32}
# MIXTURE-OF-EXPERT KNOBS, empty for a dense policy so nothing changes for the existing arms.
# EXPERT_PARAMS names fused 3-D expert tensors that target_modules cannot reach; EXPERT_R and
# EXPERT_ALPHA give them their own rank and scaling. For OLMoE the settings PERFT measured are
#   EXPERT_PARAMS="mlp.experts.gate_up_proj mlp.experts.down_proj" EXPERT_R=2 EXPERT_ALPHA=8
# which is 16.8M trainable and keeps alpha/r equal to attention's.
EXPERT_PARAMS=${EXPERT_PARAMS:-}
EXPERT_R=${EXPERT_R:-}
EXPERT_ALPHA=${EXPERT_ALPHA:-}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-}
LORA_ALPHA=${LORA_ALPHA:-64}

# Where trainer state goes so a killed run resumes instead of restarting. Empty disables
# it, which is right for a smoke run and wrong for anything long: see the flags below.
CKPT_ROOT=${CKPT_ROOT:-}

# Which registered scorer computes the reward. `pedagogy` averages the four signed
# dimensions on their raw 1-3 scales; `pedagogy_z` z-scores each one inside the group first,
# so they contribute equally rather than in proportion to their spread. plugin.py argues
# both sides. They are two arms of one experiment, not a setting with a right answer.
SCORER=${SCORER:-pedagogy}
REWARD_PLUGINS=${REWARD_PLUGINS:-projects/pedagogy_rm/plugin.py}
SCORER_ARGS=${SCORER_ARGS:-}
GROUP_REWARD_MODE=${GROUP_REWARD_MODE:-replace}
REWARD_WEIGHT=${REWARD_WEIGHT:-1.0}
GROUP_SCORER_STRICT=${GROUP_SCORER_STRICT:-0}
GROUP_MAX_SCORE=${GROUP_MAX_SCORE:-1.0}
APPLY_VERIFIABLE_REWARD=${APPLY_VERIFIABLE_REWARD:-1}
ENABLE_QUEUE_DASHBOARD=${ENABLE_QUEUE_DASHBOARD:-1}

# Which fitted head, and whether a length band is added on top of it. Both are arguments to
# the scorer rather than edits to it, so an arm is a wrapper that exports three variables.
#
# head.npz scores four dimensions; head5.npz adds `correct`, which was dropped in the first
# round for missing its agreement gate and re-added after the rewritten rubric reached 0.43.
#
# LENGTH_BAND IS EMPTY BY DEFAULT, so the two finished arms still reproduce exactly. It is a
# flat in-band bonus in whitespace words, and it exists because every dimension the head
# scores improves as a turn gets shorter - -0.26 per log-word over the five, and the human's
# own ratings agree at -0.49. No reweighting of monotone terms can say "now it is too short".
HEAD=${HEAD:-data/head.npz}
LENGTH_BAND=${LENGTH_BAND:-}
LENGTH_WEIGHT=${LENGTH_WEIGHT:-0}
# Empty means a hard band, which is what arm C ran. A pair like "8-90" makes the term fall
# linearly to zero at those word counts instead of cliff-edging, so 200 words costs more than 60.
LENGTH_RAMP=${LENGTH_RAMP:-}

scorer_spec="$SCORER:head=$HEAD"
[ "$REWARD_WEIGHT" != "1.0" ] && scorer_spec="$scorer_spec,reward_weight=$REWARD_WEIGHT"
[ -n "$SCORER_ARGS" ] && scorer_spec="$scorer_spec,$SCORER_ARGS"
if [ -n "$LENGTH_BAND" ]; then
    scorer_spec="$scorer_spec,length_band=$LENGTH_BAND,length_weight=$LENGTH_WEIGHT"
    [ -n "$LENGTH_RAMP" ] && scorer_spec="$scorer_spec,length_ramp=$LENGTH_RAMP"
fi

CACHE_ROOT=${CACHE_ROOT:-${SCRATCH:-/tmp}}
export HF_HOME=${HF_HOME:-$CACHE_ROOT/hf-cache}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$CACHE_ROOT/hf-cache/hub}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$CACHE_ROOT/xdg-cache}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$CACHE_ROOT/triton-cache}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$CACHE_ROOT/inductor-cache}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=${PYTHONPATH:-$PWD}

# Ray places LEARNERS learner actors and ENGINES vLLM engines of TP cards each, so the
# three have to add up to the machine. They are checked here rather than left to fail
# somewhere inside Ray placement, where the error is a group that never becomes ready
# rather than a sentence about arithmetic. The platform runs its own version of this
# check before a job is ever submitted; this one is for the clusters that do not.
NEEDED=$((LEARNERS + ENGINES * TP))
COLOCATE=0
if [ "$NEEDED" -gt "$GPUS" ]; then
    if [ "$GPUS" -eq 1 ] && [ "$LEARNERS" -eq 1 ] && [ "$ENGINES" -eq 1 ] && [ "$TP" -eq 1 ]; then
        # The one oversubscription that works: --single_gpu_mode gives the learner 0.48
        # of the card and vLLM 0.5, and the weight sync drops from NCCL to CUDA IPC
        # because both live on the same device.
        COLOCATE=1
    else
        echo "LEARNERS=$LEARNERS + ENGINES=$ENGINES x TP=$TP needs $NEEDED cards, have GPUS=$GPUS" >&2
        exit 2
    fi
elif [ "$NEEDED" -lt "$GPUS" ]; then
    echo "LEARNERS=$LEARNERS + ENGINES=$ENGINES x TP=$TP uses $NEEDED of $GPUS cards; $((GPUS - NEEDED)) would idle" >&2
    exit 2
fi

tuning=()
[ -n "$POLICY_REVISION" ] && tuning+=(--model_revision "$POLICY_REVISION")
if [ "$MODE" = full ]; then
    # 3e-7 rather than the 1e-5 LoRA wants: this update lands on the weights themselves
    # rather than on a low-rank adapter starting from zero.
    tuning+=(--learning_rate "${LR:-3e-7}" --deepspeed_stage 3)
    if [ "$OFFLOAD" = 1 ]; then
        # Adam on the host. Costs step time - the copy is over PCIe and the update is on
        # CPU - and buys back 88GB/LEARNERS of device memory, which is the difference
        # between fitting and not on any 48GB card.
        tuning+=(--deepspeed_offload_optimizer)
    fi
elif [ "$MODE" = lora ]; then
    # Stage 2 is fastest when the frozen base fits on one learner. Stage 3 is
    # the A100-40GB path: grpo_fast gathers one LoRA target at a time and sends
    # base+delta to vLLM without ever materializing the whole 30B checkpoint.
    tuning+=(
        --learning_rate "${LR:-1e-5}"
        --deepspeed_stage "${ZERO_STAGE:-2}"
        --use_peft
        --lora_r "$LORA_R"
        ${LORA_TARGET_MODULES:+--lora_target_modules $LORA_TARGET_MODULES}
        ${EXPERT_PARAMS:+--lora_target_parameters $EXPERT_PARAMS}
        ${EXPERT_R:+--lora_expert_rank "$EXPERT_R"}
        ${EXPERT_ALPHA:+--lora_expert_alpha "$EXPERT_ALPHA"}
        --lora_alpha "$LORA_ALPHA"
        --lora_dropout 0.0
    )
else
    echo "MODE must be full or lora, got '$MODE'" >&2
    exit 2
fi
[ -n "${DEEPSPEED_ZPG:-}" ] && tuning+=(--deepspeed_zpg "$DEEPSPEED_ZPG")

[ "$COLOCATE" = 1 ] && tuning+=(--single_gpu_mode)
[ "$VLLM_EP" = 1 ] && tuning+=(--vllm_enable_expert_parallel True)

# --push_to_hub defaults to TRUE upstream, which publishes the trained policy to the
# Hub under whatever account the environment happens to be logged into. It also fails
# before training starts: setup_runtime_variables resolves the target repo by calling
# HfApi().whoami(), so on a cluster with HF_HUB_OFFLINE set the run dies in argument
# handling rather than at the end.
tuning+=(--push_to_hub False)

# Tracking is off unless a project is named. The eduLLM platform always names one - it
# is a required field on the submission form and the W&B key reaches the container
# through its execution role. On the MIT nodes there is no route to the internet, so the
# caller sets WANDB_MODE=offline and the run writes a local directory to sync later;
# without that the reward curve, which is the actual output of the experiment, exists
# only in a terminal.
if [ -n "${WANDB_PROJECT:-}" ]; then
    tuning+=(--with_tracking --wandb_project "$WANDB_PROJECT")
    # Passed explicitly wherever it is known, because leaving it unset makes
    # open-instruct call wandb.login() to check for an Ai2 team it will not find - a
    # network round trip that fails on an offline node before training starts.
    [ -n "${WANDB_ENTITY:-}" ] && tuning+=(--wandb_entity "$WANDB_ENTITY")
fi

# RESUMABLE STATE, WHICH ON A PREEMPTABLE PARTITION IS THE DIFFERENCE BETWEEN A REQUEUE
# COSTING MINUTES AND COSTING THE RUN. mit_preemptable really does preempt - a smoke run
# was taken at 2:13 and requeued - and Slurm restarts the script from the top, so without
# this the second attempt begins at step zero. grpo_fast resumes on its own when the
# directory holds state: it reads optimization_steps_done and continues from the step
# after. State can be saved more often than an exported model: under LoRA the state
# excludes frozen parameters and is cheap, while an export still gathers adapters.
if [ -n "$CKPT_ROOT" ]; then
    tuning+=(--checkpoint_state_dir "$CKPT_ROOT/$EXP" --checkpoint_state_freq "$STATE_SAVE_FREQ")
fi

# Tested against "0" explicitly rather than for emptiness, because "0" is a non-empty
# string and ${GRAD_CKPT:+...} would cheerfully add the flag it was meant to remove.
[ "$GRAD_CKPT" != "0" ] && tuning+=(--gradient_checkpointing)

# vLLM's share of a card it has to itself can be generous; a card it shares with a
# learner and with the scorer's frozen encoder cannot. The caller sets it, because only
# the caller knows the card.
VLLM_UTIL=${VLLM_UTIL:-$([ "$COLOCATE" = 1 ] && echo 0.30 || echo 0.55)}

echo "mode=$MODE policy=$POLICY revision=${POLICY_REVISION:-default}" \
     "gpus=$GPUS learners=$LEARNERS engines=${ENGINES}x${TP} expert_parallel=$VLLM_EP" \
     "offload=$OFFLOAD colocate=$COLOCATE vllm_util=$VLLM_UTIL"

exec python -u open_instruct/grpo_fast.py \
    --exp_name "$EXP" \
    --model_name_or_path "$POLICY" \
    --tokenizer_name_or_path "$POLICY" \
    --use_slow_tokenizer False \
    --dataset_mixer_list "$TRAIN_DATA" 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list "$EVAL_DATA" 1.0 \
    --dataset_mixer_eval_list_splits train \
    --reward_plugins "$REWARD_PLUGINS" \
    --group_scorer "$scorer_spec" \
    --group_reward_mode "$GROUP_REWARD_MODE" \
    --group_scorer_strict "$([ "$GROUP_SCORER_STRICT" = 1 ] && echo True || echo False)" \
    --group_max_possible_score "$GROUP_MAX_SCORE" \
    --apply_verifiable_reward "$([ "$APPLY_VERIFIABLE_REWARD" = 1 ] && echo True || echo False)" \
    --max_prompt_token_length "$MAX_PROMPT" \
    --response_length "$RESPONSE_LEN" \
    --pack_length "$PACK_LEN" \
    --num_unique_prompts_rollout "$PROMPTS" \
    --num_samples_per_prompt_rollout "$SAMPLES" \
    --wandb_response_samples "$WANDB_RESPONSE_SAMPLES" \
    --wandb_response_log_every "$WANDB_RESPONSE_EVERY" \
    --temperature 1.0 \
    --beta "$BETA" \
    --inflight_updates "$([ "$INFLIGHT_UPDATES" = 1 ] && echo True || echo False)" \
    --async_steps "$ASYNC_STEPS" \
    --use_rho_correction True \
    --rho_clamp_lower_bound "$RHO_CLAMP_LOWER" \
    --rho_clamp_upper_bound "$RHO_CLAMP_UPPER" \
    --rho_mask_lower_bound "$RHO_MASK_LOWER" \
    --rho_mask_upper_bound "$RHO_MASK_UPPER" \
    --enable_queue_dashboard "$([ "$ENABLE_QUEUE_DASHBOARD" = 1 ] && echo True || echo False)" \
    "${tuning[@]}" \
    --lr_scheduler_type constant_with_warmup \
    --warmup_ratio 0.03 \
    --total_episodes "$EPISODES" \
    --per_device_train_batch_size "$MICRO_BATCH" \
    --num_mini_batches 1 \
    --num_epochs 1 \
    --num_learners_per_node "$LEARNERS" \
    --vllm_num_engines "$ENGINES" \
    --vllm_tensor_parallel_size "$TP" \
    --vllm_gpu_memory_utilization "$VLLM_UTIL" \
    --vllm_enforce_eager "$([ "$VLLM_ENFORCE_EAGER" = 1 ] && echo True || echo False)" \
    --local_eval_every "$EVAL_EVERY" \
    --save_freq "$SAVE_FREQ" \
    --keep_last_n_checkpoints "$KEEP_CKPTS" \
    --output_dir "$OUTPUT_DIR" \
    --seed "$SEED"
