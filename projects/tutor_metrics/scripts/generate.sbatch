#!/bin/bash
#SBATCH -J tutor_moments
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
# 6h is the mit_normal_gpu ceiling, and the default is 2h, so this must be stated.
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/orcd/scratch/orcd/013/zsophia/open-instruct/logs/%x_%j.out
#SBATCH --error=/orcd/scratch/orcd/013/zsophia/open-instruct/logs/%x_%j.err
#
# Serve Qwen3-30B-A3B once, then solve the items and generate scenario-conditioned moments
# against it. Both stages in one allocation because the server takes minutes to load and the
# two stages together take less than that to run - two jobs would pay the load twice and risk
# the second never being scheduled.
#
# ONE H200 IS ENOUGH, and it is worth saying why given the model is nominally 30B. It is a
# mixture of experts: 30B of weights but 3B active per token, so 57GB of bf16 sits in 141GB of
# HBM with room for a large KV cache. Tensor parallelism would add communication for nothing.
set -euo pipefail

# Snapshot and re-exec, because bash reads a script by byte offset rather than into memory, so
# editing this file while a job runs makes it resume mid-line in the NEW bytes. That has already
# cost this repository one completed training run, recorded as FAILED after saving every
# checkpoint, when the tail of the word "compatible" in a comment became a command.
if [ -z "${TUTOR_SNAPSHOT:-}" ]; then
    snapshot="${TMPDIR:-/tmp}/tutor_gen_${SLURM_JOB_ID:-$$}.sbatch"
    cp "${BASH_SOURCE[0]}" "$snapshot"
    export TUTOR_SNAPSHOT="$snapshot"
    exec bash "$snapshot"
fi

SCRATCH=${SCRATCH:-/orcd/scratch/orcd/013/zsophia}
REPO=${REPO:-$SCRATCH/open-instruct}
MODEL=${MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}
ITEMS=${ITEMS:-$SCRATCH/grpo_tutor/data/state_tests/train_items.jsonl}
N_ITEMS=${N_ITEMS:-1000}
OUT_DIR=${OUT_DIR:-$REPO/data/tutor_metrics}
PORT=${PORT:-8311}

export HF_HOME=$SCRATCH/hf-cache HF_HUB_CACHE=$SCRATCH/hf-cache/hub
export HF_HUB_OFFLINE=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
# Keyed by hostname because flashinfer caches compiled kernels under a path every node mounts,
# and the nodes do not share a libstdc++. A kernel built on a spack-gcc node and loaded on a
# system-gcc one dies with `GLIBCXX_3.4.26 not found`, which reads like a broken install.
export FLASHINFER_CACHE_DIR=$SCRATCH/flashinfer-cache/$(hostname -s)
mkdir -p "$FLASHINFER_CACHE_DIR" "$OUT_DIR" "$REPO/logs"

set +u
command -v module >/dev/null 2>&1 || source /usr/share/lmod/lmod/init/bash
module load cuda/13.0.1
source "$SCRATCH/venv_oi/bin/activate"
set -u
cd "$REPO"
export PYTHONPATH=$REPO

echo "started $(date -Is) on $(hostname), job ${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

vllm serve "$MODEL" \
    --port "$PORT" \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --disable-log-requests \
    > "$REPO/logs/vllm_${SLURM_JOB_ID:-$$}.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT

# Poll rather than sleep a fixed time: a 57GB load varies with how warm the page cache is, and a
# fixed sleep is either wasted minutes or a race that fails on a cold node.
echo "waiting for vllm on :$PORT"
for i in $(seq 1 120); do
    if curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
        echo "vllm up after ${i}0s"; break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "FATAL: vllm died during load; see logs/vllm_${SLURM_JOB_ID:-$$}.log" >&2
        tail -30 "$REPO/logs/vllm_${SLURM_JOB_ID:-$$}.log" >&2
        exit 1
    fi
    sleep 10
done
curl -sf "http://localhost:$PORT/v1/models" >/dev/null || { echo "FATAL: vllm never came up" >&2; exit 1; }

BASE="http://localhost:$PORT/v1"

echo "=== stage 1: reference solutions ==="
python -u -m projects.tutor_metrics.solve_items \
    --items "$ITEMS" --out "$OUT_DIR/items_solved.jsonl" \
    --base-url "$BASE" --model "$MODEL" --limit "$N_ITEMS" --concurrency 48

echo "=== stage 2: scenario-conditioned moments ==="
python -u -m projects.tutor_metrics.generate \
    --items "$OUT_DIR/items_solved.jsonl" --out "$OUT_DIR/moments.jsonl" \
    --base-url "$BASE" --model "$MODEL" --concurrency 48

echo "=== stage 3: scenario fidelity and spread ==="
# PLAN.md phase 3 gate. If the student model did not occupy the state it was told to, the
# prescribed target is attached to the wrong student and the spine of the reward is measuring
# nothing. Cheap to check, and there is no point labelling anything until it passes.
python -u -m projects.tutor_metrics.audit --moments "$OUT_DIR/moments.jsonl" --base-url "$BASE" --model "$MODEL"

echo "done $(date -Is)"
