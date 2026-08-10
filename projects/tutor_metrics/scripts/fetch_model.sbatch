#!/bin/bash
#SBATCH -J qwen3_fetch
#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/orcd/scratch/orcd/013/zsophia/open-instruct/logs/%x_%j.out
#SBATCH --error=/orcd/scratch/orcd/013/zsophia/open-instruct/logs/%x_%j.err
#
# Fetch Qwen3-30B-A3B-Instruct-2507 into the shared cache, on a CPU partition.
#
# ON A CPU NODE BECAUSE A GPU SPENT DOWNLOADING IS A GPU WASTED. This is ~60GB over the network
# and touches no accelerator; the generation job that follows takes the H200 and expects the
# weights already local. Splitting them also means a failed download does not lose a GPU
# allocation that took hours to schedule.
#
# HF_HUB_OFFLINE IS DELIBERATELY UNSET HERE, and it is the whole point of this script. Every
# other job in this repository pins it to 1 so a cache miss fails loudly rather than silently
# fetching a different revision. That pin is why the RLVR run died in nineteen seconds: it
# needed a dataset nobody had fetched. This job is the one place allowed to reach the network,
# so the pinned jobs can stay pinned.
set -euo pipefail

SCRATCH=${SCRATCH:-/orcd/scratch/orcd/013/zsophia}
export HF_HOME=$SCRATCH/hf-cache HF_HUB_CACHE=$SCRATCH/hf-cache/hub
export HF_HUB_ENABLE_HF_TRANSFER=1
MODEL=${MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}

set +u
command -v module >/dev/null 2>&1 || source /usr/share/lmod/lmod/init/bash
source "$SCRATCH/venv_oi/bin/activate"
set -u

echo "fetching $MODEL into $HF_HUB_CACHE at $(date -Is)"
df -h "$SCRATCH" | tail -1

python - <<'PY'
import os
from huggingface_hub import snapshot_download

model = os.environ.get("MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
# allow_patterns keeps the .pt and .gguf mirrors some repos carry out of a 60GB download.
path = snapshot_download(
    model,
    allow_patterns=["*.safetensors", "*.json", "*.txt", "*.jinja", "*.model"],
    max_workers=8,
)
print("cached at", path)

# Assert the tokeniser has a chat template. An OLMoE run in this repository failed at startup
# because HF_HUB_OFFLINE meant the template was never fetched, and the error named the template
# rather than the cache - two hours to find. Checking here makes the download job own that.
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(path)
assert tok.chat_template, f"{model} has no chat template; the offline generation job will fail"
print("chat template present, vocab", len(tok))
PY

echo "done $(date -Is)"
