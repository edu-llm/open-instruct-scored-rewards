# Research image for the eduLLM platform build pipeline.
#
# What this image is for: GRPO on the pedagogy policies, including Qwen3-30B-A3B,
# rewarded by the hidden-state probe in projects/pedagogy_rm and run on AWS Batch.
#
# The platform supplies the registered base image as --build-arg BASE_IMAGE, already
# pinned to a digest. Its verify_dockerfile_base gate runs before the build and refuses
# this file unless BASE_IMAGE is declared in the global scope with no default and every
# stage roots at ${BASE_IMAGE}. The ARG is bare on purpose: a default would be a base
# image nobody registered, and the provenance record would still name the registered
# digest.
ARG BASE_IMAGE

FROM ${BASE_IMAGE}

# THE REGISTERED PYTHON BASE RATHER THAN A CUDA ONE, WHICH IS THE SAME CHOICE OLMo-core
# MADE AND FOR THE SAME REASON: a second base is a second thing to review, scan and
# re-pin. It works because the torch and vLLM wheels PyPI serves for linux x86_64 carry
# their own CUDA runtime libraries as nvidia-* dependencies, so nothing here links
# against a system CUDA install. Only the driver has to come from the host, and it does.
#
# As with OLMo-core's image, the registered base digest therefore describes a small
# fraction of what is installed and the rest is pinned by version below. A reader of the
# provenance record who takes base_image_digest as a description of this image will be
# wrong, and this is where they find out.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /opt/open-instruct

# A C COMPILER, because DeepSpeed compiles fused ops on first use and torch.compile is
# unusable without one. git because vLLM's version machinery reads it at import on some
# paths and prints a traceback when it is absent.
RUN apt-get update \
 && apt-get install --no-install-recommends --yes build-essential git \
 && rm -rf /var/lib/apt/lists/*

# THE DEPENDENCY SET IS THE CLUSTER FILE RATHER THAN uv.lock, and that is deliberate.
# The lock pins vllm==0.19.1, whose only x86_64 wheel is tagged manylinux_2_31; 0.21.0
# ships manylinux_2_24, satisfies the pyproject floor and exports every symbol
# open_instruct imports. The file carries the full argument. Installing it here rather
# than syncing the lock also keeps this image and the MIT cluster environment on one set
# of versions, which is the only way a result from one is comparable to the other.
COPY projects/pedagogy_rm/scripts/requirements-cluster.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --requirement /tmp/requirements.txt

# DEEPSPEED WILL NOT IMPORT WITHOUT A CUDA COMPILER, which is a surprise in an image
# that compiles nothing at build time. The requirements file above installs the pinned
# nvcc and carries the reasoning; this is only where the toolkit gets found. CUDA_HOME
# names the directory holding bin/nvcc, which is what installed_cuda_version() reads.
#
# The build fails here rather than in a $55/hour Batch job if any of that is wrong. It is
# the cheapest place to learn it: an import error in the container arrives after the
# queue wait, the image pull and the model download.
#
# THE LAST LINE ASSERTS WHAT THE LINE ABOVE ONLY PRINTS, and it reads both answers back out
# of the requirements file rather than restating them, so there is one copy of each pin and
# it is the copy pip installed from. Two things can go wrong and only one of them is
# covered elsewhere. The platform refuses an image whose torch reports no CUDA runtime, so
# a CPU wheel is already caught before the push; nothing outside this repository notices a
# torch that moved off the version this environment was built around, or a torch whose CUDA
# major stopped agreeing with the nvcc pinned beside it -- which is the pair DeepSpeed
# compares, and the pair that decides whether `import deepspeed` works at all.
ENV CUDA_HOME=/usr/local/lib/python3.12/site-packages/nvidia/cu13
RUN "${CUDA_HOME}/bin/nvcc" --version \
 && python -c "import accelerate, deepspeed, torch, vllm; print('accelerate', accelerate.__version__, 'deepspeed', deepspeed.__version__, 'torch', torch.__version__, 'vllm', vllm.__version__)" \
 && python -c "import re, torch; req = open('/tmp/requirements.txt').read(); pin = re.search(r'^torch==(\S+)', req, re.M).group(1); nvcc = re.search(r'^nvidia-cuda-nvcc==(\d+)[.]', req, re.M).group(1); assert torch.__version__ == pin, f'torch was moved off the pin {pin}: {torch.__version__}'; assert (torch.version.cuda or '').split('.')[0] == nvcc, f'nvidia-cuda-nvcc is pinned at CUDA {nvcc} and this torch is built for CUDA {torch.version.cuda}'" \
 && python -c "import inspect; from deepspeed import DeepSpeedEngine; from transformers import Qwen3MoeForCausalLM; from vllm import AsyncEngineArgs; assert Qwen3MoeForCausalLM; assert 'enable_expert_parallel' in inspect.signature(AsyncEngineArgs).parameters; assert 'exclude_frozen_parameters' in inspect.signature(DeepSpeedEngine.save_checkpoint).parameters"

# THE CHECKSUM A CHECKPOINT WRITE ASKS FOR IS NOT IN botocore, AND NOTHING HERE BROUGHT
# IT IN. The eduLLM platform commits a run's output by putting the payload with
# `ChecksumAlgorithm="CRC32C"` and then a `_SUCCESS` object carrying the digest S3
# attested back; its reader compares the two and calls what it cannot verify corrupt.
# That protocol is inlined into every command the platform submits, so it is not
# something this repository opts into.
#
# botocore has no pure-Python CRC-32C. `botocore.httpchecksum` adds the algorithm to its
# supported set only when `awscrt` imports, and it refuses before opening a socket when
# it cannot. boto3 depends on botocore without extras, so an image gets the client and
# not the checksum.
#
# THIS REPOSITORY IS WHERE A LATE FAILURE WOULD COST MOST: its GPU training
# profile permits one attempt up to 24 hours. A deterministic checksum failure
# after the loop would therefore leave no automatic retry and no published output.
# OLMo-core found it that way: run run_019fcf53 on 2026-08-05 trained twenty steps on a
# T4, logged every one of them, and died on the first line after the loop with
# `MissingDependencyException: Using CRC32C requires an additional dependency`.
#
# boto3 IS NAMED BESIDE IT RATHER THAN ASSUMED. requirements-cluster.txt does not pin
# boto3 and nothing in it declares botocore, so whether either arrives at all is a
# transitive accident of the vllm and ray trees. Both on one pip invocation, because one
# resolution picks a botocore the two agree on where a later separate install can move
# botocore under boto3's feet. Unpinned on purpose: a version here would be a second copy
# of what the resolver already decided, free to go stale in silence, and awscrt changes
# only how fast a fixed standard value is arrived at.
#
# TWO ASSERTIONS, BECAUSE THERE ARE TWO WAYS TO GET THIS WRONG. The install succeeding
# proves a distribution is on disk; it does not prove botocore reached it. botocore
# imports awscrt inside a try/except and sets HAS_CRT False on any failure -- an
# incompatible version, a wheel built for another libc -- so an image can carry awscrt and
# refuse CRC32C anyway, silently, until a run has already been paid for. The first
# assertion is the exact call that died, answered by Stubber so it needs no network, no
# credentials and no bucket, while botocore still resolves the checksum before the stub is
# consulted. The second is awscrt's own answer to CRC-32C of b"123456789", which the
# standard fixes at 0xE3069283.
#
# Above the COPY, like everything else here, so editing the probe or the prompts does not
# reinstall it.
RUN pip install --no-cache-dir boto3 "botocore[crt]" \
 && python -c "import boto3; from botocore import UNSIGNED; from botocore.config import Config; from botocore.stub import Stubber; client = boto3.client('s3', region_name='us-east-1', config=Config(signature_version=UNSIGNED)); stub = Stubber(client); stub.add_response('put_object', {}); stub.activate(); client.put_object(Bucket='b', Key='k', Body=b'123456789', ChecksumAlgorithm='CRC32C')" \
 && python -c "from awscrt.checksums import crc32c; assert crc32c(b'123456789') == 0xE3069283, 'awscrt computes something other than CRC-32C'"

COPY . /opt/open-instruct

# Weight sync is a silent-corruption boundary for MoE: Qwen uses separate
# expert modules while OLMoE uses stacked tensors. Exercise both layouts in the
# Linux/vLLM image, since vLLM has no macOS wheel for the local preflight.
RUN python -m pytest -q \
      open_instruct/test_moe_weight_split.py \
      projects/pedagogy_rm/test_platform_run.py \
      projects/tutor_metrics/test_probe_reward.py \
      projects/tutor_metrics/test_plugin.py \
      projects/tutor_metrics/test_upgrade_probe_head.py

# THE MODEL WEIGHTS ARE NOT BAKED IN. Qwen3-30B-A3B is about 61GB in bf16 before
# the separate probe encoder, which would make the research image needlessly large.
# Batch fetches immutable revisions from the Hub at container start. HF_HOME is local
# to the attempt, so a retry downloads again; checkpoint state, not the model cache,
# is what the platform persists.
ENV HF_HOME=/tmp/hf-cache \
    HF_HUB_CACHE=/tmp/hf-cache/hub \
    CACHE_ROOT=/tmp \
    PYTHONPATH=/opt/open-instruct

# THE MULTI-HEAD REWARD AND ITS PROMPTS ARE BAKED IN. The NPZ pins one frozen
# encoder revision and stores one tiny readout per metric; the JSONL rows carry
# the prescribed student-state target that makes contingency computable.
# `train_qwen3_30b_a3b.sh` runs a schema/context/leakage preflight before any
# model download or GPU allocation is used for training.
RUN mkdir -p data/rl \
 && cp projects/pedagogy_rm/artifacts/*.npz data/ \
 && cp projects/pedagogy_rm/artifacts/*.jsonl data/rl/ \
 && cp projects/pedagogy_rm/artifacts/rl_train.jsonl data/rl/train.jsonl \
 && cp projects/pedagogy_rm/artifacts/rl_eval.jsonl data/rl/eval.jsonl

RUN test -s data/tutor_metrics_head.npz \
 && test -s data/rl/qwen30_train.jsonl \
 && test -s data/rl/qwen30_eval.jsonl \
 && python -m projects.tutor_metrics.preflight_rl \
      --head data/tutor_metrics_head.npz \
      --train data/rl/qwen30_train.jsonl \
      --eval data/rl/qwen30_eval.jsonl \
      --expected-encoder Qwen/Qwen2.5-14B-Instruct \
      --expected-revision cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8

# No ENTRYPOINT and no CMD. The platform execs the command from the submission form,
# and a default here would be a second answer to what this image runs.
