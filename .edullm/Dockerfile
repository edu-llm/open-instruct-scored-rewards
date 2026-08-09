# Research image for the eduLLM platform build pipeline.
#
# What this image is for: GRPO on OLMo-2-7B-Instruct rewarded by the hidden-state probe
# in projects/pedagogy_rm, run on AWS Batch. See projects/pedagogy_rm/PLATFORM.md for
# the submission itself.
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
ENV CUDA_HOME=/usr/local/lib/python3.12/site-packages/nvidia/cu13
RUN "${CUDA_HOME}/bin/nvcc" --version \
 && python -c "import deepspeed, torch, vllm; print('deepspeed', deepspeed.__version__, 'torch', torch.__version__, 'vllm', vllm.__version__)"

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
# THIS REPOSITORY IS THE ONE WHERE IT WOULD COST MOST, because open-instruct-scored-rewards
# -train is the only workload profile outside pre-training that declares two attempts
# against a checkpoint contract. The failure is deterministic and arrives after the
# training loop, so both attempts reach the same line and neither keeps anything.
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

# ONE ENTRY POINT, WHICH IS THE ONLY WAY TO REACH vLLM'S SUBPROCESSES. vLLM resolves a model
# architecture through a per-process ModelRegistry, and under AsyncLLM the model is built in
# neither this image's entrypoint nor the Ray actor: EngineCore is a subprocess and its
# tensor-parallel workers are children of that. An override registered anywhere else is registered
# in a process that never instantiates a model.
#
# vllm.general_plugins is loaded by vLLM in every one of those processes at the point in its own
# startup where importing vLLM machinery is safe, and it is discovered through installed
# distribution metadata -- which this image otherwise has none of, because it COPYs the tree and
# sets PYTHONPATH instead of installing it. Hence one tiny package carrying one entry point.
#
# --no-deps because it declares none and none are wanted: vllm and open_instruct are already here,
# and letting pip resolve would risk disturbing a dependency set pinned above on purpose.
# The import is asserted at build time so a broken plugin fails here rather than as a model that
# quietly loads through vLLM's own class an hour into a GPU run.
RUN pip install --no-cache-dir --no-deps /opt/open-instruct/packaging/olmoe_vllm_plugin \
 && python -c "import importlib.metadata as m; eps=[e for e in m.entry_points(group='vllm.general_plugins')]; assert any(e.name=='olmoe_eagle3' for e in eps), f'entry point missing, found {eps}'; print('vllm.general_plugins:', [e.name for e in eps])"

# THE MODEL WEIGHTS ARE NOT BAKED IN. OLMo-2-7B-Instruct is about 15GB, which would take
# this image past what a standard GitHub runner can build, and the Batch instances sit in
# public subnets with a route to an internet gateway, so the container fetches them from
# the Hub on first use. HF_HOME is on the container filesystem rather than anywhere
# shared: every attempt downloads its own copy, which costs minutes of a run that lasts
# hours.
ENV HF_HOME=/tmp/hf-cache \
    HF_HUB_CACHE=/tmp/hf-cache/hub \
    CACHE_ROOT=/tmp \
    PYTHONPATH=/opt/open-instruct

# THE PROBE AND ITS PROMPTS ARE BAKED IN, because they are what makes this run this run.
# head.npz is the fitted reward - 225KB of Ridge coefficients plus the metadata naming
# the encoder checkpoint, pooling and layer each dimension reads - and the two jsonl
# files are the tutoring moments the policy is asked to respond to. They live under
# projects/pedagogy_rm/artifacts/ in git because the repository's .gitignore excludes
# data/, and are copied to the paths scripts/train.sh expects so that the same command
# line runs here and on a cluster where data/ is real.
RUN mkdir -p data/rl \
 && cp projects/pedagogy_rm/artifacts/head.npz data/head.npz \
 && cp projects/pedagogy_rm/artifacts/rl_train.jsonl data/rl/train.jsonl \
 && cp projects/pedagogy_rm/artifacts/rl_eval.jsonl data/rl/eval.jsonl

# No ENTRYPOINT and no CMD. The platform execs the command from the submission form,
# and a default here would be a second answer to what this image runs.
