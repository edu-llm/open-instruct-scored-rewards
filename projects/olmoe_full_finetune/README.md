# OLMoE full-parameter midtraining with cosine decay

This project continues pretraining `allenai/OLMoE-1B-7B-0924` on the OLMo 2
stage-2 Dolmino mixture. It is causal-language-model midtraining over raw text,
not Tulu SFT. Every parameter remains trainable: no LoRA, QLoRA, expert
freezing, or hot-expert subset is used.

## Recipe provenance

The dense reference is AllenAI's official
`OLMo2-7B-stage2-seed42.yaml`:

- dataset: `allenai/dolmino-mix-1124`;
- 50B-token source mix: DCLM 47.2%, FLAN 16.6%, peS2o 5.85%, Wikipedia
  7.11%, StackExchange 2.45%, and stage-2 math 20.8%;
- sequence length: 4096;
- global batch: 1024 sequences;
- 11,931 optimizer steps (approximately 50B tokens);
- AdamW, betas `(0.9, 0.95)`, epsilon `1e-8`, weight decay `0.1` with input
  embeddings excluded from decay;
- peak learning rate: `6.1499e-5`;
- no warmup; and
- bf16.

The model revision is pinned to
`6d84c48581ece794365f2b8e9cfb043c68ade9c5`; the dataset is pinned to
`a319f19eef1e257417b11ea8c30da266ae175557`. The streaming loader enforces both
revisions and packs documents, separated by EOS, into complete 4096-token
training examples.

There are four deliberate adaptations:

1. the base checkpoint is OLMoE rather than OLMo 2 7B;
2. the requested scheduler is cosine-to-zero rather than the dense recipe's
   linear-to-zero schedule;
3. the OLMoE tokenizer replaces the incompatible OLMo 2 tokenizer; and
4. four H100s replace the original large launch. Gradient accumulation rises to
   preserve the 1024-sequence global batch.

The model's `0.01` load-balancing auxiliary loss and the original OLMoE `0.001`
router z-loss are retained. Gradient checkpointing and DeepSpeed ZeRO-3 reduce
memory use. Training references are deleted immediately after backward, and
unused CUDA allocator blocks are released after each optimizer step. This
cache release is conservative and may reduce throughput; set
`EMPTY_CACHE_STEPS=0` to disable it after memory behavior is established.

## Sweep

The array tests peak learning rates `4e-5`, `6.1499e-5`, and `8e-5`. It runs one
cell at a time because every cell needs the account's four-GPU limit.

Run a one-step full-parameter smoke first, then submit the sweep only if the
smoke succeeds:

```bash
cd /orcd/scratch/orcd/013/zsophia/open-instruct
bash projects/olmoe_full_finetune/submit.sh
```

`submit.sh` makes the sweep depend on successful completion of the smoke. The
job IDs are recorded under `runs/submissions.tsv`.

Inspect the current state and recent accounting:

```bash
bash projects/olmoe_full_finetune/status.sh
```

Outputs are isolated by learning rate and seed under:

```text
projects/olmoe_full_finetune/runs/
  dolmino_smoke/
  dolmino_lr_4e-5_seed_42/
  dolmino_lr_6.1499e-5_seed_42/
  dolmino_lr_8e-5_seed_42/
  logs/
```

The preemptable jobs checkpoint every 20 optimizer steps and automatically
resume from the latest complete checkpoint in the same output directory.

## Manual launch and dry run

Print the exact command without starting training:

```bash
DRY_RUN=1 bash projects/olmoe_full_finetune/train.sh
```

Submit one full run:

```bash
LR=6.1499e-5 RUN_NAME=dolmino_lr_6.1499e-5_seed_42 \
  sbatch projects/olmoe_full_finetune/train.sbatch
```

The launcher rejects a global batch that is not divisible by
`GPUS * MICRO_BATCH_SIZE`, and prints the effective batch, token budget, pinned
revisions, source mixture, scheduler, and output directory before Accelerate
starts. Checkpoints include one data-stream state per rank so a preempted run
resumes without replaying the Dolmino stream.

## AWS Batch through eduLLM

The repository root's `.edullm/run.yaml` is a one-step, eight-A100 smoke test.
The platform does not currently provision H100 profiles, so this uses
`gpu-8xa100`. It launches eight Accelerate/DeepSpeed ranks, names bf16 in the
submitted command for the hardware guard, and streams the public pinned model
and Dolmino revisions from Hugging Face.

The trainer still writes DeepSpeed checkpoints to local POSIX storage first.
On AWS, rank zero then uploads every checkpoint shard to
`$EDULLM_CHECKPOINT_DIR`, using CRC32C for multipart S3 writes. The remote
`COMPLETED` manifest is written last; a retry ignores torn uploads, downloads
the newest complete step before model construction, and restores the optimizer,
scheduler, random states, and one Dolmino stream state per rank. Once a newer
step is committed, older remote steps are pruned. The final Hugging Face export
is uploaded under `$EDULLM_OUTPUT_PREFIX/final/`.

Install the current platform CLI and run its local admission check:

```bash
uv tool install --force git+https://github.com/edu-llm/platform
edullm check --json \
  --repository open-instruct-scored-rewards \
  --team post-training \
  --experiment olmoe-dolmino-midtrain \
  --dataset none
```

Read every refusal and the live cost/approval result before submitting. After
the branch's research image has built, the smoke submission is:

```bash
edullm submit \
  --repository open-instruct-scored-rewards \
  --team post-training \
  --experiment olmoe-dolmino-midtrain \
  --dataset none
```

Use `edullm status --json` for polling. Do not call AWS directly from a laptop.
The 50B-token full recipe is not encoded in the root run specification: the
registered Open-Instruct training profile is limited to 24 hours and one
attempt, while the measured recipe is far longer even on eight cards. A full
AWS launch therefore needs an approved workload/runtime plan rather than
silently turning this smoke specification into an incomplete training run.
