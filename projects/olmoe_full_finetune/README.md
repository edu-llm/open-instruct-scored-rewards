# OLMoE full-parameter SFT with cosine decay

This project applies the Open-Instruct OLMo 2 7B SFT recipe to the native
`allenai/OLMoE-1B-7B-0924` checkpoint. It is full-parameter training: no LoRA,
QLoRA, expert freezing, or hot-expert subset is used.

## Recipe provenance

The dense reference is `scripts/train/olmo2/finetune_7b.sh`:

- dataset: `allenai/tulu-3-sft-olmo-2-mixture-0225`, weight 1.0;
- sequence length: 4096;
- global batch: 128 sequences;
- AdamW, weight decay 0;
- peak learning rate: `2e-5`;
- warmup: 3%;
- two epochs;
- bf16; and
- DeepSpeed ZeRO-3.

The model revision is pinned to
`6d84c48581ece794365f2b8e9cfb043c68ade9c5`. The dataset revision observed
when this launcher was written was
`d91a0785ade02942520280fb484866fce41e448f`; Open-Instruct does not expose a
dataset-revision argument, so the revision is recorded in run metadata rather
than enforced by the loader.

There are three deliberate changes from the dense reference:

1. the base checkpoint is OLMoE rather than OLMo 2 7B;
2. the post-warmup schedule is cosine rather than linear; and
3. four H100s replace the original 64-GPU launch. Gradient accumulation rises
   from 2 to 32 so the global batch remains exactly 128.

Gradient checkpointing is enabled to leave memory headroom on four GPUs.
`--load_balancing_loss` asks Transformers to return router logits; OLMoE then
adds its checkpoint-configured router auxiliary coefficient to the language
model loss.

## Sweep

The array tests peak learning rates `1.5e-5`, `1.75e-5`, and `2e-5`. It runs
one cell at a time because every cell needs the account's four-GPU limit.

Run a three-step full-parameter smoke first, then submit the sweep only if the
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
  smoke/
  lr_1.5e-5_seed_8/
  lr_1.75e-5_seed_8/
  lr_2e-5_seed_8/
  logs/
```

The preemptable jobs checkpoint every 20 optimizer steps and automatically
resume from the latest complete checkpoint in the same output directory.

## Manual launch and dry run

Print the exact command without starting training:

```bash
LR=2e-5 DRY_RUN=1 bash projects/olmoe_full_finetune/train.sh
```

Submit one full run:

```bash
LR=2e-5 RUN_NAME=lr_2e-5_seed_8 \
  sbatch projects/olmoe_full_finetune/train.sbatch
```

The launcher rejects a global batch that is not divisible by
`GPUS * MICRO_BATCH_SIZE`, and prints the effective batch, model revision,
dataset revision, scheduler, and output directory before Accelerate starts.
