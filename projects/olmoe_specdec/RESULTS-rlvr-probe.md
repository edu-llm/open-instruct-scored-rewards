# RLVR probe — 2026-08-08

Goal: measure, on a **real** `grpo_fast.py` RLVR run, the two numbers that decide whether an
EAGLE-3 draft is worth training. No draft model is involved.

| metric | decides |
|---|---|
| `spec/favourable_iteration_fraction` (`f`) | share of forward passes at concurrency ≤ 64 |
| `spec/generation_share` (R_gen) | the Amdahl ceiling on the real recipe |

Decision rule, from the verification-cost curve in `RESULTS-baseline-sweep.md`, at α = 3:
`S_gen ≈ f·(3/1.45) + (1−f)·1.0`, then `S_step ≤ 1/(R_gen/S_gen + (1−R_gen))`.
**f ≳ 0.4 → train the draft. f ≲ 0.1 → stop.**

## Attempt 1 — `run_019fe323-9746-7017-95fa-8eec46fbb243`

`FAILED`, exit 1, after **30.7 s**. No measurement. Too early for a model load, so a config fault
rather than a resource one.

```
setup_runtime_variables -> args.hf_entity = maybe_use_ai2_hf_entity() -> HfApi().whoami()
huggingface_hub.errors.LocalTokenNotFoundError: Token is required to call the /whoami-v2 endpoint
```

**The crash is the least interesting part of this.** `grpo_fast.py` defaults
`push_to_hub: bool = True`, and that block resolves an HF entity so it can push a checkpoint to
`<entity>/open_instruct_dev`. A throughput probe with no business publishing anything was one
Hugging Face token away from **uploading a model to the Hub**. It failed safe only because this
image carries no token — that is luck, not design.

Fixed by passing `--push_to_hub false` explicitly, which is what a measurement run should have said
in the first place.

**This applies to the eventual real RLVR run too**, and to anyone else driving `grpo_fast.py`
outside Ai2: the default is to push. Decide `--push_to_hub` deliberately rather than inheriting it.

### What the rest of the startup path was audited for, instead of discovered

After the first failure the remaining startup work was read rather than probed, since each probe
costs a queue wait and a lead approval:

- `streaming_config.dataset_local_cache_dir` is redirected to a `/weka` path **only** under
  `is_beaker_job()`. Not a Beaker job, so it stays local. Safe.
- `maybe_use_ai2_wandb_entity()` is gated on `args.with_tracking`, which defaults off and is not
  passed. Safe. (Metrics therefore come from stdout, not W&B.)
- `try_launch_beaker_eval_jobs_on_weka` is `and`-ed with `is_beaker_job()`. Safe.
- Model and dataset downloads run unauthenticated, which the baseline sweep already proved works
  against this image.

Remaining un-derisked at attempt 2: the memory sizing. 6 learners under ZeRO-3 plus a reference
policy on 40 GB cards is reasoned, not measured, and OLMoE's absent GQA makes KV heavier than
parameter counts suggest. An OOM would land after the model load, a few minutes in.

## Attempt 2

Dispatched against the same published image — the command is a workflow input rather than part of
the image, so fixing it needed no rebuild.

Result pending.
