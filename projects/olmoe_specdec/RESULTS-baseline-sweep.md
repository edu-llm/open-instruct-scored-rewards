# Baseline generation sweep — 2026-08-08

Two autoregressive arms, no draft model, both `SUCCEEDED` exit 0.

| run | shape | wall clock |
|---|---|---|
| `run_019fe2f0-dc6c-70e6-ab47-ecd4cc12e3d1` | `gpu-8xa100`, TP=2 | 8.3 min |
| `run_019fe2f1-2a77-7030-8eaf-2c5414e5847e` | `gpu-1xl40s`, TP=1 | 19.4 min |

`allenai/OLMoE-1B-7B-0125-DPO`, real RLVR-GSM prompts through the tulu template, bf16,
temperature 1.0, `min_tokens = max_tokens` so every cell generates its full length.

**`35 passed`, `PYTEST_EXIT=0` on both shapes** — the first execution anywhere of
`test_package_imports_vllm.py`, `test_olmoe_eagle3.py` and `test_metrics_vllm.py`. That is what
verifies the `@support_torch_compile` subclassing argument on real vLLM rather than by simulation:
the compile wrapper appears exactly once in the MRO, the subclass defines no `__init__`,
`load_weights` is inherited by object identity, `supports_eagle3` is true for ours and false for
upstream's, and the default auxiliary layers are `(2, 8, 13)`.

## Throughput (tok/s)

| offered batch | L40S len 256 | L40S len 1024 | A100 len 256 | A100 len 1024 |
|---:|---:|---:|---:|---:|
| 16 | 950.0 | 879.1 | 2,297.4 | 2,198.0 |
| 64 | 2,704.2 | 2,309.9 | 6,552.9 | 5,947.4 |
| 256 | 6,990.3 | 4,372.9 | 14,815.1 | 12,544.4 |
| 768 | 7,085.1 | 4,451.0 | 15,406.4 | 12,547.5 |

## Saturation

From 256 to 768 offered sequences, throughput does not move and latency triples:

| | throughput | latency |
|---|---|---|
| L40S len 256 | +1.36% | ×2.96 |
| L40S len 1024 | +1.79% | ×2.95 |
| A100 len 256 | +3.99% | ×2.88 |
| A100 len 1024 | **+0.02%** | **×3.00** |

The last row is a textbook saturated pipeline: 3× the offered load, 3.00× the latency, 0.02% more
work done. **Offered batch above ~256 is not concurrency, it is queueing**, so those cells measure
the engine's ceiling (≈12.5k tok/s on A100 TP=2 at length 1024) rather than a wider forward pass.
This is also why the 256→768 column cannot be read as a verification-cost measurement.

## The verification-cost curve, which is the point

Speculative decoding replaces `α` sequential forwards with one forward carrying `k+1` tokens per
sequence. A forward with `B` sequences × `k+1` tokens costs about what a decode forward at
`B·(k+1)` sequences costs — so **this table is the verification-cost curve**, and it can be read
without training a draft.

Per-forward time is `t(B) = B / throughput(B)`. Break-even acceptance length at concurrency `B` is
`t(4B)/t(B)` for `k=3`:

| concurrency | A100 len 256 | A100 len 1024 | L40S len 256 | L40S len 1024 |
|---|---|---|---|---|
| 16 | 1.40 | 1.48 | 1.41 | 1.52 |
| 64 | 1.77 | 1.90 | 1.55 | 2.11 |
| 256 | saturated | saturated | saturated | saturated |

EAGLE-3 reaches **α ≈ 2.7–3.3** in-domain (arXiv:2604.26779 Tables 3–5). At α = 3.0 that implies a
*generation-stage* speedup of:

| | concurrency 16 | concurrency 64 |
|---|---|---|
| A100 len 256 | **2.14×** | **1.70×** |
| A100 len 1024 | 2.03× | 1.58× |
| L40S len 256 | 2.13× | 1.94× |
| L40S len 1024 | 1.97× | 1.42× |

In the saturated regime the cost of extra tokens per forward is essentially linear, so no
achievable α wins there and drafting overhead makes it a net loss.

**Correction to an earlier reading of this data.** A first pass quoted break-even ≈ 3.9 at batch
256, obtained by extrapolating the 3× step (256→768) as though it were 4×. That step is
queue-contaminated and cannot support the inference. The measured 4× steps are the two rows above,
and they are considerably more favourable to speculation.

## Verdict: not a null, and not yet a go

Speculative decoding on this workload is decided by one unmeasured quantity: **how much of a real
rollout runs at low concurrency.** The curve above says it wins by 1.4–2.1× on generation below
concurrency 64 and cannot win once saturated. Nothing here says which regime an RLVR step spends
its time in, because this sweep forced `min_tokens = max_tokens`, holding all 768 sequences alive
to the end and so reporting the maximally unfavourable concurrency by construction.

A real RLVR-GSM step does the opposite. Answers are short, they stop on `</answer>` at staggered
times, and the step drains from its nominal 768 toward zero — and the draining tail is the cheap
regime. **So these numbers are a lower bound on speculation's value, not an estimate of it.**

### The deciding measurement

`submissions/rlvr-probe-a100.json` — a real 20-step `grpo_fast.py` run on the full recipe, no draft
model. It reports `spec/generation_share` (R_gen on the real recipe),
`spec/favourable_iteration_fraction` (share of forward passes at concurrency ≤ 64), the full
`spec/concurrency_le_*` histogram, and the real response-length distribution.

Combine as: `S_gen ≈ f · (α/1.45) + (1−f) · 1.0`, then `S_step ≤ 1/(R_gen/S_gen + (1−R_gen))`.

Worked, at α = 3:

| f | S_gen | S_step at R_gen 0.5 | at R_gen 0.7 |
|---|---|---|---|
| 0.7 | 1.75× | 1.27× | 1.41× |
| 0.4 | 1.43× | 1.18× | 1.28× |
| 0.1 | 1.11× | 1.05× | 1.07× |

So `f ≳ 0.4` makes this worth building; `f ≲ 0.1` does not, and the draft never needs training.

## Operational notes

- The engine's max sustainable throughput, not its per-sequence rate, is what bounds a rollout:
  ≈15.4k tok/s (A100 TP=2, len 256) and ≈12.5k (len 1024).
- **OLMoE has no GQA** — 16 query heads, 16 KV heads, head_dim 128 — so KV costs ~128 KB/token.
  At length 1024 that is ~118 GB for 768 sequences against 67 GB of usable KV on A100 TP=2, which
  is why offered 768 queues. On 40 GB cards KV binds before weights do, and that should drive the
  RL learner/engine split rather than parameter memory.
- Results are published to `$EDULLM_OUTPUT_PREFIX/sweep_generation.json` (CRC32C) and to stdout
  after every cell. The S3 write is confirmed working; the per-cell stdout copy is what makes a
  timeout survivable.
