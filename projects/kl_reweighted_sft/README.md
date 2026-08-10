# kl_reweighted_sft

SFT that spends less gradient on the tokens the frozen base model found most surprising, so the
tuned model lands closer to where it started. One knob, a temperature `T`: as `T -> inf` this is
ordinary SFT exactly, and lower `T` reweights harder.

The method was developed on OLMo-2-1B and OLMoE-1B-7B; what those runs taught us is recorded
below, because most of the decisions here are inherited from measurements rather than chosen
fresh.

## Which base model

Two candidates, both supported. Nothing in the method cares which; the differences are all in
loading and in how much card the weights leave over.

| | `openai/gpt-oss-20b` | `Qwen/Qwen3-30B-A3B-Instruct-2507` |
|---|---|---|
| total / active | 21B / 3.6B | 30.5B / 3.3B |
| **bf16 weights** | **~42GB** | **~61GB** |
| layers, experts | 24, 32 experts top-4 | 48, 128 experts top-8 |
| vocab | 201k | 152k |
| ships as | MXFP4, needs `Mxfp4Config(dequantize=True)` | bf16, loads as-is |
| attention | eager only — its sinks rule out FlashAttention-2 | FlashAttention-2 works |
| router parameter | `mlp.router` | `mlp.gate` |

Qwen is the easier model in every respect but one: no dequantisation step, a faster attention
kernel, and a smaller vocabulary that makes variant b's per-token KL cheaper, since that
intermediate is (positions x vocab) and is the largest transient in the precompute.

The exception is the one that matters. **61GB against 42GB on an 80GB card** leaves roughly
19GB for activations, the LoRA state and the CUDA context, so Qwen wants gradient checkpointing
and a small per-device batch, and it has much less headroom before a long sequence tips it into
sharding. gpt-oss has room to spare on the same card. Both are fine on `gpu-8xh100`, where the
eight-way data parallelism does the throughput work either way.

Expert layout is identical across both, and OLMoE: transformers v5 stores experts as fused
`(num_experts, ...)` Parameters named `mlp.experts.gate_up_proj` and `mlp.experts.down_proj`.
Only the router name differs, which is why the freeze check is keyed by `model_type`.

## The method

For every loss-bearing token `t` we compute a distance-from-base `s_t` and turn it into a
multiplier via a temperatured softmax of `-s_t`, normalised to mean 1 so the effective learning
rate is unchanged and only the *distribution* of gradient across tokens moves.

| variant | signal | needs |
|---|---|---|
| a | `s_t = -log pi_0(y_t \| ctx)` — base surprise | one frozen-base pass |
| b | `s_t = KL(pi_0 \|\| pi_SFT)` — forward KL | a vanilla SFT adapter |

**Only variant b is being run here.** Variant a was unstable at every scale we tried it: at
OLMoE its T=4 arm finished with a held-out pedagogy NLL of 1.635 against the *untrained base's*
1.451, meaning it did not merely learn slowly, it ended up worse than not training at all. It
also lost math where every other arm gained it.

## Run matrix

Five runs. The control is trained first because variant b's signal is defined against it.

| run | variant | T | purpose |
|---|---|---|---|
| `sft-control` | — | — | vanilla SFT; the reference for variant b and the baseline to beat |
| `b-T0.5` | b | 0.5 | hardest reweighting |
| `b-T1` | b | 1 | |
| `b-T2` | b | 2 | |
| `b-T4` | b | 4 | lightest reweighting |

**Why the sweep stops at T=4.** At OLMoE, T=8 was indistinguishable from doing nothing:

| run | KL (no SI) | pedagogy NLL |
|---|---|---|
| b-T4 | 0.0633 | 0.8686 |
| b-T8 | 0.0655 | 0.8648 |
| vanilla SFT | 0.0673 | **0.8646** |

T=8 landed within 0.3% of the control on the metric the method exists to move, so a T=8 arm
would spend a run re-measuring `sft-control`. The temperature is applied to a **robust z-scored**
signal, so it is already normalised to that signal's own spread and does not need rescaling when
the model gets bigger — "larger model, therefore raise T" does not follow the way it would if T
were in raw nats.

## Configuration

LoRA, attention plus experts, routers untouched.

| | value | why |
|---|---|---|
| `lora_r` / `lora_alpha` | 8 / 16 | OpenAI's gpt-oss cookbook recipe; carried over to Qwen unchanged |
| `lora_dropout` | 0 | required: PEFT's `ParamWrapper` adapts bare Parameters and has nowhere to put a dropout mask |
| `lora_target_modules` | `q_proj k_proj v_proj o_proj` | |
| `lora_target_parameters` | `mlp.experts.gate_up_proj`, `mlp.experts.down_proj` | the experts are one fused 3-D tensor, invisible to `target_modules` |
| `lora_expert_rank` | 2 | there is one LoRA pair per expert, so rank r across 32 (gpt-oss) or 128 (Qwen) experts is that many times the budget of rank r on one dense layer |
| `lora_expert_alpha` | 4 | keeps the experts' alpha/r scaling equal to attention's instead of silently larger |
| `learning_rate` | 1e-4 | what OLMoE used; between the cookbook's 2e-4 and the 5e-5 that expert-targeting guides suggest |

Data is `meric533/socrateach-sft` at revision `v2`. Checkpoints are log-spaced, since the model
moves fastest early and a uniform grid spends most of its saves on a plateau.

## Compute

Built for `gpu-8xh100`: 80GB per card, and gpt-oss-20b dequantises from MXFP4 to about **42GB**,
so the model fits on each card and this is plain data-parallel replication rather than ZeRO-3.

The 40GB `gpu-8xa100` cards cannot hold 42GB at all, so running there would force sharding
underneath LoRA, fused-MoE `target_parameters` and a `disable_adapter` reference pass — a
combination this repository does not exercise anywhere. It is also above the platform's hourly
rate ceiling, so every submission naming it classifies as an admin EXCEPTION.

## Layout

| file | |
|---|---|
| `weighting.py` | the signal, the cache, and the mean-1 multipliers |
| `modeling.py` | gpt-oss/OLMoE loading and LoRA targeting, with the router-freeze check |
| `test_weighting.py` | CPU tests: no GPU, no network, no model download |

Run the tests with `uv run pytest projects/kl_reweighted_sft/`.
