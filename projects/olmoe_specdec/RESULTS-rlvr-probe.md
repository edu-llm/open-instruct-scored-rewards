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

## Attempt 3 — `run_019fe36d-9120-7091-a406-6ff233cf291e`

`FAILED`, exit **15** (SIGTERM), `Job attempt duration exceeded timeout`. Ran the full two-hour
cap, 22:25:44 → 00:25:56, and produced **no measurement at all**. ~$44 of A100 time for nothing.

Both fixes from attempt 2 were in the image (`cd42286`, build verified before submitting). It made
no difference to how far the run got:

| attempt | died at | what happened |
|---|---|---|
| 2 | 3.6 min | `KeyError: ...experts.gate_up_proj` + OOM, reported and fatal |
| 3 | **3.7 min** | engine died, error **masked**, then 116 minutes of retries |

The engine failed at essentially the same point. What changed is only that the failure became
invisible:

```
22:29:27  <engine error on /completions>
          vllm/entrypoints/openai/server_utils.py:357, in engine_error_handler
              server=req.app.state.server,
          AttributeError: 'State' object has no attribute 'server'
22:29:27  Retrying request to /completions in 0.42s
22:31:11  ActorManager - Stopping queue polling thread...
23:29:27  Retrying request to /completions in 0.39s      <- an hour later
00:25:56  killed by the runtime cap
```

**So the weight-sync question is still open.** The engine died at about the moment the first sync
happens, and the error that would say whether it was the sync is the one that got swallowed. The MoE
unfuse may be correct and untested, or wrong in a new way. Nothing here distinguishes those.

### Two integration bugs, and the second is what cost the money

**1. Engine errors are unreportable.** open-instruct builds the vLLM app with `build_app(args)` then
`init_app_state(engine_client, app.state, args)`, but vLLM 0.21's `engine_error_handler` reads
`req.app.state.server`, which `init_app_state` does not set. So the handler raises while handling the
error, and the original exception is lost. Every engine failure in this integration is invisible.

**2. A dead engine does not fail the run.** The OpenAI client retried `/completions` for 116 minutes
against an engine that was gone. Batch reported `RUNNING` throughout, because the process was alive
and looping. This is the expensive bug: it converts a 4-minute diagnosis into a full-cap bill, and
it will do so on every future failure until fixed.

Neither is specific to speculative decoding, and both will hit any `grpo_fast.py` run on this fork
whose engine dies for any reason.

### What this cost, and the misread that made it worse

Four probe attempts, no measurement: 30 s, 217 s, 0 s, and 2 h. The last one is the only expensive
one, and it was avoidable — at 44 minutes I read "still `RUNNING`" as healthy progress and said so.
**A retry loop and a working run are indistinguishable in the Batch job record**; the only thing that
separates them is whether new step metrics are appearing in the log, which I had not checked.
The lesson is cheap to state: for a job whose progress is measurable, check the progress, not the
status.

A tighter `maximum_runtime_hours` would have limited the damage, but it is a mitigation rather than a
fix — the run should abort when its engine dies, not survive to be killed by a clock.

## Attempt 2 — `run_019fe337-7245-7035-bbf6-0984be48c46b`

`FAILED`, exit 1, after **217 s**. Still no measurement. It got much further: Ray actors up, both
learners and the vLLM engine built (`speculative_config=None`, TP=2, v0.21.0), and it died in the
**first weight sync** — `PolicyTrainerRayProcess.broadcast_to_vllm()`.

Two independent faults in that one operation.

### Fault 1 — fused vs per-expert MoE weight names. This blocks the whole project.

```
gpu_worker.py:1051  update_weights -> load_weights([(name, weight)])
olmoe.py:497        OlmoeForCausalLM.load_weights -> AutoWeightsLoader
olmoe.py:429        param = params_dict[name]
KeyError: 'layers.0.mlp.experts.gate_up_proj'
```

The learner is broadcasting a parameter named `layers.0.mlp.experts.gate_up_proj` — a **fused**
expert tensor with no expert index. vLLM 0.21's OLMoE expects the **unfused** checkpoint layout: its
`get_expert_mapping` calls `fused_moe_make_expert_params_mapping` with
`ckpt_gate_proj_name="gate_proj"`, `ckpt_up_proj_name="up_proj"`, which matches
`experts.<i>.gate_proj.weight` per expert. The fused name matches nothing, falls through to a direct
`params_dict[name]` lookup, and raises.

The likely origin is the MoE refactor in transformers 5.x, which packs experts into single
`gate_up_proj` / `down_proj` tensors; this image pins `transformers>=5.4.0` against `vllm==0.21.0`.
Worth confirming by printing the learner's parameter names, but the mechanism is not in doubt: the
two halves of the weight-sync path disagree about how OLMoE's experts are named.

**Consequences, which reach past this experiment:**

- **The eventual RLVR run cannot work on this image as it stands.** Weight sync is not optional in
  GRPO — the policy has to reach the rollout engine every step. This is the critical path, not a
  detour, and it must be fixed whether or not speculative decoding is ever used.
- **It is not caused by anything in `spec_decode/`.** This run passed `speculative_config=None`, so
  the registration never fired and the model was upstream's `OlmoeForCausalLM`.
- **It is MoE-specific.** A dense policy has no expert tensors, so `pedagogy_rm`'s
  OLMo-2-7B-Instruct GRPO runs will not hit it. Anyone pointing this repo at an MoE will.

Three candidate fixes, in increasing order of blast radius: remap the fused names to per-expert
names in the broadcast path (contained, ours); pin transformers below the MoE refactor (touches the
shared image and whatever else needs 5.4); or move to a vLLM that understands fused expert names
(also the shared image). The first is the only one that does not change a dependency other people
build on, so it is the one to try first — but it needs the actual parameter names confirmed.

### Fault 2 — CUDA OOM in the same broadcast, which is the risk that was flagged

```
ray::PolicyTrainerRayProcess.broadcast_to_vllm()
torch.OutOfMemoryError: Tried to allocate 256.00 MiB. GPU 0 has a total capacity of 39.49 GiB
of which 31.50 MiB is free. ... 36.62 GiB is allocated by PyTorch
```

39.45 of 39.49 GiB in use. This is the memory sizing that was reasoned rather than measured: under
ZeRO-3 the broadcast has to gather sharded parameters back to full size, so peak memory during sync
exceeds steady-state training memory by roughly a full copy of the model. 6 learners on 40 GB cards
does not leave room for that.

Options, all of which move R_gen and so must be re-measured rather than assumed: fewer learners with
more memory each; `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (the error's own suggestion —
cheap to try, addresses fragmentation rather than the underlying peak); chunked broadcast if
open-instruct supports it; or ZeRO-3 parameter offload during sync.

## Where this leaves the question

Still **no `f`, no R_gen, and therefore no verdict** on speculative decoding. What has been
established, at a cost of a few dollars:

- The integration itself is sound. Three vLLM test files pass, the EAGLE-3 target registers, the
  stat logger attaches, and the engine builds — verified in the baseline sweep.
- The verification-cost curve is measured, and says speculation wins 1.4–2.1× on generation below
  concurrency 64 and cannot win once saturated (`RESULTS-baseline-sweep.md`).
- The one missing input is the real rollout's concurrency profile, which needs a working RL step.
- **And a working RL step is now blocked on a weight-sync incompatibility that has nothing to do
  with speculative decoding.** That is the thing to fix next, because the RLVR run needs it
  regardless of what this experiment concludes.

## Next attempt: what has to change first

Not another submission of the same thing. Two code fixes, both cheap, both needed before another
approval is worth spending:

1. **Make the engine's error visible.** Set `app.state.server` where open-instruct calls
   `init_app_state`, or catch the engine exception before vLLM's handler can mask it. Until this
   lands, a failed run tells us nothing, which is how attempt 3 cost a full cap.
2. **Abort on a dead engine.** A `/completions` failure against a dead engine should end the run,
   not retry. Cap the retries or check engine liveness in the actor's background-thread check
   (`LLMRayActor.check_background_threads` already raises when the loop thread dies; the engine
   dying needs the same treatment).

With those, a failure like attempt 3's costs four minutes and prints its cause. That is the
difference between this being answerable and not.

Worth doing at the same time, since it is free: drop `maximum_runtime_hours` for probes so that a
hang is bounded by minutes rather than hours. A mitigation, not a substitute for fix 2.
