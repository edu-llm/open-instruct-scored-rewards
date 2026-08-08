# Submissions

Working payloads, kept because reconstructing one from a refusal message costs more than a file.

| run id | shape | worst case | what it is |
|---|---|---|---|
| `run_019fe273-b38d-70bc-b1d0-aa34a7691610` | `gpu-8xa100` | $43.92 | Autoregressive baseline, TP=2. Submitted with `edullm submit` from `.edullm/run.yaml` |
| `run_019fe294-1341-70f4-8299-6037b02df62a` | `gpu-1xl40s` | $3.72 | Same job at TP=1, dispatched from `baseline-l40s.json` |

Both against commit `762f21f`, experiment `olmoe-specdec-baseline{,-l40s}`, team `post-training`,
dataset `none`. Each runs the two vLLM-dependent test files and then the generation sweep.

## Why two shapes for one measurement

`gpu-8xa100` is the shape the eventual RL runs need, and on 2026-08-08 the P pool was dry:
`describe-scaling-activities` reported `InsufficientInstanceCapacity` for `p4d.24xlarge` in all
four availability zones the compute environment can reach. The A100 job sat in `RUNNABLE`
billing nothing.

That wait is unbounded rather than merely long, and the reason is worth writing down. Each queue
carries an 1800-second cancel under `CAPACITY:INSUFFICIENT_INSTANCE_CAPACITY`, but Batch matches
those rules on the job's `statusReason` and leaves it **null** for this failure -- the capacity
error exists only in the autoscaling group's activity history, which no job field reflects. So
`edullm status` prints "Why: nothing reported" and nothing times the job out.

`gpu-1xl40s` draws on the G pool, which the platform's own notes call the plentiful one, and it
started within minutes of approval. It is 12x cheaper and holds OLMoE (14 GB in bf16) in 48 GB
with room for KV cache.

## Two things learned submitting these

**`edullm check` said `automatic` and both runs went to `run-approval-lead` anyway.** Check
reaches no network, so it cannot see the org's rolling daily spend ceiling, which had been
crossed. The $3.72 run was gated exactly like the $43.92 one -- being cheap does not buy an
automatic release once the day's ceiling is gone.

**The launcher waiver must be a prefix assignment, not `export`.** The guard parses the command
as a shell would, so in `export EDULLM_LAUNCH_CHECK=waived` the token is an *argument to export*
and does not count; `process_per_device` still refuses. Write
`EDULLM_LAUNCH_CHECK=waived python ...`, once per invocation, since a prefix assignment scopes to
the single command it precedes.

## Not waived: the checkpoint contract

`open-instruct-scored-rewards-train` declares a checkpoint every 30 minutes and a resume on
retry. Rather than waive it, `sweep_generation.py` writes each finished cell to
`$EDULLM_CHECKPOINT_DIR` and skips it on a retry. Completed cells are the only state this job
has, so they are exactly what a resume needs -- the contract is met rather than ticked.

`-train` rather than `-check` for the 2-hour bound: this job pays an image pull, a ~14 GB model
download and MoE engine startup before it measures anything, and `-check` caps at 1 hour. Losing
a run to the wall clock would also throw away the queue wait, which is the expensive part.
