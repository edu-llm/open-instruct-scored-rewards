"""Per-step speculative-decoding metrics, in the form the speedup analysis needs.

Three quantities, and the third is the one worth having:

``spec/generation_share`` -- R_gen, the fraction of step time spent generating. This is the
Amdahl ceiling on anything done to the rollout engine, and it is the single number that decides
whether this experiment can succeed at all. The paper's workload sits at 0.65-0.72; ours is a
1B-active MoE emitting short GSM answers, so it may sit much lower, and that is worth knowing
from the baseline arm before a draft model is ever trained.

``spec/acceptance_length`` -- α, mean tokens per speculation step, drained from the engine.

``spec/step_speedup_bound`` -- the paper's §2.2 bound evaluated at *our* measured R_gen and α::

    S_step <= 1 / (R_gen / α + (1 - R_gen))

Logging the bound beside the realised step time is what makes a disappointing result
interpretable rather than merely disappointing. If measured speedup tracks the bound, the
integration is working and the workload simply has little generation to accelerate. If measured
speedup falls well below the bound, the loss is overhead -- drafting cost, verification, batch
effects -- and that is a different problem with different fixes. Reporting only "we got 1.05x"
cannot distinguish those two, and they lead to opposite decisions.
"""

from __future__ import annotations

from typing import Any

import ray

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)

_COUNTER_KEYS = ("num_drafts", "num_draft_tokens", "num_accepted_tokens", "num_iterations", "running_reqs_total")


def step_speedup_bound(generation_share: float, acceptance_length: float) -> float | None:
    """The paper's §2.2 step-speedup bound. ``None`` when it is not defined.

    Derived under the assumption that a speculation step costs the same as one autoregressive
    forward pass, so it is an upper bound and not a prediction: drafting overhead, prefill and
    batching all push the realised number below it.
    """
    if acceptance_length <= 0:
        return None
    denominator = generation_share / acceptance_length + (1.0 - generation_share)
    if denominator <= 0:
        return None
    return 1.0 / denominator


def step_metrics(
    vllm_engines: list[ray.actor.ActorHandle] | None,
    collect_stats: bool,
    total_generation_time: float,
    step_time: float,
) -> dict[str, Any]:
    """Drain every engine and return this step's speculative-decoding metrics.

    Returns ``{}`` unless ``collect_stats``, so a run that does not ask for the measurement logs
    exactly what upstream logs.
    """
    if not collect_stats or not vllm_engines:
        return {}

    metrics: dict[str, Any] = {}
    if step_time > 0:
        metrics["spec/generation_share"] = total_generation_time / step_time

    totals = _drain(vllm_engines)
    if totals is None:
        return metrics

    metrics["spec/num_drafts"] = totals["num_drafts"]
    metrics["spec/num_draft_tokens"] = totals["num_draft_tokens"]
    metrics["spec/num_accepted_tokens"] = totals["num_accepted_tokens"]
    metrics.update(_concurrency_metrics(totals))

    if totals["num_drafts"] <= 0:
        # The baseline arm's normal state. Left absent rather than zero: an acceptance length of
        # 0 or 1 would mean "drafted and got nothing accepted", which is a real failure and must
        # not be confused with "did not draft".
        return metrics

    # vLLM's own definition, bonus token included, so this agrees with its log line.
    acceptance_length = 1 + totals["num_accepted_tokens"] / totals["num_drafts"]
    metrics["spec/acceptance_length"] = acceptance_length
    if totals["num_draft_tokens"] > 0:
        metrics["spec/draft_acceptance_rate"] = totals["num_accepted_tokens"] / totals["num_draft_tokens"]

    generation_share = metrics.get("spec/generation_share")
    if generation_share is not None:
        bound = step_speedup_bound(generation_share, acceptance_length)
        if bound is not None:
            metrics["spec/step_speedup_bound"] = bound
    return metrics


#: Concurrency at or below which extra tokens per forward are cheap enough that speculation has
#: real headroom. Measured on this model (see projects/olmoe_specdec/RESULTS-baseline-sweep.md):
#: quadrupling tokens per forward costs 1.40x at concurrency 16 and 1.77-1.90x at concurrency 64,
#: so break-even acceptance length is ~1.4 and ~1.8 respectively. Above ~256 the engine saturates,
#: extra tokens per forward cost linearly, and no achievable acceptance length wins. EAGLE-3
#: reaches 2.7-3.3 in-domain, which clears 1.8 with margin -- hence 64 as the boundary between
#: "wins" and "cannot win". Not a universal constant: re-measure per model and per shape.
SPECULATION_FAVOURABLE_CONCURRENCY = 64


def _concurrency_metrics(totals: dict[str, Any]) -> dict[str, Any]:
    """How much of this step's generation ran at a concurrency where speculation could pay.

    This is the measurement the whole go/no-go rests on, and it is why a real RL step is worth
    running rather than a fixed-length benchmark. A benchmark holds every sequence alive for the
    full length, so it reports one concurrency. A real rollout does not: RLVR-GSM answers are short
    and finish at staggered times, so a step starts near its nominal batch and drains, and the
    draining tail is exactly where verifying k+1 tokens is nearly as cheap as verifying one.

    ``favourable_iteration_fraction`` is the fraction of forward passes that ran at or below
    :data:`SPECULATION_FAVOURABLE_CONCURRENCY`. It counts iterations rather than tokens on purpose:
    speculation removes *forward passes*, so forwards are the unit its benefit is denominated in.
    """
    num_iterations = totals.get("num_iterations") or 0
    out: dict[str, Any] = {}
    if num_iterations <= 0:
        return out

    out["spec/engine_iterations"] = num_iterations
    mean_running = totals.get("mean_running_reqs")
    if mean_running is not None:
        out["spec/mean_running_reqs"] = mean_running

    buckets = totals.get("concurrency_buckets") or {}
    favourable = sum(count for edge, count in buckets.items() if edge <= SPECULATION_FAVOURABLE_CONCURRENCY)
    out["spec/favourable_iteration_fraction"] = favourable / num_iterations
    for edge, count in sorted(buckets.items()):
        label = "inf" if edge > 1 << 20 else str(edge)
        out[f"spec/concurrency_le_{label}"] = count / num_iterations
    return out


def _drain(vllm_engines: list[ray.actor.ActorHandle]) -> dict[str, int] | None:
    """Sum the counters across engines, or None if they could not be read.

    Counters are summed before α is computed, never averaged as ratios: a mean of per-engine α
    is not the α of the whole rollout unless every engine drafted the same number of times.

    A failure here is logged and swallowed. This is instrumentation on the side of a training
    step that has already done its work -- losing one step of telemetry is the right outcome,
    and killing the run over it is not.
    """
    try:
        per_engine = ray.get([engine.drain_spec_decode_metrics.remote() for engine in vllm_engines])
    except Exception:
        logger.exception("spec_decode: could not drain acceptance metrics; skipping this step")
        return None

    totals: dict[str, Any] = dict.fromkeys(_COUNTER_KEYS, 0)
    buckets: dict[int, int] = {}
    for engine_metrics in per_engine:
        if not engine_metrics:
            continue
        for key in _COUNTER_KEYS:
            totals[key] += engine_metrics.get(key, 0) or 0
        for edge, count in (engine_metrics.get("concurrency_buckets") or {}).items():
            buckets[edge] = buckets.get(edge, 0) + count
    totals["concurrency_buckets"] = buckets
    # Summed from raw totals rather than averaged from each actor's mean: a mean of means is only
    # the mean when every actor ran the same number of iterations, which they do not.
    if totals["num_iterations"] > 0:
        totals["mean_running_reqs"] = totals["running_reqs_total"] / totals["num_iterations"]
    return totals
