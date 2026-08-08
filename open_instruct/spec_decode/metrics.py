"""Reading acceptance length out of a running vLLM engine.

WHAT IS BEING MEASURED. Mean acceptance length α -- "the average number of tokens produced per
speculation step" (arXiv:2604.26779 §2.2) -- is the quantity the paper's speedup bound is
written in::

    S_step <= 1 / (R_gen / α + (1 - R_gen))

R_gen comes free from open-instruct's existing per-step timers (``time/group_generation_max``
over ``time/total``). α does not: it lives inside the engine, and this module is how it gets
out. vLLM computes it the same way at ``v1/spec_decode/metrics.py`` -- ``1 + accepted/drafts``,
including the bonus token -- and :func:`SpecDecodeStatLogger.drain` reproduces that definition
rather than inventing one, so our numbers and vLLM's own log line agree.

WHY A CUSTOM STAT LOGGER RATHER THAN THE METRICS ENDPOINT. open-instruct sets
``engine_args.disable_log_stats = True``, which would leave vLLM's default loggers off and the
Prometheus counters unpopulated. Passing ``stat_loggers=[...]`` to ``from_engine_args`` turns
stats back on for this one consumer -- ``AsyncLLM.__init__`` does
``self.log_stats = log_stats or has_custom_loggers`` -- without re-enabling the periodic
throughput logging that flag was set to silence. It also avoids parsing Prometheus text out of
the in-process API server.

WHY IT IS ATTACHED TO BOTH ARMS OF THE EXPERIMENT. Turning stats on costs a little time per
iteration. The measurement is a *ratio of step times* between an autoregressive arm and a
speculative one, so anything that costs time in one arm and not the other lands directly in the
result. This logger therefore records nothing interesting in the baseline arm -- ``num_drafts``
stays 0 and α is reported as ``None`` -- but it is still attached, so both arms pay the same
overhead and it cancels. That is what ``--vllm_collect_spec_decode_stats`` is for, and why it
is a flag of its own rather than something derived from whether a draft was configured.
"""

from __future__ import annotations

import threading
from typing import Any

from vllm.config import VllmConfig
from vllm.v1.metrics.loggers import StatLoggerBase
from vllm.v1.metrics.stats import IterationStats, MultiModalCacheStats, SchedulerStats

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)

#: Instances the engine built, by engine index. The factory vLLM calls is the class itself, so
#: this is the only way to reach the object it constructed. Safe as module state because the
#: ``AsyncLLM`` and its stat loggers live in the same process as the Ray actor that reads them
#: (``VLLM_ENABLE_V1_MULTIPROCESSING=0``); nothing here crosses a process boundary.
_INSTANCES: dict[int, SpecDecodeStatLogger] = {}
_LOCK = threading.Lock()

#: Upper edges of the concurrency histogram, in running requests per engine iteration. Chosen to
#: straddle the regimes the batch-scaling measurement found rather than to be evenly spaced: the
#: interesting boundary is around 64-256, where the cost of extra tokens per forward goes from
#: nearly free to nearly linear. The last bucket is open-ended above 768, the documented rollout
#: shape (48 prompts x 16 samples).
CONCURRENCY_BUCKETS = (8, 16, 32, 64, 128, 256, 512, 768, 1 << 30)


class SpecDecodeStatLogger(StatLoggerBase):
    """Accumulates speculative-decoding counters until drained.

    Totals rather than a rolling window, reset on :meth:`drain`, so that one call per RL step
    yields that step's α. Draining is what makes the series per-step; nothing here knows what
    a step is.
    """

    def __init__(self, vllm_config: VllmConfig, engine_index: int = 0):
        self.engine_index = engine_index
        self.num_spec_tokens = 0
        speculative_config = getattr(vllm_config, "speculative_config", None)
        if speculative_config is not None:
            self.num_spec_tokens = getattr(speculative_config, "num_speculative_tokens", 0) or 0
        self._reset()
        with _LOCK:
            _INSTANCES[engine_index] = self

    def _reset(self) -> None:
        self.num_drafts = 0
        self.num_draft_tokens = 0
        self.num_accepted_tokens = 0
        self.num_accepted_tokens_per_pos = [0] * self.num_spec_tokens
        self.concurrency_buckets = dict.fromkeys(CONCURRENCY_BUCKETS, 0)
        self.num_iterations = 0
        self.running_reqs_total = 0

    def record(
        self,
        scheduler_stats: SchedulerStats | None,
        iteration_stats: IterationStats | None,
        mm_cache_stats: MultiModalCacheStats | None = None,
        engine_idx: int = 0,
    ) -> None:
        if scheduler_stats is None:
            return

        # CONCURRENCY, RECORDED WHETHER OR NOT ANYTHING IS BEING DRAFTED, because it is the
        # measurement that decides whether drafting is worth doing at all. Verifying k+1 tokens
        # for B sequences costs roughly what a decode forward for B*(k+1) sequences costs, so the
        # batch-scaling curve of this model *is* its verification-cost curve. Measured on A100
        # TP=2: quadrupling tokens per forward costs 1.40x at B=16 but 2.89x at B=256, which puts
        # break-even acceptance length near 1.4 in the first regime and near 3.9 in the second --
        # and EAGLE-3 realistically reaches 2.7-3.3 (arXiv:2604.26779 Tables 3-5). So speculation
        # pays exactly to the extent that a real rollout runs at low concurrency, which a
        # fixed-length benchmark cannot show and a real RL step can: answers finish at staggered
        # times, so the tail of every step drains toward small batches.
        running = scheduler_stats.num_running_reqs
        self.num_iterations += 1
        self.running_reqs_total += running
        for bucket in CONCURRENCY_BUCKETS:
            if running <= bucket:
                self.concurrency_buckets[bucket] += 1
                break
        else:
            self.concurrency_buckets[CONCURRENCY_BUCKETS[-1]] += 1

        stats = scheduler_stats.spec_decoding_stats
        if stats is None:
            # Either speculative decoding is off, or this iteration drafted nothing. Both are
            # ordinary: the baseline arm takes this branch on every single iteration.
            return
        self.num_drafts += stats.num_drafts
        self.num_draft_tokens += stats.num_draft_tokens
        self.num_accepted_tokens += stats.num_accepted_tokens
        per_pos = stats.num_accepted_tokens_per_pos or []
        if len(per_pos) > len(self.num_accepted_tokens_per_pos):
            self.num_accepted_tokens_per_pos.extend([0] * (len(per_pos) - len(self.num_accepted_tokens_per_pos)))
        for i, count in enumerate(per_pos):
            self.num_accepted_tokens_per_pos[i] += count

    def log_engine_initialized(self) -> None:
        logger.info(
            "spec_decode: stat logger attached to engine %d (num_speculative_tokens=%d)",
            self.engine_index,
            self.num_spec_tokens,
        )

    def drain(self) -> dict[str, Any]:
        """Return the counters accumulated since the last drain, and reset.

        ``acceptance_length`` is ``None`` when nothing was drafted, rather than 0 or 1. The
        distinction matters: 1.0 would mean "drafted and every proposal was rejected", which is
        a real and bad outcome, while nothing-drafted is what the baseline arm always reports.
        Collapsing the two would make a broken drafter look like the baseline.
        """
        num_drafts = self.num_drafts
        num_draft_tokens = self.num_draft_tokens
        num_accepted = self.num_accepted_tokens
        per_pos = list(self.num_accepted_tokens_per_pos)
        buckets = dict(self.concurrency_buckets)
        num_iterations = self.num_iterations
        running_total = self.running_reqs_total
        self._reset()

        out: dict[str, Any] = {
            "num_drafts": num_drafts,
            "num_draft_tokens": num_draft_tokens,
            "num_accepted_tokens": num_accepted,
            "acceptance_length": None,
            "draft_acceptance_rate": None,
            "num_iterations": num_iterations,
            "running_reqs_total": running_total,
            "concurrency_buckets": buckets,
        }
        if num_drafts > 0:
            # vLLM's own definition, bonus token included, so this matches its log line.
            out["acceptance_length"] = 1 + num_accepted / num_drafts
            # Per-position acceptance: how often the i-th proposed token survived. The shape of
            # this is what says whether a longer draft would pay -- a tail that collapses is the
            # paper's Table 4 effect (acceptance still rising at k=7 while speedup falls).
            out["per_position_acceptance"] = [count / num_drafts for count in per_pos]
        if num_draft_tokens > 0:
            out["draft_acceptance_rate"] = num_accepted / num_draft_tokens
        return out


def drain_all() -> dict[str, Any]:
    """Drain every engine's logger in this process and combine them.

    Counters are summed before α is computed, rather than averaging each engine's α, because a
    mean of ratios is not the ratio of the sums unless every engine drafted equally often.
    """
    with _LOCK:
        instances = list(_INSTANCES.values())
    if not instances:
        return {}

    drained = [instance.drain() for instance in instances]
    num_drafts = sum(d["num_drafts"] for d in drained)
    num_draft_tokens = sum(d["num_draft_tokens"] for d in drained)
    num_accepted = sum(d["num_accepted_tokens"] for d in drained)
    num_iterations = sum(d["num_iterations"] for d in drained)
    running_total = sum(d["running_reqs_total"] for d in drained)

    buckets: dict[int, int] = dict.fromkeys(CONCURRENCY_BUCKETS, 0)
    for d in drained:
        for edge, count in d["concurrency_buckets"].items():
            buckets[edge] = buckets.get(edge, 0) + count

    combined: dict[str, Any] = {
        "num_drafts": num_drafts,
        "num_draft_tokens": num_draft_tokens,
        "num_accepted_tokens": num_accepted,
        "acceptance_length": (1 + num_accepted / num_drafts) if num_drafts > 0 else None,
        "draft_acceptance_rate": (num_accepted / num_draft_tokens) if num_draft_tokens > 0 else None,
        "num_iterations": num_iterations,
        # The raw total travels alongside the mean, because callers aggregate across Ray actors and
        # a mean of per-actor means is not the mean unless every actor ran the same number of
        # iterations. Mean is over *engine iterations*, not requests: an iteration is one forward
        # pass, so this is the average number of sequences a forward carried, which is what the
        # verification-cost curve is indexed by.
        "running_reqs_total": running_total,
        "mean_running_reqs": (running_total / num_iterations) if num_iterations > 0 else None,
        "concurrency_buckets": buckets,
    }
    return combined


def reset_registry_for_testing() -> None:
    """Forget the constructed loggers. Only for tests."""
    with _LOCK:
        _INSTANCES.clear()
