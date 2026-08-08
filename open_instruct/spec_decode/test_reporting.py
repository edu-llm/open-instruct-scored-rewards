"""Tests for the per-step speculative-decoding metrics.

Imports no vLLM. The stat-logger tests, which do, live in ``test_metrics_vllm.py``.
"""

from __future__ import annotations

import math

import pytest

from open_instruct.spec_decode import reporting


class FakeEngine:
    """Stands in for an ``LLMRayActor`` handle, including the ``.remote()`` indirection."""

    def __init__(self, payload=None, raises=False):
        self._payload = payload if payload is not None else {}
        self._raises = raises
        self.drain_spec_decode_metrics = _FakeRemote(self)


class _FakeRemote:
    def __init__(self, engine):
        self._engine = engine

    def remote(self):
        if self._engine._raises:
            raise RuntimeError("engine is gone")
        return self._engine._payload


@pytest.fixture(autouse=True)
def fake_ray(monkeypatch):
    """``ray.get`` on our fake handles is the identity; a raiser propagates."""
    monkeypatch.setattr(reporting.ray, "get", lambda refs: list(refs))


def with_concurrency(payload, buckets, num_iterations=None, running_total=None):
    """Attach a concurrency histogram, as an engine's drain() would."""
    payload = dict(payload)
    payload["concurrency_buckets"] = buckets
    payload["num_iterations"] = num_iterations if num_iterations is not None else sum(buckets.values())
    payload["running_reqs_total"] = running_total if running_total is not None else 0
    return payload


class TestConcurrencyProfile:
    """The measurement the go/no-go actually rests on.

    Verifying k+1 tokens for B sequences costs roughly a decode forward at B*(k+1). So whether
    speculation can pay is decided by how much of a real rollout runs at low concurrency, which is
    what these keys report. They must be emitted by the *baseline* arm too -- that arm drafts
    nothing, and is precisely the run that decides whether a draft is worth training.
    """

    def test_emitted_by_the_baseline_arm_which_drafts_nothing(self):
        payload = with_concurrency(counters(0, 0, 0), {8: 30, 64: 20, 768: 50}, running_total=10000)
        metrics = reporting.step_metrics([FakeEngine(payload)], True, 7.0, 10.0)
        assert "spec/acceptance_length" not in metrics
        assert metrics["spec/engine_iterations"] == 100
        assert metrics["spec/favourable_iteration_fraction"] == pytest.approx(0.5)

    def test_favourable_fraction_counts_iterations_at_or_below_the_threshold(self):
        # 8 and 64 are at or below SPECULATION_FAVOURABLE_CONCURRENCY=64; 128 and above are not.
        buckets = {8: 10, 64: 10, 128: 40, 768: 40}
        metrics = reporting.step_metrics([FakeEngine(with_concurrency(counters(0, 0, 0), buckets))], True, 7.0, 10.0)
        assert metrics["spec/favourable_iteration_fraction"] == pytest.approx(0.2)

    def test_a_saturated_rollout_reports_no_favourable_iterations(self):
        # The pessimistic case: every forward carried a full batch, so extra tokens per forward
        # cost nearly linearly and no achievable acceptance length wins.
        buckets = {768: 100}
        metrics = reporting.step_metrics([FakeEngine(with_concurrency(counters(0, 0, 0), buckets))], True, 7.0, 10.0)
        assert metrics["spec/favourable_iteration_fraction"] == pytest.approx(0.0)

    def test_mean_running_reqs_is_summed_across_engines_not_averaged(self):
        # Engine A: 10 iterations carrying 10 reqs each. Engine B: 90 iterations carrying 810 each.
        # Correct mean over forwards is (100 + 72900)/100 = 730. Averaging per-engine means would
        # give (10 + 810)/2 = 410, which is the mean of two means and not the mean.
        a = with_concurrency(counters(0, 0, 0), {16: 10}, num_iterations=10, running_total=100)
        b = with_concurrency(counters(0, 0, 0), {768: 90}, num_iterations=90, running_total=72900)
        metrics = reporting.step_metrics([FakeEngine(a), FakeEngine(b)], True, 7.0, 10.0)
        assert metrics["spec/mean_running_reqs"] == pytest.approx(730.0)

    def test_buckets_are_reported_as_fractions_that_sum_to_one(self):
        buckets = {8: 25, 64: 25, 256: 25, 768: 25}
        metrics = reporting.step_metrics([FakeEngine(with_concurrency(counters(0, 0, 0), buckets))], True, 7.0, 10.0)
        fractions = [v for k, v in metrics.items() if k.startswith("spec/concurrency_le_")]
        assert sum(fractions) == pytest.approx(1.0)

    def test_absent_when_no_iterations_were_recorded(self):
        metrics = reporting.step_metrics([FakeEngine(counters(0, 0, 0))], True, 7.0, 10.0)
        assert "spec/favourable_iteration_fraction" not in metrics
        assert "spec/engine_iterations" not in metrics


def counters(num_drafts, num_draft_tokens, num_accepted_tokens):
    return {"num_drafts": num_drafts, "num_draft_tokens": num_draft_tokens, "num_accepted_tokens": num_accepted_tokens}


def solve_generation_share(rollout_speedup: float, end_to_end: float) -> float:
    """Invert the bound for R_gen: 1/(R/S + 1 - R) = E  =>  R = (1/E - 1) / (1/S - 1)."""
    return (1 / end_to_end - 1) / (1 / rollout_speedup - 1)


class TestSpeedupBound:
    def test_is_consistent_with_the_papers_two_reported_numbers(self):
        # Paper §4.2, for one operating point: "k=3 achieves a 2.72x rollout speedup and a 1.70x
        # end-to-end speedup". Those two numbers together pin R_gen, and in the bound the
        # rollout-stage speedup plays exactly the role α does. So inverting our formula must land
        # inside the 0.65-0.72 generation share Table 1 independently measures. It does, at
        # ~0.651 -- which is a real check on the formula rather than a restatement of it.
        r_gen = solve_generation_share(rollout_speedup=2.72, end_to_end=1.70)
        assert 0.65 <= r_gen <= 0.72, r_gen
        assert reporting.step_speedup_bound(r_gen, 2.72) == pytest.approx(1.70)

    def test_alpha_of_one_is_no_speedup(self):
        # Acceptance length 1 means only the bonus token survives every time: exactly as many
        # forward passes as autoregressive decoding.
        assert reporting.step_speedup_bound(0.7, 1.0) == pytest.approx(1.0)

    def test_bound_is_capped_by_generation_share(self):
        # Amdahl: with infinite acceptance the non-generation stages still remain.
        assert reporting.step_speedup_bound(0.5, 1e9) == pytest.approx(2.0, rel=1e-6)
        assert reporting.step_speedup_bound(0.0, 1e9) == pytest.approx(1.0)

    def test_higher_generation_share_raises_the_ceiling(self):
        low = reporting.step_speedup_bound(0.3, 3.0)
        high = reporting.step_speedup_bound(0.7, 3.0)
        assert high > low

    @pytest.mark.parametrize("alpha", [0.0, -1.0])
    def test_nonpositive_alpha_has_no_bound(self, alpha):
        assert reporting.step_speedup_bound(0.7, alpha) is None


class TestStepMetrics:
    def test_returns_nothing_when_not_collecting(self):
        # Keeps an unflagged run logging exactly what upstream logs.
        engines = [FakeEngine(counters(10, 30, 20))]
        assert reporting.step_metrics(engines, False, 5.0, 10.0) == {}

    def test_returns_nothing_without_engines(self):
        assert reporting.step_metrics([], True, 5.0, 10.0) == {}
        assert reporting.step_metrics(None, True, 5.0, 10.0) == {}

    def test_generation_share(self):
        engines = [FakeEngine(counters(0, 0, 0))]
        metrics = reporting.step_metrics(engines, True, 7.0, 10.0)
        assert metrics["spec/generation_share"] == pytest.approx(0.7)

    def test_baseline_arm_reports_share_but_no_acceptance_length(self):
        # The autoregressive arm drafts nothing on every iteration. It must still report R_gen,
        # because that is the measurement the baseline arm exists to produce -- and it must not
        # report an acceptance length, because there is none.
        metrics = reporting.step_metrics([FakeEngine(counters(0, 0, 0))], True, 7.0, 10.0)
        assert "spec/generation_share" in metrics
        assert "spec/acceptance_length" not in metrics
        assert "spec/step_speedup_bound" not in metrics
        assert metrics["spec/num_drafts"] == 0

    def test_acceptance_length_includes_the_bonus_token(self):
        # vLLM's convention: 1 + accepted/drafts. 10 drafts, 20 accepted -> 3.0.
        metrics = reporting.step_metrics([FakeEngine(counters(10, 30, 20))], True, 7.0, 10.0)
        assert metrics["spec/acceptance_length"] == pytest.approx(3.0)
        assert metrics["spec/draft_acceptance_rate"] == pytest.approx(20 / 30)

    def test_bound_is_emitted_from_measured_values(self):
        metrics = reporting.step_metrics([FakeEngine(counters(10, 30, 20))], True, 7.0, 10.0)
        assert metrics["spec/step_speedup_bound"] == pytest.approx(reporting.step_speedup_bound(0.7, 3.0))

    def test_counters_are_summed_before_alpha_not_averaged_after(self):
        # Two engines with very different draft counts. Summing first gives
        # 1 + (2 + 30)/(2 + 10) = 3.666...; averaging each engine's alpha would give
        # ((1 + 1) + (1 + 3)) / 2 = 3.0. The first is the alpha of the actual rollout.
        engines = [FakeEngine(counters(2, 6, 2)), FakeEngine(counters(10, 30, 30))]
        metrics = reporting.step_metrics(engines, True, 7.0, 10.0)
        assert metrics["spec/acceptance_length"] == pytest.approx(1 + 32 / 12)
        assert metrics["spec/acceptance_length"] != pytest.approx(3.0)

    def test_zero_step_time_omits_share_without_dividing_by_zero(self):
        metrics = reporting.step_metrics([FakeEngine(counters(10, 30, 20))], True, 0.0, 0.0)
        assert "spec/generation_share" not in metrics
        assert metrics["spec/acceptance_length"] == pytest.approx(3.0)

    def test_a_dead_engine_loses_telemetry_and_not_the_run(self):
        # Instrumentation must not kill a training step that has already done its work.
        metrics = reporting.step_metrics([FakeEngine(raises=True)], True, 7.0, 10.0)
        assert metrics["spec/generation_share"] == pytest.approx(0.7)
        assert "spec/acceptance_length" not in metrics

    def test_empty_payload_from_an_engine_is_tolerated(self):
        engines = [FakeEngine({}), FakeEngine(counters(10, 30, 20))]
        metrics = reporting.step_metrics(engines, True, 7.0, 10.0)
        assert metrics["spec/acceptance_length"] == pytest.approx(3.0)

    def test_all_metrics_are_finite_scalars(self):
        # grpo_fast filters its console line to float|int and hands the rest to wandb; a None or
        # a NaN sneaking in here shows up as a broken panel rather than an error.
        metrics = reporting.step_metrics([FakeEngine(counters(10, 30, 20))], True, 7.0, 10.0)
        for key, value in metrics.items():
            assert isinstance(value, float | int), key
            assert math.isfinite(value), key
