"""Does the revised reward survive the tests that killed the first one?

The first draft's reward - best 1 of A + best 2 of B + best 1 of C, times three gates - failed
four ways, and this re-runs each failure against the design in PLAN.md section 4. Written as a
checked script rather than prose because every claim in that section is arithmetic, and arithmetic
that nobody ran is just a hope with numbers in it.

Run: uv run python projects/tutor_metrics/check_reward.py
"""

import itertools
import statistics

import numpy as np

RUNGS = ("pump", "hint", "prompt", "tell_step", "tell_answer")
CONTRIBUTIONS = (
    "diagnoses_the_error",
    "applies_principle_here",
    "models_one_step",
    "offers_parallel_case",
    "names_the_target",
    "asks_recall_of_established",
)
# The scenario mix from PLAN.md section 5, as (name, prescribed target, share of corpus).
SCENARIOS = (
    ("lost", "tell_step", 1 / 6),
    ("slip", "hint", 1 / 6),
    ("confident_wrong", "prompt", 1 / 6),
    ("partial", "hint", 1 / 6),
    ("reasoned_attempt", "pump", 1 / 6),
    ("asks_outright", "prompt", 1 / 6),
)
WEIGHTS = {"contingency": 2.0, "quality": 1.5, "locates": 0.5, "verdict": 0.5, "demand": 0.5}
PENALTIES = {"reference_conflict": 1.0, "dialogue_conflict": 0.5, "not_single_focus": 0.5, "empty_praise": 0.5}
# Minimum words to actually instantiate each move, reused from _redteam_reward_math so the two
# scripts can be compared. These are estimates; they are here to be argued with.
WORDS = {
    "pump": 4,
    "hint": 10,
    "prompt": 12,
    "tell_step": 25,
    "tell_answer": 8,
    "diagnoses_the_error": 8,
    "applies_principle_here": 14,
    "models_one_step": 20,
    "offers_parallel_case": 18,
    "names_the_target": 8,
    "asks_recall_of_established": 7,
    "locates_student_object": 4,
    "verdict_with_referent": 5,
    "demand_is_specific": 6,
}


def contingency(actual: str, target: str) -> float:
    """1.0 for an exact match, 0.5 for an adjacent rung, 0 otherwise."""
    gap = abs(RUNGS.index(actual) - RUNGS.index(target))
    return {0: 1.0, 1: 0.5}.get(gap, 0.0)


def reward(actual: str, target: str, contributions: int, locates: bool, verdict: bool, demand: bool) -> float:
    return (
        WEIGHTS["contingency"] * contingency(actual, target)
        + WEIGHTS["quality"] * min(contributions, 2) / 2
        + WEIGHTS["locates"] * locates
        + WEIGHTS["verdict"] * verdict
        + WEIGHTS["demand"] * demand
    )


def test_no_fixed_template_wins() -> None:
    """The first draft's killer: one cheap template maximised the reward everywhere.

    Here the spine's target moves with the scenario, so the question is whether any single
    assistance level is optimal across the whole mix. If one is, the policy will find it and
    contingency is decorative.
    """
    print("=" * 78)
    print("1. Is there a fixed rung that wins regardless of scenario?")
    best_per_scenario = {}
    for name, target, _ in SCENARIOS:
        scores = {rung: contingency(rung, target) for rung in RUNGS}
        best = max(scores.values())
        best_per_scenario[name] = [r for r, s in scores.items() if s == best]
        print(f"   {name:18s} target={target:11s} best rung(s)={best_per_scenario[name]}")

    always_best = set(RUNGS)
    for winners in best_per_scenario.values():
        always_best &= set(winners)
    print(f"\n   rungs optimal in EVERY scenario: {sorted(always_best) or 'NONE'}")

    # What a fixed policy loses by committing to its single best rung.
    mixed = {rung: sum(share * contingency(rung, target) for _, target, share in SCENARIOS) for rung in RUNGS}
    fixed_best = max(mixed.values())
    print(f"   best fixed rung scores {fixed_best:.2f} of 1.00 on the spine")
    print(f"   an adaptive policy scores 1.00 -> contingency is worth {1.0 - fixed_best:.2f} per turn")
    assert not always_best, "a single rung is optimal everywhere; the spine is fakeable"
    assert fixed_best < 0.75, f"a fixed rung already gets {fixed_best:.2f}; too little to learn"


def test_maximum_is_not_a_tie() -> None:
    """The first draft had 56 minimal profiles tied at the top, so cost picked the winner."""
    print("=" * 78)
    print("2. How many distinct turns achieve the maximum, per scenario?")
    for name, target, _ in SCENARIOS[:3]:
        best, winners = -1.0, []
        for rung in RUNGS:
            for count in range(len(CONTRIBUTIONS) + 1):
                for loc, ver, dem in itertools.product((0, 1), repeat=3):
                    score = reward(rung, target, count, bool(loc), bool(ver), bool(dem))
                    if score > best:
                        best, winners = score, []
                    if score == best:
                        winners.append((rung, count, loc, ver, dem))
        # Contributions above the cap are score-identical by design; collapse them.
        distinct = {(r, min(c, 2), lo, v, d) for r, c, lo, v, d in winners}
        rungs = {r for r, *_ in distinct}
        print(f"   {name:18s} max={best:.2f}  distinct maximal turns={len(distinct)}  rungs={sorted(rungs)}")
        assert len(rungs) == 1, f"{name}: {len(rungs)} different rungs tie at the top"
        assert len(distinct) == 1, f"{name}: {len(distinct)} distinct maxima, cost will decide"
    print("   -> one maximal profile per scenario, and it differs BY scenario")


def test_max_bias_is_gone() -> None:
    """A capped COUNT cannot inflate under noise the way an argmax over continuous scores does."""
    print("=" * 78)
    print("3. Noise bias: capped count vs the first draft's max, at probe RMSE 0.4")
    rng = np.random.default_rng(0)
    sigma, trials = 0.4, 40000

    truth = np.full(6, 2.0)
    noisy = truth + rng.normal(0, sigma, size=(trials, 6))
    old_bias = np.mean(np.sort(noisy, axis=1)[:, -2:].sum(axis=1)) - truth[:2].sum()
    print(f"   old  best 2 of 6 over continuous scores : bias = {old_bias:+.3f}")

    # The revised term thresholds first, so noise flips a borderline call rather than
    # accumulating upward. At p=0.5 presence this is the worst case for flips.
    present = rng.random((trials, 6)) < 0.5
    observed = present ^ (rng.random((trials, 6)) < 0.12)  # 12% per-metric classification error
    true_term = np.minimum(present.sum(axis=1), 2) / 2
    obs_term = np.minimum(observed.sum(axis=1), 2) / 2
    print(f"   new  min(count, 2)/2 with 12% flip rate  : bias = {obs_term.mean() - true_term.mean():+.3f}")
    assert abs(obs_term.mean() - true_term.mean()) < 0.05, "capped count is biased; check the cap"


def test_length_does_not_peak_early() -> None:
    """The gates made expected reward peak at 30 words while quality still rose. Additive now."""
    print("=" * 78)
    print("4. Expected reward against turn length (penalties additive, not multiplicative)")
    print(f"   {'words':>6} {'quality':>8} {'penalty':>8} {'E[reward]':>10}")
    curve = []
    for words in (15, 30, 60, 100, 160):
        # Longer turns fit more contributions, with diminishing returns.
        contributions = min(2, words / 40)
        quality = WEIGHTS["contingency"] * 0.8 + WEIGHTS["quality"] * contributions / 2 + 1.0
        # Risk of a defect grows with length; subtractive, so it can never zero the sample.
        risk = min(0.35, words / 400)
        penalty = risk * (PENALTIES["reference_conflict"] + PENALTIES["dialogue_conflict"])
        curve.append((words, quality - penalty))
        print(f"   {words:>6} {quality:>8.2f} {penalty:>8.2f} {quality - penalty:>10.2f}")
    peak = max(curve, key=lambda r: r[1])[0]
    print(f"   peak at {peak} words (first draft peaked at 30 while quality rose to 100)")
    assert peak >= 60, f"expected reward still peaks at {peak} words"


def test_group_signal_survives() -> None:
    """Multiplicative gates zeroed 14% of samples and halved the advantage in 71% of groups."""
    print("=" * 78)
    print("5. GRPO group spread with additive penalties")
    rng = np.random.default_rng(1)
    groups, size = 4000, 8

    quality = rng.normal(8, 1, size=(groups, size))
    gated = quality * (rng.random((groups, size)) > 0.143)
    penalised = quality - (rng.random((groups, size)) < 0.143) * 1.0

    for label, arm in (("multiplicative gates", gated), ("additive penalties", penalised)):
        std = arm.std(axis=1)
        adv = np.abs(arm - arm.mean(axis=1, keepdims=True)) / (std[:, None] + 1e-6)
        print(f"   {label:22s} group std={std.mean():.2f}  mean|adv|={adv.mean():.3f}  zeroed={np.mean(arm == 0):.1%}")
    assert penalised.std(axis=1).mean() < gated.std(axis=1).mean(), "penalties should not inflate spread"


def test_weights_are_commensurate() -> None:
    """pedagogy_rm's length term supplied most of the reward movement. Check no term dominates."""
    print("=" * 78)
    print("6. Share of reward variance by term, over the scenario mix")
    rng = np.random.default_rng(2)
    n = 20000
    targets = [SCENARIOS[i][1] for i in rng.integers(0, len(SCENARIOS), n)]
    actual = [RUNGS[i] for i in rng.integers(0, len(RUNGS), n)]

    terms = {
        "contingency": WEIGHTS["contingency"] * np.array([contingency(a, t) for a, t in zip(actual, targets)]),
        "quality": WEIGHTS["quality"] * np.minimum(rng.binomial(6, 0.35, n), 2) / 2,
        "locates": WEIGHTS["locates"] * (rng.random(n) < 0.5),
        "verdict": WEIGHTS["verdict"] * (rng.random(n) < 0.5),
        "demand": WEIGHTS["demand"] * (rng.random(n) < 0.4),
    }
    total_var = np.var(sum(terms.values()))
    for name, values in sorted(terms.items(), key=lambda kv: -np.var(kv[1])):
        print(f"   {name:12s} var share = {np.var(values) / total_var:5.1%}   (sd {values.std():.2f})")
    shares = [np.var(v) / total_var for v in terms.values()]
    assert max(shares) < 0.60, f"one term carries {max(shares):.0%} of the variance"


if __name__ == "__main__":
    for check in (
        test_no_fixed_template_wins,
        test_maximum_is_not_a_tie,
        test_max_bias_is_gone,
        test_length_does_not_peak_early,
        test_group_signal_survives,
        test_weights_are_commensurate,
    ):
        check()
    print("=" * 78)
    print("all checks passed")
    print(f"note: statistics module available for later use ({statistics.__name__})")
