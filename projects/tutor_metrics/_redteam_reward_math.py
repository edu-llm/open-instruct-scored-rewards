"""Numbers behind the red-team review of PLAN.md's capped top-K reward.

Two questions:
  1. How degenerate is argmax(reward)? Enumerate the profiles that tie at the top.
  2. How much does max-over-noisy-probes bias and compress the signal?

Nothing here depends on data that does not exist yet; it is arithmetic on the
reward as written in PLAN.md section 3.
"""

from __future__ import annotations

import itertools

import numpy as np

A = ["asks_self_explanation", "asks_next_step", "asks_prediction", "asks_recall", "asks_self_check"]
B = [
    "locates_student_object",
    "diagnoses_the_error",
    "applies_principle_here",
    "models_one_step",
    "offers_parallel_case",
    "names_the_target",
]
C = ["verdict_with_referent"]

ANTAGONISTIC = [
    ("models_one_step", "asks_self_explanation"),
    ("applies_principle_here", "asks_prediction"),
    ("asks_recall", "applies_principle_here"),
    ("offers_parallel_case", "models_one_step"),
]

# My estimate of the minimum number of words needed to earn a top score on each
# metric, written out in the review so it can be argued with. Shared words are
# handled separately; this is an upper bound on the cheapest turn.
MIN_WORDS = {
    "asks_self_explanation": 7,
    "asks_next_step": 4,
    "asks_prediction": 10,
    "asks_recall": 7,
    "asks_self_check": 7,
    "locates_student_object": 3,
    "diagnoses_the_error": 4,
    "applies_principle_here": 7,
    "models_one_step": 8,
    "offers_parallel_case": 25,
    "names_the_target": 7,
    "verdict_with_referent": 4,
}


def quality(profile: set[str], hi: int = 3, lo: int = 1) -> float:
    """best 1 of A + best 2 of B + best 1 of C, on a lo..hi ordinal."""

    def scores(group):
        return sorted((hi if k in profile else lo) for k in group)[::-1]

    return scores(A)[0] + sum(scores(B)[:2]) + scores(C)[0]


def n_penalised(profile: set[str]) -> int:
    return sum(1 for x, y in ANTAGONISTIC if x in profile and y in profile)


def enumerate_maxima() -> None:
    all_metrics = A + B + C
    best, rows = -1e9, []
    for r in range(len(all_metrics) + 1):
        for combo in itertools.combinations(all_metrics, r):
            p = set(combo)
            q = quality(p)
            if q > best:
                best, rows = q, []
            if q == best:
                rows.append(p)

    clean = [p for p in rows if n_penalised(p) == 0]
    minimal = [p for p in clean if len(p) == min(len(x) for x in clean)]

    print(
        f"max quality (1-3 scale)                : {best:.0f} of 12  (floor for an empty turn: {quality(set()):.0f})"
    )
    print(f"profiles achieving it                  : {len(rows)}")
    print(f"  ... with no antagonistic pair        : {len(clean)}")
    print(f"  ... and of minimum cardinality       : {len(minimal)}  (each uses {len(minimal[0])} metrics)")

    costed = sorted(minimal, key=lambda p: sum(MIN_WORDS[k] for k in p))
    print("\ncheapest maximal profiles, by summed minimum word cost (before sharing words):")
    for p in costed[:5]:
        cost = sum(MIN_WORDS[k] for k in p)
        print(f"  {cost:3d} words  {sorted(p)}")

    never = [m for m in all_metrics if not any(m in p for p in costed[:10])]
    print(f"\nmetrics absent from the 10 cheapest maxima: {never}")


def max_bias(sigma: float = 0.4, n: int = 400_000, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    print(f"\nupward bias of max/top-2 under probe noise, RMSE sigma={sigma} on a 1-3 scale")
    print("(all true values equal, so any positive number here is pure noise being rewarded)")
    for rho in (0.0, 0.3, 0.6):
        for label, k, take in (("A: best 1 of 5", 5, 1), ("B: best 2 of 6", 6, 2)):
            common = rng.standard_normal((n, 1))
            idio = rng.standard_normal((n, k))
            e = sigma * (np.sqrt(rho) * common + np.sqrt(1 - rho) * idio)
            bias = np.sort(e, axis=1)[:, ::-1][:, :take].sum(axis=1).mean()
            print(f"  rho={rho:.1f}  {label}: +{bias:.3f}")
    print("  (Group C is a single metric, so it contributes no max bias.)")


def gap_compression(sigma: float = 0.4, n: int = 200_000, seed: int = 1) -> None:
    """A flat-and-low bad turn against a peaked good turn."""
    rng = np.random.default_rng(seed)
    bad = {"A": np.full(5, 1.2), "B": np.full(6, 1.2)}
    good = {"A": np.array([3.0, 1.0, 1.0, 1.0, 1.0]), "B": np.array([3.0, 3.0, 1.0, 1.0, 1.0, 1.0])}

    def measured(prof):
        a = prof["A"] + sigma * rng.standard_normal((n, 5))
        b = prof["B"] + sigma * rng.standard_normal((n, 6))
        return a.max(axis=1) + np.sort(b, axis=1)[:, -2:].sum(axis=1)

    def true(prof):
        return prof["A"].max() + np.sort(prof["B"])[-2:].sum()

    mb, mg = measured(bad).mean(), measured(good).mean()
    tb, tg = true(bad), true(good)
    print(f"\ntrue     bad={tb:.2f}  good={tg:.2f}  gap={tg - tb:.2f}")
    print(f"measured bad={mb:.2f}  good={mg:.2f}  gap={mg - mb:.2f}")
    print(f"signal lost to max-bias: {100 * (1 - (mg - mb) / (tg - tb)):.0f}% of the good/bad gap")


def gate_vs_length() -> None:
    """What a multiplicative gate does to a length-positive quality term."""
    print("\nE[reward] = E[quality] x P(all three gates pass), for a plausible leak/conflict rate")
    print(" words  quality  P(gate)  E[reward]")
    for words, q, p in ((15, 7.5, 0.95), (30, 9.0, 0.88), (60, 10.5, 0.72), (100, 11.0, 0.55)):
        print(f"  {words:4d}   {q:5.2f}   {p:5.2f}   {q * p:6.2f}")
    print("  quality rises 47% from 15 to 100 words and E[reward] peaks at 30.")


def straddling(sigma: float = 0.4, n: int = 400_000, seed: int = 3) -> None:
    """Under max, being vaguely half-good at five things nearly ties being good at one."""
    rng = np.random.default_rng(seed)
    weak = (np.full((n, 5), 2.0) + sigma * rng.standard_normal((n, 5))).max(1).mean()
    sharp = (np.array([2.8, 1.0, 1.0, 1.0, 1.0]) + sigma * rng.standard_normal((n, 5))).max(1).mean()
    print(f"\nstraddler (all five truly 2.0) measures {weak:.3f}")
    print(f"specialist (one truly 2.8)    measures {sharp:.3f}")
    print(f"cost of doing nothing well: {sharp - weak:.3f} of one Group-A point")


def grpo_gate_variance(group: int = 8, per_gate_fail: float = 0.05, n: int = 200_000, seed: int = 4) -> None:
    """A zeroing gate inflates the group std that GRPO divides advantages by."""
    rng = np.random.default_rng(seed)
    print(f"\nGRPO group of {group}; quality ~ N(8, 1) among survivors")
    for fail in range(3):
        r = np.concatenate([rng.normal(8.0, 1.0, (n, group - fail)), np.zeros((n, fail))], axis=1)
        adv = (r - r.mean(1, keepdims=True)) / (r.std(1, keepdims=True) + 1e-6)
        print(
            f"  {fail} gated out: group std={r.std(1).mean():5.2f}  "
            f"mean |adv| among survivors={np.abs(adv[:, : group - fail]).mean():.3f}"
        )
    p_fail = 1 - (1 - per_gate_fail) ** 3
    p_clean = (1 - p_fail) ** group
    print(f"  three gates at {per_gate_fail:.0%} each -> {p_fail:.1%} of samples zeroed")
    print(f"  -> only {p_clean:.0%} of groups are clean; {1 - p_clean:.0%} have their signal halved")


def effective_temperature(sigma: float = 0.4) -> None:
    """d E[max]/d s_i = P(i is argmax), so probe noise silently softens the max."""
    print(f"\nWith probe RMSE {sigma}, max is really a soft-max whose temperature is the noise.")
    print("On a flat Group-A profile each of the five metrics carries ~20% of the gradient;")
    print("on a peaked one the argmax carries ~100%. Improving the probe SHARPENS the")
    print("objective and reintroduces the plateaus. Nobody chose that temperature.")


if __name__ == "__main__":
    enumerate_maxima()
    max_bias()
    gap_compression()
    gate_vs_length()
    straddling()
    grpo_gate_variance()
    effective_temperature()
