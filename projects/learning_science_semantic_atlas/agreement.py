"""Rater agreement per construct. THE DECISION GATE, and it runs before anything else.

    python -m projects.learning_science_semantic_atlas.agreement \
        --labels data/labels/*.jsonl --min-pairs 20

    # the same rater's second pass over the same units
    python -m projects.learning_science_semantic_atlas.agreement \
        --labels data/labels/gpt_pass1.jsonl --retest data/labels/gpt_pass2.jsonl

WHY THIS COMES FIRST. Nothing downstream can be more reliable than the labels it
is computed from. If two careful raters disagree about whether a construct is
present, the construct is not well posed, and no volume of labelling and no
better model repairs it - the ceiling is set by the question. Run this the moment
two raters have overlapped, while the rubric can still be rewritten cheaply.

WHICH NUMBER TO READ, and why it is not always the same one. Cohen's kappa
corrects observed agreement for the agreement two raters would reach by guessing
independently at their own base rates. That correction misbehaves exactly where
this project lives: on a construct present in one unit in twenty, two raters can
agree on 90% of units and still score kappa below zero, because chance agreement
is estimated at 90% too. The number is not wrong, it is answering a question
nobody asked - it measures agreement about *which* rare units are positive, and
it divides by almost nothing.

So the headline statistic is chosen by the label, not by taste:

    ordinal (fidelity, quality)     linearly weighted kappa. Linear rather than
                                    quadratic: quadratic forgives one-point gaps
                                    so heavily that a scale nobody agrees on can
                                    still look respectable.
    binary, prevalence 0.2 to 0.8   Cohen's kappa.
    binary, rarer or commoner       Gwet's AC1, whose chance term is built from
                                    the mean marginal rather than the product of
                                    the marginals, and so stays interpretable as
                                    prevalence goes to zero.

Every statistic is reported for every field regardless, because a construct where
raw agreement is high and kappa is negative is a construct with a prevalence
problem, and seeing both numbers is how you find that out.

THE GATE, applied to applicability, presence and fidelity. Quality is advisory:
it is an all-things-considered judgement, raters are expected to differ about it,
and low agreement there does not invalidate the construct.

    >= 0.60   keep.
    0.40-0.60 keep, but the ceiling is low. Report downstream results against it.
    < 0.40    do not use this construct for measurement. Rewrite it and re-label.
    too few   underpowered, which is a different verdict from bad. Label more.

UNDERPOWERED COVERS THREE WAYS OF NOT HAVING THE EVIDENCE, kept apart from
unreliability because each of them otherwise arrives as a number that reads like
a pass:

    too few pairs       fewer than ``min_pairs`` comparable judgements.
    too few positives   fewer than ``min_positive`` units anyone called positive.
                        AC1 climbs towards 1 on agreed negatives alone, so a
                        construct found three times in forty scores 1.00 without
                        anyone having shown they can find it.
    a constant field    nobody's rating ever moved. Both kappas are 0/0 there and
                        AC1 answers 1.0, so the headline is undefined for binary
                        fields exactly as it is for ordinal ones. The single
                        exception is unanimous applicability: if no rater ever
                        declined then there is no applicability call to gate, and
                        it excluded nothing from the fields below.

A high not-applicable rate is the same signal arriving politely: the raters are
telling you the construct does not fit the corpus.

TEST-RETEST is the same rater twice, and it bounds everything else. A rater who
does not agree with themselves cannot agree with anyone, so a low retest number
with a high inter-rater number means the raters share a bias rather than a
reading.
"""

from __future__ import annotations

import argparse
import collections
import glob
import itertools
import math
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from projects.learning_science_semantic_atlas.labeling import (
    BINARY_FIELDS,
    LABEL_SCHEMA,
    ORDINAL_FIELDS,
    Scale,
    read_jsonl,
)

GATE_PASS = 0.60
GATE_CAUTION = 0.40
MIN_PAIRS = 20

# Units somebody has to have called positive before agreement about a binary
# field means anything. On a construct found three times in forty, AC1 is 1.00
# and is entirely the thirty-seven agreed negatives; the raters have not yet
# demonstrated they can find the construct, which is a power problem and not a
# reliability one. LABELING_GUIDE.md sets the same floor as gate G1.
MIN_POSITIVE = 8

# Below this positive rate (or above its complement) Cohen's kappa is dividing by
# a chance term that has swallowed the signal, and AC1 becomes the honest report.
LOW_PREVALENCE = 0.20

GATE_FIELDS: tuple[str, ...] = ("applicable", "presence")
ADVISORY_FIELDS: tuple[str, ...] = ("quality",)
ALL_FIELDS: tuple[str, ...] = (*BINARY_FIELDS, *ORDINAL_FIELDS)
MIN_FIDELITY_UNITS = 8
MIN_FIDELITY_KAPPA = 0.40


# ------------------------------------------------------------------ statistics


def percent_agreement(pairs: Sequence[tuple[int, int]]) -> float:
    return statistics.fmean(float(x == y) for x, y in pairs) if pairs else float("nan")


def within_one(pairs: Sequence[tuple[int, int]]) -> float:
    return statistics.fmean(float(abs(x - y) <= 1) for x, y in pairs) if pairs else float("nan")


def _check_in_scale(values: Iterable[int], cats: Sequence[int], where: str) -> None:
    """Refuse a rating the scale does not contain, instead of dropping it.

    Every chance term here is a sum over ``cats``. A rating outside them is
    counted in the observed agreement and missed by the expected agreement, so
    the statistic comes back plausible and wrong rather than absent.
    """
    stray = sorted({v for v in values if v not in set(cats)})
    if stray:
        raise ValueError(f"{where}: rating(s) {stray} are outside the scale {list(cats)}")


def cohens_kappa(pairs: Sequence[tuple[int, int]], categories: Sequence[int] | None = None) -> float:
    """Chance-corrected agreement, chance being the raters' independent marginals.

    Returns NaN when the chance term reaches 1, which happens when both raters
    used a single category throughout. That is not a bug to paper over: it is the
    degenerate end of the prevalence problem AC1 exists to handle.
    """
    if not pairs:
        return float("nan")
    cats = list(categories) if categories is not None else sorted({v for pair in pairs for v in pair})
    _check_in_scale((v for pair in pairs for v in pair), cats, "cohens_kappa")
    n = len(pairs)
    observed = statistics.fmean(float(x == y) for x, y in pairs)
    left = collections.Counter(x for x, _ in pairs)
    right = collections.Counter(y for _, y in pairs)
    expected = sum((left[c] / n) * (right[c] / n) for c in cats)
    return (observed - expected) / (1 - expected) if expected < 1 else float("nan")


def weighted_kappa(pairs: Sequence[tuple[int, int]], scale: Scale, weights: str = "linear") -> float:
    """Kappa with partial credit for being one point off, over an ordinal scale."""
    cats = list(range(scale.lo, scale.hi + 1))
    span = max(len(cats) - 1, 1)
    if weights == "linear":
        cost = lambda i, j: abs(i - j) / span  # noqa: E731
    elif weights == "quadratic":
        cost = lambda i, j: ((i - j) / span) ** 2  # noqa: E731
    else:  # checked before the empty case, so a typo cannot pass by having no data
        raise ValueError(f"weights must be 'linear' or 'quadratic', got {weights!r}")
    if not pairs:
        return float("nan")

    n = len(pairs)
    index = {c: i for i, c in enumerate(cats)}
    _check_in_scale((v for pair in pairs for v in pair), cats, f"weighted_kappa on {scale.name}")
    left = collections.Counter(x for x, _ in pairs)
    right = collections.Counter(y for _, y in pairs)

    observed = statistics.fmean(cost(index[x], index[y]) for x, y in pairs)
    expected = sum(cost(index[a], index[b]) * (left[a] / n) * (right[b] / n) for a in cats for b in cats)
    return 1.0 - observed / expected if expected else float("nan")


def gwet_ac1(item_ratings: Iterable[Sequence[int]], categories: Sequence[int] | None = None) -> float:
    """Gwet's AC1: agreement corrected by a chance term that does not chase prevalence.

    Cohen estimates chance agreement as the product of each rater's marginals,
    which grows towards 1 as a category becomes universal, so kappa collapses on
    rare constructs even when the raters agree almost everywhere. AC1 estimates
    it from the mean marginal instead - ``sum pi_k (1 - pi_k) / (q - 1)`` - which
    goes to zero as one category takes over, leaving AC1 near the observed
    agreement, which is the answer a reader expects.

    Takes one sequence of ratings per item, so items rated by different numbers
    of raters are handled directly.
    """
    usable = [list(r) for r in item_ratings if len(r) >= 2]
    if not usable:
        return float("nan")
    cats = list(categories) if categories is not None else sorted({v for r in usable for v in r})
    _check_in_scale((v for r in usable for v in r), cats, "gwet_ac1")
    q = len(cats)
    if q < 2:
        return 1.0  # a single category was ever used; there is nothing to correct for

    observed = statistics.fmean(
        sum(r.count(c) * (r.count(c) - 1) for c in cats) / (len(r) * (len(r) - 1)) for r in usable
    )
    marginal = {c: statistics.fmean(r.count(c) / len(r) for r in usable) for c in cats}
    expected = sum(marginal[c] * (1 - marginal[c]) for c in cats) / (q - 1)
    return (observed - expected) / (1 - expected) if expected < 1 else float("nan")


def prevalence(pairs: Sequence[tuple[int, int]]) -> float:
    """Positive rate over both raters' votes, for deciding whether kappa is usable."""
    if not pairs:
        return float("nan")
    return statistics.fmean(float(v) for pair in pairs for v in pair)


# -------------------------------------------------------------- label plumbing


@dataclass(frozen=True)
class FieldAgreement:
    """Every statistic for one field, plus which one is the headline and why."""

    construct: str
    field: str
    kind: str
    n_units: int
    n_pairs: int
    n_raters: int
    exact: float
    headline: str
    value: float
    cohen: float = float("nan")
    kappa_w: float = float("nan")
    ac1: float = float("nan")
    within1: float = float("nan")
    prevalence: float = float("nan")
    # Binary fields only: units where somebody voted each way. The minority count
    # is the evidence a chance-corrected number is standing on.
    n_positive: int = 0
    n_negative: int = 0

    @property
    def degenerate(self) -> bool:
        """Pairs exist but nobody varied, so chance agreement is total and the headline is undefined.

        Reported rather than smoothed over. Cohen's kappa and weighted kappa are
        0/0 here; AC1 answers 1.0, because its chance term goes to zero, which is
        defensible as a statistic and useless as a gate: a judge that gives every
        unit the same score would sail through. Unanimity on a constant is not
        evidence that the raters mean the same thing, so the headline stays
        undefined for binary fields exactly as it does for ordinal ones, and the
        construct stays ungated until someone's ratings move. AC1 is still
        reported in its own column - that is the number the reader needs to see
        to understand why the field is undefined.
        """
        return self.n_pairs > 0 and math.isnan(self.value)

    def verdict(self, min_pairs: int = MIN_PAIRS, min_positive: int = MIN_POSITIVE) -> str:
        if self.n_pairs < min_pairs or math.isnan(self.value):
            return "underpowered"
        if self.kind == "binary" and self.n_positive < min_positive:
            return "underpowered"
        if self.value >= GATE_PASS:
            return "pass"
        return "caution" if self.value >= GATE_CAUTION else "fail"


@dataclass(frozen=True)
class ConceptReliability:
    construct: str
    fields: dict[str, FieldAgreement]
    not_applicable_rate: float
    rubric_misfit_rate: float
    n_units: int
    n_raters: int
    verdict: str
    measurement_mode: str
    reasons: tuple[str, ...] = ()

    @property
    def usable(self) -> bool:
        return self.verdict in ("pass", "caution")


def index_labels(
    records: Iterable[Mapping[str, object]], *, exclude_quarantined: bool = True
) -> dict[str, dict[str, dict[str, Mapping[str, object]]]]:
    """``{construct: {unit_id: {rater: record}}}``.

    Units quarantined as few-shot exemplars are dropped by default. A rater shown
    the answer agrees with whoever produced it, and counting that as agreement
    inflates the one number the whole project is gated on.
    """
    out: dict[str, dict[str, dict[str, Mapping[str, object]]]] = {}
    for record in records:
        if exclude_quarantined and record.get("quarantined"):
            continue
        construct = str(record.get("construct", ""))
        unit_id = str(record.get("unit_id", ""))
        rater = str(record.get("rater", ""))
        if not unit_id or not rater:
            raise ValueError(f"label record needs unit_id and rater, got {dict(record)}")
        by_unit = out.setdefault(construct, {}).setdefault(unit_id, {})
        if rater in by_unit:
            raise ValueError(f"rater {rater!r} labelled unit {unit_id!r} twice; use test_retest for repeat passes")
        by_unit[rater] = record
    return out


def _value(record: Mapping[str, object], name: str) -> int | None:
    """A field as an int, or None when the contract says it does not apply.

    Nulls already encode inapplicability - fidelity is null unless present,
    everything is null when not applicable - so pairing on non-null values is
    exactly complete-case analysis with no special cases.
    """
    raw = record.get(name)
    if raw is None:
        return None
    if isinstance(raw, bool):
        return int(raw)
    if isinstance(raw, int):
        return raw
    raise ValueError(f"field {name!r} must be a bool, int or null, got {raw!r}")


@dataclass(frozen=True)
class _Comparable:
    """What one field of one construct contributes, gathered in a single pass."""

    pairs: tuple[tuple[int, int], ...]
    ratings: tuple[tuple[int, ...], ...]
    n_units: int
    n_positive: int
    n_negative: int


def _comparable(units: Mapping[str, Mapping[str, Mapping[str, object]]], name: str) -> _Comparable:
    """Rater pairs and per-unit ratings for one field, plus how the votes fell.

    An item rated by three raters contributes three pairs, so it counts more than
    one rated by two. That is the usual pooling and it is fine with a balanced
    rater set; with a wildly unbalanced one, read AC1, which takes the ratings
    per unit rather than pooled pairs.

    Raters are ordered by name before pairing. Pooled pairs are ordered, so
    Cohen's kappa reads one rater's marginal as the row and the other's as the
    column; taking them in whatever order the label files happened to be
    concatenated in moved weighted kappa by 0.18 across the orderings of four
    raters on fixed data. Sorting does not make the pooled statistic symmetric -
    with three or more raters it cannot be, which is another reason AC1 is the
    number to read there - but it does make it a property of the data.
    """
    pairs: list[tuple[int, int]] = []
    ratings: list[tuple[int, ...]] = []
    positive = negative = 0
    for records in units.values():
        values = [v for v in (_value(records[rater], name) for rater in sorted(records)) if v is not None]
        if len(values) < 2:
            continue
        pairs.extend(itertools.combinations(values, 2))
        ratings.append(tuple(values))
        positive += any(v == 1 for v in values)
        negative += any(v == 0 for v in values)
    return _Comparable(
        pairs=tuple(pairs), ratings=tuple(ratings), n_units=len(ratings), n_positive=positive, n_negative=negative
    )


def field_agreement(
    construct: str, units: Mapping[str, Mapping[str, Mapping[str, object]]], name: str, *, weights: str = "linear"
) -> FieldAgreement:
    """Agreement on one field of one construct, with the headline statistic chosen."""
    got = _comparable(units, name)
    pairs = got.pairs
    raters = {r for records in units.values() for r in records}
    scale = ORDINAL_FIELDS.get(name)
    categories = tuple(range(scale.lo, scale.hi + 1)) if scale else (0, 1)
    exact = percent_agreement(pairs)
    ac1 = gwet_ac1(got.ratings, categories)

    if scale is not None:
        kappa_w = weighted_kappa(pairs, scale, weights)
        return FieldAgreement(
            construct=construct,
            field=name,
            kind="ordinal",
            n_units=got.n_units,
            n_pairs=len(pairs),
            n_raters=len(raters),
            exact=exact,
            headline="kappa_w",
            value=kappa_w,
            kappa_w=kappa_w,
            ac1=ac1,
            within1=within_one(pairs),
            cohen=cohens_kappa(pairs, categories),
        )

    rate = prevalence(pairs)
    cohen = cohens_kappa(pairs, categories)
    rare = rate == rate and (rate < LOW_PREVALENCE or rate > 1 - LOW_PREVALENCE)
    headline = "ac1" if rare else "cohen"
    # One category throughout leaves nothing for a chance correction to work on.
    # Cohen's kappa already says so by being 0/0; AC1 answers 1.0 and would hand a
    # rubber-stamping judge a pass, so the headline follows kappa here and AC1
    # keeps its column.
    varied = len({v for pair in pairs for v in pair}) > 1
    return FieldAgreement(
        construct=construct,
        field=name,
        kind="binary",
        n_units=got.n_units,
        n_pairs=len(pairs),
        n_raters=len(raters),
        exact=exact,
        headline=headline,
        value=(ac1 if rare else cohen) if varied else float("nan"),
        cohen=cohen,
        ac1=ac1,
        prevalence=rate,
        n_positive=got.n_positive,
        n_negative=got.n_negative,
    )


def not_applicable_rate(units: Mapping[str, Mapping[str, Mapping[str, object]]]) -> float:
    """Share of labels where a rater declined the construct as not applicable."""
    votes = [_value(r, "applicable") for records in units.values() for r in records.values()]
    present = [v for v in votes if v is not None]
    return statistics.fmean(float(v == 0) for v in present) if present else float("nan")


def _flags(record: Mapping[str, object]) -> tuple[str, ...]:
    """A record's flags, tolerating the single-string spelling of a one-flag list."""
    raw = record.get("flags") or ()
    if isinstance(raw, str):
        return (raw,)
    return tuple(str(f) for f in raw)


def rubric_misfit_rate(units: Mapping[str, Mapping[str, Mapping[str, object]]]) -> float:
    """Share of label appearances explicitly flagged ``rubric_misfit``."""
    votes = [float("rubric_misfit" in _flags(r)) for records in units.values() for r in records.values()]
    return statistics.fmean(votes) if votes else float("nan")


def reliability_gate(
    records: Iterable[Mapping[str, object]],
    *,
    min_pairs: int = MIN_PAIRS,
    min_positive: int = MIN_POSITIVE,
    weights: str = "linear",
    exclude_quarantined: bool = True,
) -> dict[str, ConceptReliability]:
    """Per-construct verdict on whether the labels can carry a measurement.

    Applicability and presence determine whether the construct is measurable.
    Fidelity is gated separately: LABELING_GUIDE.md G3 keeps reliable binary
    presence even when the ordinal is noisy, so a failed fidelity gate changes
    ``measurement_mode`` to ``binary_only`` rather than sinking the construct.
    """
    indexed = index_labels(records, exclude_quarantined=exclude_quarantined)
    out: dict[str, ConceptReliability] = {}
    for construct, units in indexed.items():
        fields = {name: field_agreement(construct, units, name, weights=weights) for name in ALL_FIELDS}
        reasons: list[str] = []
        verdict = "pass"
        for name in GATE_FIELDS:
            field = fields[name]
            got = field.verdict(min_pairs, min_positive)
            if name == "applicable" and field.degenerate and field.n_negative == 0:
                # Every rater said the construct applies to every unit. There is no
                # applicability judgement to be reliable about, and none was used to
                # drop anything from presence or fidelity below, so this is a
                # decision nobody made rather than one nobody agreed on.
                reasons.append("applicable: no rater ever declined, so there is no applicability call to gate")
                continue
            if got == "underpowered" and field.degenerate:
                reasons.append(
                    f"{name}: {field.n_pairs} pairs but every rating identical, so no chance-corrected "
                    f"statistic is defined - agreement on a constant proves nothing"
                )
            elif got == "underpowered" and field.n_pairs < min_pairs:
                reasons.append(f"{name}: only {field.n_pairs} comparable pairs, need {min_pairs}")
            elif got == "underpowered":
                reasons.append(
                    f"{name}: only {field.n_positive} unit(s) anyone called positive, need {min_positive} - "
                    f"{field.headline}={field.value:.2f} is carried by the negatives"
                )
            elif got == "fail":
                reasons.append(f"{name}: {field.headline}={field.value:.2f} below {GATE_CAUTION:.2f}")
            elif got == "caution":
                reasons.append(f"{name}: {field.headline}={field.value:.2f}, low ceiling")
            verdict = _worse(verdict, got)
        fidelity = fields["fidelity"]
        fidelity_usable = (
            fidelity.n_units >= MIN_FIDELITY_UNITS
            and fidelity.value == fidelity.value
            and fidelity.value >= MIN_FIDELITY_KAPPA
        )
        measurement_mode = "presence_and_fidelity" if fidelity_usable else "binary_only"
        if not fidelity_usable:
            detail = (
                f"only {fidelity.n_units} both-present units, need {MIN_FIDELITY_UNITS}"
                if fidelity.n_units < MIN_FIDELITY_UNITS
                else (
                    "chance-corrected fidelity agreement is undefined"
                    if fidelity.value != fidelity.value
                    else f"kappa_w={fidelity.value:.2f} below {MIN_FIDELITY_KAPPA:.2f}"
                )
            )
            reasons.append(f"fidelity: {detail}; retain presence and discard the ordinal")
        misfit = rubric_misfit_rate(units)
        if misfit == misfit and misfit > 0.20:
            reasons.append(f"rubric_misfit: {misfit:.0%} exceeds the 20% rewrite threshold")
            verdict = _worse(verdict, "fail")
        out[construct] = ConceptReliability(
            construct=construct,
            fields=fields,
            not_applicable_rate=not_applicable_rate(units),
            rubric_misfit_rate=misfit,
            n_units=len(units),
            n_raters=len({r for recs in units.values() for r in recs}),
            verdict=verdict,
            measurement_mode=measurement_mode,
            reasons=tuple(reasons),
        )
    return out


_ORDER = {"pass": 0, "caution": 1, "underpowered": 2, "fail": 3}


def _worse(left: str, right: str) -> str:
    return left if _ORDER[left] >= _ORDER[right] else right


# ------------------------------------------------------------------ test-retest


def test_retest(
    first: Iterable[Mapping[str, object]],
    second: Iterable[Mapping[str, object]],
    *,
    weights: str = "linear",
    exclude_quarantined: bool = True,
) -> dict[str, dict[str, FieldAgreement]]:
    """One rater against their own second pass, per construct and field.

    Matching is on (construct, unit, rater), so only units the same rater saw
    twice count. Raters present in only one pass are ignored rather than compared
    with someone else, which would silently turn intra-rater reliability into
    inter-rater reliability and report the wrong ceiling.
    """
    left = index_labels(first, exclude_quarantined=exclude_quarantined)
    right = index_labels(second, exclude_quarantined=exclude_quarantined)

    out: dict[str, dict[str, FieldAgreement]] = {}
    matched_raters: set[str] = set()
    for construct, units in left.items():
        paired: dict[str, dict[str, Mapping[str, object]]] = {}
        for unit_id, records in units.items():
            repeat = right.get(construct, {}).get(unit_id, {})
            for rater, record in records.items():
                if rater not in repeat:
                    continue
                matched_raters.add(rater)
                # Two passes of one rater are two "raters" as far as the pairing goes.
                paired[f"{unit_id}\x1f{rater}"] = {"pass1": record, "pass2": repeat[rater]}
        if paired:
            out[construct] = {name: field_agreement(construct, paired, name, weights=weights) for name in ALL_FIELDS}
    if not matched_raters:
        raise ValueError(
            "no rater labelled the same unit in both passes - test-retest is unmeasurable.\n"
            "Re-run the same rater over a slice of the same packet, keeping the rater name."
        )
    return out


# -------------------------------------------------------------------- consensus


def majority(values: Sequence[int]) -> int | None:
    """The modal value, or None when the top two tie.

    A tie is data. Breaking it by rater order invents a label and hides the units
    that most need a human, so it is returned as unresolved instead.
    """
    if not values:
        return None
    counts = collections.Counter(values).most_common()
    if len(counts) > 1 and counts[0][1] == counts[1][1]:
        return None
    return counts[0][0]


def _round_half_up(value: float) -> int:
    # Python's round() goes to even, so round(0.5)=0 and round(1.5)=2; on an
    # ordinal scale that is an arbitrary asymmetry between adjacent categories.
    return math.floor(value + 0.5)


def consensus_ordinal(values: Sequence[int], method: str = "median") -> int | None:
    """A single ordinal value from several, staying on the rubric's own scale."""
    if not values:
        return None
    if method == "median":
        return _round_half_up(statistics.median(values))
    if method == "mean":
        return _round_half_up(statistics.fmean(values))
    raise ValueError(f"method must be 'median' or 'mean', got {method!r}")


def consensus_labels(
    records: Iterable[Mapping[str, object]],
    *,
    min_raters: int = 2,
    method: str = "median",
    exclude_quarantined: bool = False,
) -> list[dict[str, object]]:
    """Collapse several raters into one label per unit, following the contract.

    Applicability is decided first and gates the rest, exactly as the label
    contract does: a unit the panel calls not applicable has no presence to
    average, and fidelity is only defined where presence carried. Fields left
    unresolved by a tie are null and named in ``unresolved`` so a caller can send
    them to a human instead of quietly averaging them away.

    A unit any rater saw as a few-shot exemplar stays flagged ``quarantined``, and
    a flag any rater raised survives as the union. Consensus output is the usual
    proxy input to ``ppi.py``, which excludes quarantined units by default and
    cannot do so from a row that dropped the flag on its way through here; the
    same holds for the rubric-misfit rate, which is read off these records.
    """
    indexed = index_labels(records, exclude_quarantined=exclude_quarantined)
    out: list[dict[str, object]] = []
    for construct, units in indexed.items():
        for unit_id, by_rater in sorted(units.items()):
            if len(by_rater) < min_raters:
                continue
            some = next(iter(by_rater.values()))
            unresolved: list[str] = []
            row: dict[str, object] = {
                "schema": LABEL_SCHEMA,
                "unit_id": unit_id,
                "item_id": some.get("item_id"),
                "construct": construct,
                "rater": "consensus",
                "n_raters": len(by_rater),
                "quarantined": any(bool(r.get("quarantined")) for r in by_rater.values()),
                "flags": sorted({f for r in by_rater.values() for f in _flags(r)}),
                "applicable": None,
                "presence": None,
                "fidelity": None,
                "quality": None,
            }

            applicable = majority([v for v in (_value(r, "applicable") for r in by_rater.values()) if v is not None])
            if applicable is None:
                unresolved.append("applicable")
                row["unresolved"] = unresolved
                out.append(row)
                continue
            row["applicable"] = bool(applicable)
            if not applicable:
                row["unresolved"] = unresolved
                out.append(row)
                continue

            presence = majority([v for v in (_value(r, "presence") for r in by_rater.values()) if v is not None])
            quality = [v for v in (_value(r, "quality") for r in by_rater.values()) if v is not None]
            row["quality"] = consensus_ordinal(quality, method)
            if not quality:
                unresolved.append("quality")
            if presence is None:
                unresolved.append("presence")
            else:
                row["presence"] = bool(presence)
                if presence:
                    fidelity = [v for v in (_value(r, "fidelity") for r in by_rater.values()) if v is not None]
                    row["fidelity"] = consensus_ordinal(fidelity, method)
                    if not fidelity:
                        unresolved.append("fidelity")
            row["unresolved"] = unresolved
            out.append(row)
    return out


# ------------------------------------------------------------------------- CLI


def _load(patterns: Sequence[str]) -> list[dict[str, object]]:
    paths = sorted(set(itertools.chain.from_iterable(glob.glob(p) or [p] for p in patterns)))
    records: list[dict[str, object]] = []
    for path in paths:
        records.extend(read_jsonl(path))
    if not records:
        raise SystemExit(f"no label records found in {patterns}")
    return records


def _cell(value: float, width: int, spec: str = ".2f") -> str:
    """A dash where a statistic does not apply, so the eye goes to the ones that do."""
    return f"{'-':>{width}}" if math.isnan(value) else f"{value:>{width}{spec}}"


def _print_fields(fields: Mapping[str, FieldAgreement], min_pairs: int, min_positive: int = MIN_POSITIVE) -> None:
    header = (
        f"    {'field':<12} {'pairs':>6} {'pos':>5} {'exact':>7} {'prev':>6} {'cohen':>7} "
        f"{'kappa_w':>8} {'ac1':>7}  {'headline':>9} {'verdict':>13}"
    )
    print(header)
    print("    " + "-" * (len(header) - 4))
    for name in ALL_FIELDS:
        got = fields[name]
        positives = f"{got.n_positive:>5}" if got.kind == "binary" else f"{'-':>5}"
        cells = (
            f"    {name:<12} {got.n_pairs:>6} {positives} {_cell(got.exact, 7, '.0%')} "
            f"{_cell(got.prevalence, 6)} {_cell(got.cohen, 7)} {_cell(got.kappa_w, 8)} {_cell(got.ac1, 7)}  "
            f"{got.headline:>9} {got.verdict(min_pairs, min_positive):>13}"
        )
        advisory = "  (advisory)" if name in ADVISORY_FIELDS else ""
        print(cells + advisory)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", nargs="+", required=True, help="label jsonl, one or more; globs allowed")
    parser.add_argument("--retest", nargs="*", default=[], help="a second pass by the same rater(s)")
    parser.add_argument("--min-pairs", type=int, default=MIN_PAIRS)
    parser.add_argument("--min-positive", type=int, default=MIN_POSITIVE, help="units someone called positive")
    parser.add_argument("--weights", default="linear", choices=("linear", "quadratic"))
    parser.add_argument("--include-quarantined", action="store_true", help="count few-shot units too (do not)")
    args = parser.parse_args()

    records = _load(args.labels)
    gates = reliability_gate(
        records,
        min_pairs=args.min_pairs,
        min_positive=args.min_positive,
        weights=args.weights,
        exclude_quarantined=not args.include_quarantined,
    )
    raters = sorted({str(r.get("rater")) for r in records})
    print(f"{len(records)} labels, {len(raters)} raters ({', '.join(raters)}), {len(gates)} constructs\n")

    for construct, gate in sorted(gates.items()):
        print(
            f"  {construct}  —  {gate.verdict.upper()}   "
            f"{gate.measurement_mode}   {gate.n_units} units, {gate.n_raters} raters"
        )
        print(f"    not applicable on {gate.not_applicable_rate:.0%} of labels", end="")
        print("   <- a high rate means the construct does not fit the corpus" if gate.not_applicable_rate > 0.3 else "")
        print(f"    rubric_misfit on {gate.rubric_misfit_rate:.0%} of labels", end="")
        print("   <- exceeds the 20% rewrite threshold" if gate.rubric_misfit_rate > 0.2 else "")
        _print_fields(gate.fields, args.min_pairs, args.min_positive)
        for reason in gate.reasons:
            print(f"    ! {reason}")
        print()

    # Unreliable and unmeasured are different verdicts with different remedies, so
    # they are not printed as one list of things to avoid.
    failing = sorted(c for c, g in gates.items() if g.verdict == "fail")
    if failing:
        print(f"  DO NOT MEASURE WITH: {', '.join(failing)}")
        print("     Rewrite the construct and re-label a slice. A number computed here is rater noise.")
    thin = sorted(c for c, g in gates.items() if g.verdict == "underpowered")
    if thin:
        print(f"  NOT YET MEASURABLE: {', '.join(thin)}")
        print("     Not a failure - there is no evidence either way yet. Label more, and oversample")
        print("     the positives if the reason above is that too few units drew one.")
    usable = sorted(c for c, g in gates.items() if g.usable)
    if usable:
        print(f"  Usable: {', '.join(usable)}")
        print("     Carry the headline value forward; downstream results are bounded by it, not by 1.0.")

    if args.retest:
        print("\n  Test-retest, same rater twice — the ceiling on everything above")
        retest = test_retest(
            records, _load(args.retest), weights=args.weights, exclude_quarantined=not args.include_quarantined
        )
        for construct, fields in sorted(retest.items()):
            print(f"\n  {construct}")
            _print_fields(fields, args.min_pairs, args.min_positive)


if __name__ == "__main__":
    main()
