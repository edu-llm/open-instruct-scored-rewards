"""Correct an agent-labelled mean with a small human slice. Prediction-powered inference.

    python -m projects.learning_science_semantic_atlas.ppi mean \
        --proxy data/labels/agent_consensus.jsonl --gold data/labels/sophia.jsonl \
        --construct generative_elicitation --field presence --sampling-prob 0.2

    python -m projects.learning_science_semantic_atlas.ppi accuracy \
        --proxy data/labels/agent_consensus.jsonl --gold data/labels/sophia.jsonl \
        --system data/labels/policy_b.jsonl --construct generative_elicitation --field presence \
        --sampling-prob 0.2

THE PROBLEM. An agent judge can label the whole corpus, and it is biased by an
unknown amount. A human can label a hundred units, and is the definition of
truth but has no power. Reporting the agent's mean is precise and wrong;
reporting the human's mean is right and too wide to distinguish anything. The
rectified estimator uses both:

    theta = mean(f over every unit) + weighted mean over the gold slice of (y - f)

The first term is the agent's mean, cheap and dense. The second is the average
error of the agent, measured where a human looked. If the agent is unbiased the
correction is noise around zero; if it is biased by a constant the correction
removes exactly that constant. Either way the estimate is centred on what the
humans would have said - the agent buys precision, never truth.

THE RECTIFIER IS A WEIGHTED MEAN, NOT A WEIGHTED TOTAL. Each gold unit is
weighted by ``1/pi``, and the weights are normalised by their own sum rather than
by N. Under equal sampling probabilities that is exactly the plain mean of the
residual, which is the estimator in the PPI literature. Under unequal ones it is
the standard inverse-probability-weighted mean: a ratio estimator, so consistent
and asymptotically unbiased rather than exactly unbiased in a finite sample. The
alternative - dividing the inverse-probability-weighted total by N - is exactly
unbiased but adds the variance of the realized number of gold labels to every
interval. In a cluster bootstrap that term dominates, so a proxy that predicts
the human perfectly would still show a wide interval. The small finite-sample
bias is the better trade, and it vanishes when pi is constant.

WHAT THIS BUYS AND WHAT IT DOES NOT. It is unbiased for any proxy, good or bad: a
useless proxy widens the interval, it does not shift the estimate. It cannot
repair a gold slice that was not a probability sample. Those are the two facts
worth remembering, and the second one is where projects go wrong.

ASSUMPTIONS, all of them, because the estimator is only as good as these:

 1. The gold slice is a probability sample of the units, drawn independently of
    the labels. Humans who labelled "the interesting ones", or the ones an agent
    flagged as uncertain, break this and no interval fixes it. ``draw_gold_slice``
    exists so the design is made by this module rather than reconstructed later.
 2. ``sampling_prob`` is the design probability, known before the labels existed.
    Estimating it after the fact from the labels reintroduces the bias it corrects.
 3. Proxy labels exist for every unit and were produced without seeing any gold
    label. A judge few-shotted on gold units is correlated with its own
    correction; ``labeling.py`` quarantines exemplars by item for this reason, and
    quarantined units should be excluded here as well.
 4. Human labels are the estimand's definition. PPI corrects agent-versus-human
    disagreement; it says nothing about whether the human was right.
 5. Units sharing an ``item_id`` are dependent - same stem, same reference, often
    the same student turn - so every interval here resamples whole items. Treating
    variants of one item as independent would understate the width by roughly the
    square root of the number of variants per item.
 6. Clusters are independent and reasonably numerous. The percentile bootstrap
    undercovers with very few clusters, so ``n_clusters`` is reported alongside
    every interval; below about twenty, read the interval as indicative.
 7. The estimand is a mean over the N units supplied. A field that is null for
    some units (fidelity is null where the construct is absent) is not a mean over
    all N until you say what null means, which is what ``null_as`` is for.
    Conditioning on presence instead would condition on an estimated event.

FAILING RATHER THAN GUESSING. Invalid sampling probabilities and unrecognised
gold ids raise ``PPIError``. Silent coercion here would produce a plausible number
with no defensible meaning, which is worse than a traceback.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from projects.learning_science_semantic_atlas.labeling import read_jsonl

# Six standard deviations of the cluster-level count, so the realized-size check
# fires on a misdescribed design and not on an unlucky draw.
COUNT_TOLERANCE_SD = 6.0
FEW_CLUSTERS = 20


class PPIError(ValueError):
    """The design is not one this estimator can be honest about."""


@dataclass(frozen=True)
class GoldDesign:
    """A gold slice and the probabilities that produced it. Keep this next to the labels."""

    unit_ids: tuple[Hashable, ...]
    item_ids: tuple[Hashable, ...]
    sampling_prob: float
    seed: int


@dataclass(frozen=True)
class PPIEstimate:
    """A rectified mean, its bootstrap interval, and what it is being compared with."""

    theta: float
    ci_low: float
    ci_high: float
    se: float
    proxy_mean: float
    gold_mean: float
    gold_ci_low: float
    gold_ci_high: float
    n_units: int
    n_gold: int
    n_clusters: int
    n_gold_clusters: int
    n_boot: int
    n_boot_used: int
    alpha: float

    @property
    def rectifier(self) -> float:
        """How far the agent's mean was off. This is the whole correction."""
        return self.theta - self.proxy_mean

    @property
    def width(self) -> float:
        return self.ci_high - self.ci_low

    @property
    def gold_width(self) -> float:
        return self.gold_ci_high - self.gold_ci_low

    @property
    def gain(self) -> float:
        """Gold-only interval width over this one: what the proxy labels bought.

        Both widths can be zero when every item is internally identical, which is
        a property of the corpus and not a free infinite gain, so that case is 1.
        """
        if self.width:
            return self.gold_width / self.width
        return float("inf") if self.gold_width else 1.0

    @property
    def few_clusters(self) -> bool:
        return self.n_clusters < FEW_CLUSTERS

    def summary(self) -> str:
        note = "  (few clusters; indicative only)" if self.few_clusters else ""
        # Classical PPI is unbiased for any proxy but not guaranteed to be tighter
        # than gold alone: a proxy uncorrelated with the human adds its own
        # variance to the rectifier. "narrowed the interval 0.47x" would read to a
        # skimming eye as the opposite of what happened, so the direction is named.
        if not math.isfinite(self.gain):
            gain = "removed the interval entirely"
        elif self.gain > 1.0:
            gain = f"narrowed the interval {self.gain:.2f}x"
        elif self.gain == 1.0:
            gain = "left the interval unchanged"
        elif self.gain > 0.0:
            gain = f"WIDENED the interval {1.0 / self.gain:.2f}x - this proxy is not buying precision"
        else:
            gain = "WIDENED the interval from nothing at all - all of the width is the proxy's"
        return (
            f"theta={self.theta:.4f}  {1 - self.alpha:.0%} CI [{self.ci_low:.4f}, {self.ci_high:.4f}]  se={self.se:.4f}\n"
            f"  proxy-only mean {self.proxy_mean:.4f} (rectifier {self.rectifier:+.4f})\n"
            f"  gold-only mean  {self.gold_mean:.4f}  CI [{self.gold_ci_low:.4f}, {self.gold_ci_high:.4f}]"
            f"  -> proxy labels {gain}\n"
            f"  {self.n_units} units in {self.n_clusters} items; "
            f"{self.n_gold} gold in {self.n_gold_clusters} items; "
            f"{self.n_boot_used}/{self.n_boot} resamples usable{note}"
        )


@dataclass(frozen=True)
class _Design:
    """Validated arrays. Everything past this point can assume the design is sane."""

    proxy: np.ndarray
    gold: np.ndarray
    labelled: np.ndarray
    prob: np.ndarray
    cluster: np.ndarray
    n_clusters: int


def _as_prob_map(sampling_prob: float | Mapping[Hashable, float], unit_ids: Sequence[Hashable]) -> np.ndarray:
    """Per-unit probabilities. Ids must match the units exactly - no near-misses."""
    if not isinstance(sampling_prob, Mapping):
        return np.full(len(unit_ids), float(sampling_prob), dtype=float)
    unknown = sorted(map(str, set(sampling_prob) - set(unit_ids)))[:5]
    if unknown:
        raise PPIError(f"sampling_prob has ids that are not units: {unknown}")
    missing = sorted(map(str, set(unit_ids) - set(sampling_prob)))[:5]
    if missing:
        raise PPIError(f"sampling_prob is missing a probability for unit(s) {missing}")
    return np.asarray([float(sampling_prob[u]) for u in unit_ids], dtype=float)


def _validate(
    unit_ids: Sequence[Hashable],
    item_ids: Sequence[Hashable],
    proxy: Sequence[float],
    gold: Mapping[Hashable, float],
    sampling_prob: float | Mapping[Hashable, float],
) -> _Design:
    if len(unit_ids) == 0:
        raise PPIError("no units")
    if not (len(unit_ids) == len(item_ids) == len(proxy)):
        raise PPIError(f"unit_ids, item_ids and proxy must align: got {len(unit_ids)}, {len(item_ids)}, {len(proxy)}")

    duplicates = [str(u) for u, count in collections.Counter(unit_ids).items() if count > 1][:5]
    if duplicates:
        raise PPIError(f"duplicate unit id(s) {duplicates}; every unit must appear once")
    if any(i is None or str(i) == "" for i in item_ids):
        raise PPIError("every unit needs an item_id: the interval is clustered on it")

    proxy_array = np.asarray([float("nan") if v is None else float(v) for v in proxy], dtype=float)
    bad = [str(u) for u, v in zip(unit_ids, proxy_array) if not math.isfinite(v)][:5]
    if bad:
        raise PPIError(
            f"proxy label missing or not finite for unit(s) {bad}. PPI needs a proxy for every unit; "
            "if a field is null by contract, pass null_as to say what null means."
        )

    if not gold:
        raise PPIError("the gold slice is empty; there is nothing to correct with")
    strays = sorted(map(str, set(gold) - set(unit_ids)))[:5]
    if strays:
        raise PPIError(f"gold ids that are not units in this corpus: {strays}")

    index = {u: i for i, u in enumerate(unit_ids)}
    gold_array = np.zeros(len(unit_ids), dtype=float)
    labelled = np.zeros(len(unit_ids), dtype=bool)
    for unit_id, value in gold.items():
        if value is None or not math.isfinite(float(value)):
            raise PPIError(f"gold label for {unit_id!r} is {value!r}; a human label must be a finite number")
        position = index[unit_id]
        gold_array[position] = float(value)
        labelled[position] = True

    prob = _as_prob_map(sampling_prob, unit_ids)
    if not np.all(np.isfinite(prob)):
        raise PPIError("sampling probabilities must be finite")
    invalid = [str(u) for u, p in zip(unit_ids, prob) if not 0.0 < p <= 1.0][:5]
    if invalid:
        raise PPIError(f"sampling probabilities must be in (0, 1]; offending unit(s) {invalid}")
    impossible = [str(u) for u, p, got in zip(unit_ids, prob, labelled) if got and p <= 0][:5]
    if impossible:
        raise PPIError(f"unit(s) {impossible} carry a gold label but had probability 0 of being sampled")
    certain = [str(u) for u, p, got in zip(unit_ids, prob, labelled) if p >= 1.0 and not got][:5]
    if certain:
        raise PPIError(
            f"unit(s) {certain} have sampling probability 1 but no gold label. "
            "Either the design is not what was passed, or those labels are missing."
        )

    codes = {}
    cluster = np.empty(len(unit_ids), dtype=int)
    for position, item_id in enumerate(item_ids):
        cluster[position] = codes.setdefault(item_id, len(codes))

    _check_realized_count(prob, cluster, labelled, len(codes))
    return _Design(
        proxy=proxy_array, gold=gold_array, labelled=labelled, prob=prob, cluster=cluster, n_clusters=len(codes)
    )


def _check_realized_count(prob: np.ndarray, cluster: np.ndarray, labelled: np.ndarray, n_clusters: int) -> None:
    """Does the gold slice look like it came from these probabilities?

    ``sum(pi)`` is the expected number of gold labels. A realized count far from
    it means the probabilities describe a different design than the one that
    produced the slice, which is the one error in PPI that leaves no trace in the
    output. The tolerance is computed at cluster level because units within an
    item are sampled together, which makes the count far more variable than a
    unit-level calculation would suggest.
    """
    expected = float(prob.sum())
    size = np.bincount(cluster, minlength=n_clusters).astype(float)
    rate = np.bincount(cluster, weights=prob, minlength=n_clusters) / size
    tolerance = COUNT_TOLERANCE_SD * math.sqrt(float((size**2 * rate * (1.0 - rate)).sum()))
    realized = float(labelled.sum())
    if abs(realized - expected) > max(tolerance, 1e-9):
        raise PPIError(
            f"{int(realized)} gold labels but the sampling probabilities expect about {expected:.1f} "
            f"(tolerance {tolerance:.1f}). The probabilities do not describe the slice that was labelled; "
            "pass the design that was actually used."
        )


@dataclass(frozen=True)
class _Totals:
    """Per-cluster sums, so a bootstrap resample is a matrix product rather than a loop."""

    size: np.ndarray
    proxy: np.ndarray
    residual: np.ndarray
    gold: np.ndarray
    weight: np.ndarray

    def evaluate(self, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(theta, gold_only, weight_sum)`` for one or many cluster resamples."""
        weight_sum = counts @ self.weight
        safe = np.where(weight_sum > 0, weight_sum, 1.0)
        theta = (counts @ self.proxy) / (counts @ self.size) + (counts @ self.residual) / safe
        return theta, (counts @ self.gold) / safe, weight_sum


def _cluster_totals(design: _Design) -> _Totals:
    weight = np.where(design.labelled, 1.0 / design.prob, 0.0)  # zero off the slice, so sums need no masking
    by_cluster = lambda values: np.bincount(design.cluster, weights=values, minlength=design.n_clusters)  # noqa: E731
    return _Totals(
        size=np.bincount(design.cluster, minlength=design.n_clusters).astype(float),
        proxy=by_cluster(design.proxy),
        residual=by_cluster(weight * (design.gold - design.proxy)),
        gold=by_cluster(weight * design.gold),
        weight=by_cluster(weight),
    )


def ppi_mean(
    unit_ids: Sequence[Hashable],
    item_ids: Sequence[Hashable],
    proxy: Sequence[float],
    gold: Mapping[Hashable, float],
    sampling_prob: float | Mapping[Hashable, float],
    *,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
    min_usable: float = 0.9,
) -> PPIEstimate:
    """Rectified mean of the human label, with a cluster bootstrap interval.

    ``gold`` maps unit id to human label and covers only the sampled slice;
    ``proxy`` covers every unit in order. ``sampling_prob`` is a scalar when every
    unit had the same chance of being sampled, or a per-unit mapping otherwise.

    Resamples items, not units, and discards resamples that happen to contain no
    gold unit - in those the rectifier is silently zero, so keeping them would
    drag the bootstrap distribution towards the uncorrected proxy mean and make
    the interval look tighter than it is.
    """
    if n_boot < 2:
        raise PPIError(f"n_boot must be at least 2 to form an interval, got {n_boot}")
    if not 0.0 < alpha < 1.0:
        raise PPIError(f"alpha must be in (0, 1), got {alpha}")
    design = _validate(unit_ids, item_ids, proxy, gold, sampling_prob)
    totals = _cluster_totals(design)

    whole = np.ones((1, design.n_clusters))
    point, gold_point, _ = totals.evaluate(whole)
    theta, gold_mean = float(point[0]), float(gold_point[0])
    proxy_mean = float(design.proxy.mean())

    rng = np.random.default_rng(seed)
    counts = rng.multinomial(design.n_clusters, np.full(design.n_clusters, 1.0 / design.n_clusters), size=n_boot)
    boot_theta, boot_gold, weight_sum = totals.evaluate(counts)
    usable = weight_sum > 0
    if usable.sum() < max(min_usable * n_boot, 1):
        raise PPIError(
            f"only {int(usable.sum())} of {n_boot} item-level resamples contained a gold unit. "
            f"The gold slice covers {int(np.unique(design.cluster[design.labelled]).size)} of "
            f"{design.n_clusters} items, which is too few to bootstrap; label more items."
        )

    low, high = float(alpha / 2), float(1 - alpha / 2)
    theta_draws = boot_theta[usable]
    gold_draws = boot_gold[usable]
    return PPIEstimate(
        theta=theta,
        ci_low=float(np.quantile(theta_draws, low)),
        ci_high=float(np.quantile(theta_draws, high)),
        se=float(theta_draws.std(ddof=1)) if theta_draws.size > 1 else float("nan"),
        proxy_mean=proxy_mean,
        gold_mean=gold_mean,
        gold_ci_low=float(np.quantile(gold_draws, low)),
        gold_ci_high=float(np.quantile(gold_draws, high)),
        n_units=len(unit_ids),
        n_gold=int(design.labelled.sum()),
        n_clusters=design.n_clusters,
        n_gold_clusters=int(np.unique(design.cluster[design.labelled]).size),
        n_boot=n_boot,
        n_boot_used=int(usable.sum()),
        alpha=alpha,
    )


def ppi_accuracy(
    unit_ids: Sequence[Hashable],
    item_ids: Sequence[Hashable],
    system: Sequence[object],
    agent: Sequence[object],
    human: Mapping[Hashable, object],
    sampling_prob: float | Mapping[Hashable, float],
    *,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
    min_usable: float = 0.9,
) -> PPIEstimate:
    """Accuracy of ``system`` against human labels, corrected by the agent judge.

    Accuracy is a mean of an indicator, so this is ``ppi_mean`` on two derived
    columns: agreement with the agent everywhere, which is the proxy, and
    agreement with the human on the slice, which is the gold. Reporting
    agreement-with-the-agent alone measures how much the system resembles the
    judge; the rectifier converts that into how often it is right.
    """
    if not (len(unit_ids) == len(system) == len(agent)):
        raise PPIError(f"system and agent labels must cover every unit: {len(unit_ids)}, {len(system)}, {len(agent)}")
    missing = [str(u) for u, value in zip(unit_ids, agent) if value is None][:5]
    if missing:
        raise PPIError(f"agent label missing for unit(s) {missing}; the proxy must cover every unit")
    missing = [str(u) for u, value in zip(unit_ids, system) if value is None][:5]
    if missing:
        raise PPIError(f"system output missing for unit(s) {missing}")

    by_id = dict(zip(unit_ids, system))
    strays = sorted(map(str, set(human) - set(unit_ids)))[:5]
    if strays:
        raise PPIError(f"gold ids that are not units in this corpus: {strays}")
    proxy = [float(s == a) for s, a in zip(system, agent)]
    gold = {}
    for unit_id, value in human.items():
        if value is None:
            raise PPIError(f"human label for {unit_id!r} is null; drop the unit or label it")
        gold[unit_id] = float(by_id[unit_id] == value)
    return ppi_mean(
        unit_ids, item_ids, proxy, gold, sampling_prob, n_boot=n_boot, seed=seed, alpha=alpha, min_usable=min_usable
    )


def draw_gold_slice(
    unit_ids: Sequence[Hashable], item_ids: Sequence[Hashable], n_items: int, *, seed: int = 0
) -> GoldDesign:
    """Choose whole items uniformly at random for human labelling.

    Items rather than units, for two reasons. The packets present every variant of
    an item together, so a human who labels one labels them all, and sampling the
    unit would misdescribe what happened. And sampling whole clusters keeps the
    slice independent across the same unit the interval resamples.
    """
    if len(unit_ids) != len(item_ids):
        raise PPIError(f"unit_ids and item_ids must align: {len(unit_ids)} vs {len(item_ids)}")
    repeated = [str(u) for u, count in collections.Counter(unit_ids).items() if count > 1][:5]
    if repeated:
        # Usually a label file holding several raters, which would put the same
        # unit in the design more than once and misstate what a human was asked for.
        raise PPIError(f"duplicate unit id(s) {repeated}; the design needs one row per unit")
    order = list(dict.fromkeys(item_ids))  # first appearance order, so the draw is reproducible
    if not 0 < n_items <= len(order):
        raise PPIError(f"n_items must be between 1 and the {len(order)} items available, got {n_items}")

    rng = np.random.default_rng(seed)
    picked = {order[i] for i in rng.choice(len(order), size=n_items, replace=False)}
    return GoldDesign(
        unit_ids=tuple(u for u, i in zip(unit_ids, item_ids) if i in picked),
        item_ids=tuple(i for i in order if i in picked),
        sampling_prob=n_items / len(order),
        seed=seed,
    )


# ---------------------------------------------------------------- label records


def records_to_columns(
    records: Iterable[Mapping[str, object]],
    field: str,
    *,
    construct: str = "",
    null_as: float | None = None,
    exclude_quarantined: bool = True,
) -> tuple[list[Hashable], list[Hashable], list[float]]:
    """Label records to aligned (unit_ids, item_ids, values) for one construct and field."""
    unit_ids: list[Hashable] = []
    item_ids: list[Hashable] = []
    values: list[float] = []
    seen_constructs: set[str] = set()
    for record in records:
        seen_constructs.add(str(record.get("construct", "")))
        if construct and str(record.get("construct", "")) != construct:
            continue
        if exclude_quarantined and record.get("quarantined"):
            continue
        raw = record.get(field)
        if raw is None:
            if null_as is None:
                raise PPIError(
                    f"unit {record.get('unit_id')!r} has {field}=null. Pass null_as to make the estimand a mean "
                    f"over every unit (scoring null as that value), or restrict the corpus before calling."
                )
            raw = null_as
        unit_ids.append(str(record.get("unit_id")))
        item_ids.append(str(record.get("item_id")))
        values.append(float(raw))
    if construct and not unit_ids and construct not in seen_constructs:
        # Otherwise a mistyped construct arrives downstream as "no units", which
        # reads like a corpus problem rather than a spelling one.
        raise PPIError(f"no records for construct {construct!r}; the file has {sorted(seen_constructs)}")
    return unit_ids, item_ids, values


def by_unit(unit_ids: Sequence[Hashable], values: Sequence[float], *, what: str = "label") -> dict[Hashable, float]:
    """``{unit_id: value}``, refusing a repeated id rather than letting the last one win.

    ``dict(zip(...))`` over label records collapses silently, and the two files
    where that matters are the ones this module is built around: a gold file
    holding two humans returns one human's labels, and a system file holding two
    checkpoints returns one checkpoint's. Either way the estimate is of something
    nobody asked for, and nothing in the output says so.
    """
    if len(unit_ids) != len(values):
        raise PPIError(f"{what}: {len(unit_ids)} ids and {len(values)} values do not align")
    out: dict[Hashable, float] = {}
    repeated: list[str] = []
    for unit_id, value in zip(unit_ids, values):
        if unit_id in out:
            repeated.append(str(unit_id))
        out[unit_id] = value
    if repeated:
        raise PPIError(
            f"{what}: unit(s) {sorted(set(repeated))[:5]} appear more than once. "
            "Split the file by rater, or reduce it to one value per unit first."
        )
    return out


# ------------------------------------------------------------------------- CLI


def _load(path: str) -> list[dict[str, object]]:
    return list(read_jsonl(path))


def _parse_prob(text: str) -> float | Mapping[Hashable, float]:
    if text.endswith(".json"):
        with open(text) as handle:
            blob = json.load(handle)
        if not isinstance(blob, dict):
            raise SystemExit(f"{text}: expected an object mapping unit_id to probability")
        return {str(k): float(v) for k, v in blob.items()}
    return float(text)


def _mean(args: argparse.Namespace) -> None:
    null_as = None if args.null_as is None else float(args.null_as)
    unit_ids, item_ids, proxy = records_to_columns(
        _load(args.proxy), args.field, construct=args.construct, null_as=null_as
    )
    gold_ids, _, gold_values = records_to_columns(
        _load(args.gold), args.field, construct=args.construct, null_as=null_as
    )
    estimate = ppi_mean(
        unit_ids,
        item_ids,
        proxy,
        by_unit(gold_ids, gold_values, what=f"gold labels in {args.gold}"),
        _parse_prob(args.sampling_prob),
        n_boot=args.n_boot,
        seed=args.seed,
        alpha=args.alpha,
    )
    print(f"{args.construct or 'all constructs'} / {args.field}\n  {estimate.summary()}")


def _accuracy(args: argparse.Namespace) -> None:
    null_as = None if args.null_as is None else float(args.null_as)
    unit_ids, item_ids, agent = records_to_columns(
        _load(args.proxy), args.field, construct=args.construct, null_as=null_as
    )
    system_ids, _, system_values = records_to_columns(
        _load(args.system), args.field, construct=args.construct, null_as=null_as
    )
    gold_ids, _, gold_values = records_to_columns(
        _load(args.gold), args.field, construct=args.construct, null_as=null_as
    )
    by_id = by_unit(system_ids, system_values, what=f"system labels in {args.system}")
    missing = [u for u in unit_ids if u not in by_id][:5]
    if missing:
        raise SystemExit(f"system labels are missing unit(s) {missing}")
    estimate = ppi_accuracy(
        unit_ids,
        item_ids,
        [by_id[u] for u in unit_ids],
        agent,
        by_unit(gold_ids, gold_values, what=f"gold labels in {args.gold}"),
        _parse_prob(args.sampling_prob),
        n_boot=args.n_boot,
        seed=args.seed,
        alpha=args.alpha,
    )
    print(f"{args.construct or 'all constructs'} / {args.field}, accuracy of {args.system}\n  {estimate.summary()}")


def _draw(args: argparse.Namespace) -> None:
    unit_ids, item_ids, _ = records_to_columns(_load(args.proxy), "applicable", construct=args.construct, null_as=0.0)
    design = draw_gold_slice(unit_ids, item_ids, args.n_items, seed=args.seed)
    with open(args.out, "w") as handle:
        json.dump(
            {
                "schema": "semantic-atlas/gold-design-v1",
                "seed": design.seed,
                "sampling_prob": design.sampling_prob,
                "item_ids": list(design.item_ids),
                "unit_ids": list(design.unit_ids),
            },
            handle,
            indent=1,
        )
    print(
        f"{len(design.item_ids)} items ({len(design.unit_ids)} units) at probability "
        f"{design.sampling_prob:.4f} -> {args.out}\nLabel exactly these. Substituting an easier item breaks the estimator."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    for name, help_text in (("mean", "rectified mean of a field"), ("accuracy", "rectified accuracy of a system")):
        cmd = sub.add_parser(name, help=help_text)
        cmd.add_argument("--proxy", required=True, help="agent labels, every unit")
        cmd.add_argument("--gold", required=True, help="human labels, the sampled slice")
        cmd.add_argument("--construct", default="")
        cmd.add_argument("--field", default="presence")
        cmd.add_argument("--sampling-prob", required=True, help="a number, or a .json of unit_id -> probability")
        cmd.add_argument("--null-as", default=None, help="value to score a null field as; omit to fail on nulls")
        cmd.add_argument("--n-boot", type=int, default=2000)
        cmd.add_argument("--seed", type=int, default=0)
        cmd.add_argument("--alpha", type=float, default=0.05)
        if name == "accuracy":
            cmd.add_argument("--system", required=True, help="labels of the system being evaluated")

    draw = sub.add_parser("draw", help="choose the items a human should label")
    draw.add_argument("--proxy", required=True, help="any label file covering every unit")
    draw.add_argument("--construct", default="")
    draw.add_argument("--n-items", type=int, required=True)
    draw.add_argument("--seed", type=int, default=0)
    draw.add_argument("--out", required=True)

    args = parser.parse_args()
    if args.cmd == "mean":
        _mean(args)
    elif args.cmd == "accuracy":
        _accuracy(args)
    else:
        _draw(args)


if __name__ == "__main__":
    main()
