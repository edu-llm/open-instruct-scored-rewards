"""Analysis APIs over semantic-atlas traces.

Everything here is built around one refusal: a probe that separates two texts
has not found a construct until it beats what the texts differ on anyway.
Four control families are therefore first-class rather than optional.

- lexical: TF-IDF over the response text;
- surface: length, sentence count, digit and case rates;
- global quality: the matched quality score, used both as a control feature set
  and as a residualization target;
- token histogram and token identity: the hashed bag of response tokens, and
  the single token sitting at the traced position. Routing especially is
  token-driven, so a routing result that a token-identity control also achieves
  is a tokenization result.

Discrimination is measured *within* quality-matched sibling groups. A sibling
group contributes one paired margin, groups are the bootstrap and permutation
unit, and folds are grouped so no sibling group spans a train/test boundary.

``named_versus_enacted`` separates what a response claims from what it does: a
probe trained on congruent records is scored on the mismatch cells, where the
named and enacted labels disagree and only one of them can be right.
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, normalized_mutual_info_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from projects.learning_science_semantic_atlas.trace import SITES, AtlasTrace, load_atlas_trace

SCHEMA = "semantic-atlas-representation-report/v1"
CS = (0.01, 0.1, 1.0, 10.0)
SEED = 1701


# --------------------------------------------------------------------------
# statistics over clustered units
# --------------------------------------------------------------------------


def cluster_bootstrap_ci(
    values: dict[str, float] | Sequence[float], *, confidence: float = 0.95, samples: int = 5000, seed: int = SEED
) -> tuple[float, float, float]:
    """Resample whole clusters, because sibling groups are not independent rows."""
    array = np.asarray(list(values.values()) if isinstance(values, dict) else list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.choice(array, size=(samples, array.size), replace=True).mean(axis=1)
    alpha = (1.0 - confidence) / 2
    low, high = np.quantile(draws, [alpha, 1 - alpha])
    return float(array.mean()), float(low), float(high)


def sign_flip_test(values: dict[str, float] | Sequence[float], *, samples: int = 10000, seed: int = SEED) -> float:
    """Exchangeability test for paired margins; the sign is the only free choice."""
    array = np.asarray(list(values.values()) if isinstance(values, dict) else list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    observed = abs(float(array.mean()))
    signs = rng.choice((-1.0, 1.0), size=(samples, array.size))
    exceed = int((np.abs((signs * array).mean(axis=1)) >= observed).sum())
    return (exceed + 1) / (samples + 1)


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    if not len(p_values):
        return []
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 1.0
    for rank_index in range(len(order) - 1, -1, -1):
        original = int(order[rank_index])
        running = min(running, float(p_values[original]) * len(order) / (rank_index + 1))
        adjusted[original] = running
    return adjusted.tolist()


def sibling_margins(scores: np.ndarray, labels: np.ndarray, sibling_ids: np.ndarray) -> dict[str, float]:
    """One positive-minus-negative margin per quality-matched sibling group."""
    grouped: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for score, label, sibling in zip(scores, labels, sibling_ids):
        if int(label) < 0:
            continue
        grouped[str(sibling)][int(label)].append(float(score))
    return {
        sibling: float(np.mean(by_label[1]) - np.mean(by_label[0]))
        for sibling, by_label in grouped.items()
        if by_label.get(0) and by_label.get(1)
    }


# --------------------------------------------------------------------------
# grouped cross-validation
# --------------------------------------------------------------------------


def grouped_folds(
    groups: np.ndarray, labels: np.ndarray, *, n_splits: int = 5, seed: int = SEED
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Folds that never split a group, stratified by label when that is feasible."""
    unique = np.unique(groups)
    n_splits = int(min(n_splits, len(unique)))
    if n_splits < 2:
        raise ValueError(f"grouped cross-validation needs at least two groups, found {len(unique)}")
    indices = np.arange(len(groups))
    try:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = list(splitter.split(indices, labels, groups))
    except ValueError:
        folds = list(GroupKFold(n_splits=n_splits).split(indices, labels, groups))
    assert_group_disjoint(folds, groups)
    return folds


def assert_group_disjoint(folds: Iterable[tuple[np.ndarray, np.ndarray]], groups: np.ndarray) -> None:
    for fold_index, (train, test) in enumerate(folds):
        shared = set(np.asarray(groups)[train]) & set(np.asarray(groups)[test])
        if shared:
            raise AssertionError(f"fold {fold_index} leaks groups across the split: {sorted(shared)[:3]}")


class Residualizer:
    """Least-squares removal of nuisance columns, fitted on training rows only."""

    def __init__(self) -> None:
        self.coefficients: np.ndarray | None = None

    @staticmethod
    def _design(controls: np.ndarray) -> np.ndarray:
        controls = np.asarray(controls, dtype=float)
        controls = controls.reshape(len(controls), -1)
        return np.hstack([np.ones((len(controls), 1)), np.nan_to_num(controls)])

    def fit(self, features: np.ndarray, controls: np.ndarray) -> Residualizer:
        design = self._design(controls)
        self.coefficients = np.linalg.lstsq(design, np.asarray(features, dtype=float), rcond=None)[0]
        return self

    def transform(self, features: np.ndarray, controls: np.ndarray) -> np.ndarray:
        if self.coefficients is None:
            raise RuntimeError("Residualizer.transform called before fit")
        return np.asarray(features, dtype=float) - self._design(controls) @ self.coefficients


# --------------------------------------------------------------------------
# feature sets
# --------------------------------------------------------------------------


@dataclass
class FeatureSet:
    """Named model input plus the pipeline family it needs."""

    name: str
    data: Any
    kind: str = "dense"  # dense | text | categorical
    role: str = "activation"  # activation | control

    def subset(self, index: np.ndarray) -> Any:
        if self.kind == "text":
            return [self.data[position] for position in index]
        return np.asarray(self.data)[index]

    def take(self, keep: np.ndarray) -> FeatureSet:
        return FeatureSet(name=self.name, data=self.subset(np.flatnonzero(keep)), kind=self.kind, role=self.role)


@dataclass
class AnalysisView:
    """Rows a slot actually covers, with the label and grouping columns aligned.

    A boundary slot is missing whenever a response has fewer sentences than the
    trace budget, so every analysis restricts to covered, labelled rows instead
    of scoring zero-filled vectors.
    """

    keep: np.ndarray
    labels: np.ndarray
    siblings: np.ndarray
    groups: np.ndarray
    texts: list[str] | None

    @property
    def coverage(self) -> float:
        return float(self.keep.mean())

    @property
    def n(self) -> int:
        return int(self.keep.sum())


def analysis_view(
    trace: AtlasTrace,
    slot: int,
    *,
    label_key: str = "labels",
    group_key: str = "item_ids",
    texts: Sequence[str] | None = None,
) -> AnalysisView:
    labels = trace[label_key].astype(int)
    keep = trace.valid(slot) & (labels >= 0)
    if keep.sum() < 4:
        raise ValueError(f"slot {trace.roles[slot]!r} covers only {int(keep.sum())} labelled records")
    return AnalysisView(
        keep=keep,
        labels=labels[keep],
        siblings=trace["sibling_ids"][keep],
        groups=trace[group_key][keep],
        texts=[texts[position] for position in np.flatnonzero(keep)] if texts else None,
    )


def _estimator(kind: str, c: float):
    logistic = LogisticRegression(C=c, class_weight="balanced", max_iter=4000, random_state=SEED, solver="liblinear")
    if kind == "text":
        return make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=5000, sublinear_tf=True), logistic
        )
    if kind == "categorical":
        return make_pipeline(OneHotEncoder(handle_unknown="ignore"), logistic)
    return make_pipeline(StandardScaler(), logistic)


def _decision(model, data) -> np.ndarray:
    return np.asarray(model.decision_function(data), dtype=float).reshape(-1)


def _take(values, index):
    return [values[position] for position in index] if isinstance(values, list) else values[index]


def out_of_fold_scores(
    features: FeatureSet,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    n_splits: int = 5,
    seed: int = SEED,
    residualize_on: np.ndarray | None = None,
) -> tuple[np.ndarray, list[float]]:
    """Out-of-fold decision values with the penalty chosen inside each training fold.

    Selecting ``C`` on the outer test fold would inflate every number the rest
    of this module reports, so a second grouped split inside the training fold
    does it instead. Residualization is likewise fitted on training rows only.
    """
    labels = np.asarray(labels).astype(int)
    scores = np.full(len(labels), np.nan, dtype=float)
    chosen: list[float] = []
    if residualize_on is not None and features.kind != "dense":
        raise ValueError(f"cannot residualize a {features.kind} feature set")
    for train, test in grouped_folds(groups, labels, n_splits=n_splits, seed=seed):
        x_train, x_test = features.subset(train), features.subset(test)
        y_train = labels[train]
        if residualize_on is not None:
            residualizer = Residualizer().fit(x_train, residualize_on[train])
            x_train = residualizer.transform(x_train, residualize_on[train])
            x_test = residualizer.transform(x_test, residualize_on[test])
        try:
            inner = grouped_folds(np.asarray(groups)[train], y_train, n_splits=min(3, n_splits), seed=seed)
        except ValueError:
            inner = []
        best_c, best_score = CS[0], -np.inf
        for c in CS if inner else ():
            fold_scores = [
                balanced_accuracy_score(
                    y_train[inner_test],
                    _estimator(features.kind, c)
                    .fit(_take(x_train, inner_train), y_train[inner_train])
                    .predict(_take(x_train, inner_test)),
                )
                for inner_train, inner_test in inner
            ]
            if float(np.mean(fold_scores)) > best_score:
                best_c, best_score = c, float(np.mean(fold_scores))
        scores[test] = _decision(_estimator(features.kind, best_c).fit(x_train, y_train), x_test)
        chosen.append(best_c)
    return scores, chosen


def activation_features(trace: AtlasTrace, site: str, layer_index: int, slot: int) -> FeatureSet:
    return FeatureSet(name=f"{site}@L{layer_index}:{trace.roles[slot]}", data=trace.features(site, layer_index, slot))


def surface_control(trace: AtlasTrace) -> FeatureSet:
    return FeatureSet(name="surface", data=trace["surface_features"].astype(np.float32), role="control")


def token_histogram_control(trace: AtlasTrace) -> FeatureSet:
    return FeatureSet(name="token_histogram", data=trace["token_histogram"].astype(np.float32), role="control")


def token_identity_control(trace: AtlasTrace, slot: int) -> FeatureSet:
    return FeatureSet(
        name="token_identity",
        data=trace["position_token_id"][:, slot].reshape(-1, 1),
        kind="categorical",
        role="control",
    )


def quality_control(trace: AtlasTrace) -> FeatureSet | None:
    quality = trace["quality"].astype(np.float32)
    if not np.isfinite(quality).any():
        return None
    return FeatureSet(name="global_quality", data=np.nan_to_num(quality).reshape(-1, 1), role="control")


def lexical_control(texts: Sequence[str] | None) -> FeatureSet | None:
    if not texts:
        return None
    return FeatureSet(name="lexical", data=list(texts), kind="text", role="control")


def within_sibling_variation(features: np.ndarray, siblings: np.ndarray) -> float:
    """Fraction of sibling groups whose members differ at all at this position.

    Quality-matched siblings share a prompt, so ``prompt_last`` is byte-identical
    across a pair: the model has not read either response yet. Discrimination
    there is exactly zero for a reason that has nothing to do with the construct,
    and a report that does not say so reads like a null result.
    """
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for row, sibling in zip(np.asarray(features), siblings):
        grouped[str(sibling)].append(row)
    varying = [
        1.0 if len(rows) > 1 and not all(np.array_equal(rows[0], other) for other in rows[1:]) else 0.0
        for rows in grouped.values()
    ]
    return float(np.mean(varying)) if varying else float("nan")


def control_features(trace: AtlasTrace, slot: int, texts: Sequence[str] | None = None) -> list[FeatureSet]:
    candidates = [
        lexical_control(texts),
        surface_control(trace),
        quality_control(trace),
        token_histogram_control(trace),
        token_identity_control(trace, slot),
    ]
    return [candidate for candidate in candidates if candidate is not None]


# --------------------------------------------------------------------------
# routing features
# --------------------------------------------------------------------------


def require_routing(trace: AtlasTrace) -> None:
    if not trace.has_routing:
        raise ValueError("this trace has no routing; it came from a dense model")


def expert_count(trace: AtlasTrace) -> int:
    require_routing(trace)
    return int(trace["router_logits"].shape[-1])


def selection_multihot(trace: AtlasTrace, slot: int) -> np.ndarray:
    """``[records, layers * experts]`` indicators of which experts fired."""
    require_routing(trace)
    indices = trace["topk_indices"][:, :, slot, :].astype(np.int64)
    n_records, n_layers, _ = indices.shape
    experts = expert_count(trace)
    out = np.zeros((n_records, n_layers, experts), dtype=np.float32)
    record_grid, layer_grid = np.meshgrid(np.arange(n_records), np.arange(n_layers), indexing="ij")
    for position in range(indices.shape[-1]):
        out[record_grid, layer_grid, indices[:, :, position]] = 1.0
    return out.reshape(n_records, n_layers * experts)


def router_entropy(trace: AtlasTrace, slot: int) -> np.ndarray:
    """``[records, layers]`` entropy in nats of the full router distribution."""
    require_routing(trace)
    probabilities = trace["router_probs"][:, :, slot, :].astype(np.float64)
    probabilities = probabilities / np.clip(probabilities.sum(axis=-1, keepdims=True), 1e-12, None)
    return -(probabilities * np.log(np.clip(probabilities, 1e-12, None))).sum(axis=-1)


def native_topk_mass(trace: AtlasTrace, slot: int) -> np.ndarray:
    """``[records, layers]`` sum of OLMoE's native top-k weights.

    With ``norm_topk_prob=false`` this is the real selected mass and stays below
    one. With normalization on it is one by construction, which is why
    ``selected_probability_mass`` exists.
    """
    require_routing(trace)
    return trace["topk_weights"][:, :, slot, :].astype(np.float64).sum(axis=-1)


def selected_probability_mass(trace: AtlasTrace, slot: int) -> np.ndarray:
    """``[records, layers]`` softmax mass on the selected experts, normalization aside."""
    require_routing(trace)
    probabilities = trace["router_probs"][:, :, slot, :].astype(np.float64)
    indices = trace["topk_indices"][:, :, slot, :].astype(np.int64)
    return np.take_along_axis(probabilities, indices, axis=-1).sum(axis=-1)


def expert_usage(trace: AtlasTrace, slots: Sequence[int] | None = None) -> np.ndarray:
    """``[layers, experts]`` share of selections each expert receives."""
    require_routing(trace)
    slots = list(range(trace["topk_indices"].shape[2])) if slots is None else list(slots)
    mask = trace["position_mask"][:, slots].astype(bool)
    indices = trace["topk_indices"][:, :, slots, :].astype(np.int64)
    n_layers = indices.shape[1]
    experts = expert_count(trace)
    counts = np.zeros((n_layers, experts), dtype=np.float64)
    for layer in range(n_layers):
        selected = indices[:, layer][mask]
        counts[layer] = np.bincount(selected.reshape(-1), minlength=experts)
    totals = counts.sum(axis=1, keepdims=True)
    return counts / np.clip(totals, 1.0, None)


def routing_feature_sets(trace: AtlasTrace, slot: int) -> list[FeatureSet]:
    require_routing(trace)
    probabilities = trace["router_probs"][:, :, slot, :].astype(np.float32)
    summary = np.concatenate(
        [router_entropy(trace, slot), native_topk_mass(trace, slot), selected_probability_mass(trace, slot)], axis=1
    ).astype(np.float32)
    return [
        FeatureSet(name="router_probs", data=probabilities.reshape(len(probabilities), -1)),
        FeatureSet(name="selection_multihot", data=selection_multihot(trace, slot)),
        FeatureSet(name="routing_summary", data=summary),
    ]


def path_label_association(
    trace: AtlasTrace, slot: int, labels: np.ndarray, *, permutations: int = 200, seed: int = SEED
) -> dict[str, Any]:
    """How much a token's cross-layer expert path tells you about the label.

    A path is a high-cardinality categorical, so raw mutual information is
    optimistic; the permutation null is the number that matters.
    """
    require_routing(trace)
    valid = trace.valid(slot) & (np.asarray(labels) >= 0)
    paths = trace["expert_path_id"][:, slot][valid]
    observed_labels = np.asarray(labels)[valid].astype(int)
    if paths.size == 0 or len(np.unique(observed_labels)) < 2:
        return {"n": int(paths.size), "normalized_mutual_information": float("nan"), "null_p95": float("nan")}
    codes = np.unique(paths, return_inverse=True)[1]
    with warnings.catch_warnings():
        # Paths are a deliberately high-cardinality categorical; sklearn's
        # "this looks like regression" heuristic does not apply, and the
        # permutation null below is what corrects for the cardinality.
        warnings.simplefilter("ignore", UserWarning)
        observed = float(normalized_mutual_info_score(observed_labels, codes))
        rng = np.random.default_rng(seed)
        null = [
            float(normalized_mutual_info_score(rng.permutation(observed_labels), codes)) for _ in range(permutations)
        ]
    return {
        "n": int(paths.size),
        "distinct_paths": int(len(np.unique(paths))),
        "normalized_mutual_information": observed,
        "null_p95": float(np.quantile(null, 0.95)),
        "exceeds_null": bool(observed > float(np.quantile(null, 0.95))),
    }


# --------------------------------------------------------------------------
# discrimination
# --------------------------------------------------------------------------


@dataclass
class DiscriminationResult:
    name: str
    n_groups: int
    forced_choice_accuracy: float
    mean_margin: float
    margin_ci95: tuple[float, float]
    p_sign_flip: float
    chosen_c: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_groups": self.n_groups,
            "forced_choice_accuracy": self.forced_choice_accuracy,
            "mean_margin": self.mean_margin,
            "margin_ci95": list(self.margin_ci95),
            "p_sign_flip": self.p_sign_flip,
            "chosen_C": self.chosen_c,
        }


def discriminate(
    features: FeatureSet,
    labels: np.ndarray,
    sibling_ids: np.ndarray,
    groups: np.ndarray,
    *,
    n_splits: int = 5,
    seed: int = SEED,
    bootstrap_samples: int = 2000,
    residualize_on: np.ndarray | None = None,
) -> DiscriminationResult:
    """Score one feature set by paired margins inside quality-matched siblings."""
    scores, chosen = out_of_fold_scores(
        features, labels, groups, n_splits=n_splits, seed=seed, residualize_on=residualize_on
    )
    margins = sibling_margins(scores, labels, sibling_ids)
    estimate, low, high = cluster_bootstrap_ci(margins, samples=bootstrap_samples, seed=seed)
    accuracy = float(np.mean([value > 0 for value in margins.values()])) if margins else float("nan")
    return DiscriminationResult(
        name=features.name,
        n_groups=len(margins),
        forced_choice_accuracy=accuracy,
        mean_margin=estimate,
        margin_ci95=(low, high),
        p_sign_flip=sign_flip_test(margins, samples=max(bootstrap_samples, 1000), seed=seed),
        chosen_c=sorted({float(value) for value in chosen}),
    )


def quality_matched_sibling_discrimination(
    trace: AtlasTrace,
    *,
    site: str,
    layer_index: int,
    slot: int,
    texts: Sequence[str] | None = None,
    group_key: str = "item_ids",
    n_splits: int = 5,
    seed: int = SEED,
    bootstrap_samples: int = 2000,
) -> dict[str, Any]:
    """Activation discrimination minus the best control, inside matched siblings."""
    view = analysis_view(trace, slot, group_key=group_key, texts=texts)
    activation = activation_features(trace, site, layer_index, slot).take(view.keep)
    quality = trace["quality"].astype(np.float32)[view.keep]
    residual_on = np.nan_to_num(quality).reshape(-1, 1) if np.isfinite(quality).any() else None
    scoring: dict[str, Any] = {"n_splits": n_splits, "seed": seed, "bootstrap_samples": bootstrap_samples}

    result = discriminate(activation, view.labels, view.siblings, view.groups, **scoring)
    residualized = (
        discriminate(activation, view.labels, view.siblings, view.groups, residualize_on=residual_on, **scoring)
        if residual_on is not None
        else None
    )
    controls = [
        discriminate(control.take(view.keep), view.labels, view.siblings, view.groups, **scoring)
        for control in control_features(trace, slot, texts)
    ]
    best_control = max(controls, key=lambda item: item.forced_choice_accuracy) if controls else None
    selectivity = result.forced_choice_accuracy - best_control.forced_choice_accuracy if best_control else float("nan")
    variation = within_sibling_variation(activation.data, view.siblings)
    return {
        "site": site,
        "layer_index": layer_index,
        "role": trace.roles[slot],
        "slot_coverage": view.coverage,
        "within_sibling_variation": variation,
        "position_shared_by_siblings": bool(variation == 0.0),
        "activation": result.as_dict(),
        "activation_quality_residualized": residualized.as_dict() if residualized else None,
        "controls": [control.as_dict() for control in controls],
        "best_control": best_control.name if best_control else None,
        "selectivity_over_best_control": selectivity,
        "claim_limit": (
            "position shared by both siblings: the model has not read either response here, so zero "
            "discrimination says nothing about the construct"
            if variation == 0.0
            else "paired discrimination inside quality-matched siblings; association only, "
            "no causal role without an intervention"
        ),
    }


def named_versus_enacted(
    trace: AtlasTrace,
    *,
    site: str,
    layer_index: int,
    slot: int,
    group_key: str = "item_ids",
    n_splits: int = 5,
    seed: int = SEED,
    bootstrap_samples: int = 2000,
) -> dict[str, Any]:
    """Does the representation follow the strategy a response names, or the one it enacts?

    The probe is fitted on congruent records, where the two labels agree and so
    give identical training signal. Mismatch records then force the hypotheses
    apart: there the labels are complements, and the probe can agree with
    exactly one. Agreement with naming is therefore one minus agreement with
    enactment, not an independent measurement, and is reported for readability.

    Separately, each label is scored for plain decodability across all
    annotated records, which answers a different question: naming can be easy
    to decode while enactment is what the probe follows under conflict.

    The mismatch probe uses a fixed penalty rather than a nested search: there
    are too few mismatch cells to tune on, and tuning on them is what the whole
    held-out construction is meant to avoid. Whether a mismatch row shares an
    item group with a training row is measured and reported, because in a sibling
    design the probe can have been fitted on that row's twin.
    """
    named = trace["named_labels"].astype(int)
    enacted = trace["enacted_labels"].astype(int)
    known = (named >= 0) & (enacted >= 0) & trace.valid(slot)
    if known.sum() == 0:
        raise ValueError("this trace carries no named_label/enacted_label annotations")
    congruent = known & (named == enacted)
    mismatch = known & (named != enacted)
    if mismatch.sum() == 0:
        raise ValueError("no named/enacted mismatch records; the contrast is undefined")
    if congruent.sum() == 0:
        raise ValueError("no congruent records to fit on")
    if len(np.unique(enacted[congruent])) < 2:
        raise ValueError("every congruent record carries the same enacted label, so no probe can be fitted")

    features = activation_features(trace, site, layer_index, slot)
    groups = trace[group_key]
    congruent_index = np.flatnonzero(congruent)
    mismatch_index = np.flatnonzero(mismatch)
    trained_groups = set(np.asarray(groups)[congruent_index].tolist())
    shared_group = np.isin(np.asarray(groups)[mismatch_index], list(trained_groups))
    fitted = _fit_full(features, enacted, congruent_index)
    predictions = fitted.predict(features.subset(mismatch_index))
    agrees_enacted = float(np.mean(predictions == enacted[mismatch_index]))
    agrees_named = float(np.mean(predictions == named[mismatch_index]))
    clusters = _cluster_means(
        (predictions == enacted[mismatch_index]).astype(float), np.asarray(groups)[mismatch_index]
    )
    estimate, low, high = cluster_bootstrap_ci(clusters, samples=bootstrap_samples, seed=seed)
    report: dict[str, Any] = {
        "site": site,
        "layer_index": layer_index,
        "role": trace.roles[slot],
        "n_congruent": int(congruent.sum()),
        "n_mismatch": int(mismatch.sum()),
        "mismatch_rows_sharing_a_group_with_training": int(shared_group.sum()),
        "congruent_trained": {
            "congruent_cross_validated_balanced_accuracy": _grouped_cv_accuracy(
                features, enacted, groups, congruent_index, n_splits=n_splits, seed=seed
            ),
            "mismatch_agreement_with_enacted": agrees_enacted,
            "mismatch_agreement_with_named": agrees_named,
            "enactment_preference": agrees_enacted - agrees_named,
            "enacted_agreement_estimate": estimate,
            "enacted_agreement_ci95": [low, high],
        },
        "decodability": {
            target_name: _grouped_cv_accuracy(
                features, target, groups, np.flatnonzero(known), n_splits=n_splits, seed=seed
            )
            for target_name, target in (("enacted", enacted), ("named", named))
        },
    }
    report["reads_enactment_not_naming"] = bool(agrees_enacted > agrees_named and low > 0.5)
    report["claim_limit"] = (
        "mismatch agreement is measured on rows the probe never saw, but the mismatch cells are "
        "themselves a constructed contrast and can carry their own lexical signature"
    )
    if shared_group.any():
        report["claim_limit"] += (
            f"; and {int(shared_group.sum())} of {int(mismatch.sum())} mismatch rows sit in an item group the "
            "probe was fitted on, so it has seen a sibling of those rows"
        )
    return report


def _cluster_means(values: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, group in zip(values, groups):
        grouped[str(group)].append(float(value))
    return {group: float(np.mean(items)) for group, items in grouped.items()}


def _fit_full(features: FeatureSet, labels: np.ndarray, index: np.ndarray):
    return _estimator(features.kind, 1.0).fit(features.subset(index), np.asarray(labels)[index].astype(int))


def _grouped_cv_accuracy(
    features: FeatureSet, labels: np.ndarray, groups: np.ndarray, index: np.ndarray, *, n_splits: int, seed: int
) -> float:
    subset_labels = np.asarray(labels)[index].astype(int)
    subset_groups = np.asarray(groups)[index]
    if len(np.unique(subset_labels)) < 2 or len(np.unique(subset_groups)) < 2:
        return float("nan")
    scores = []
    for train, test in grouped_folds(subset_groups, subset_labels, n_splits=n_splits, seed=seed):
        model = _estimator(features.kind, 1.0).fit(features.subset(index[train]), subset_labels[train])
        scores.append(float(balanced_accuracy_score(subset_labels[test], model.predict(features.subset(index[test])))))
    return float(np.mean(scores))


def layer_probe_sweep(
    trace: AtlasTrace,
    *,
    sites: Sequence[str] | None = None,
    roles: Sequence[str] | None = None,
    group_key: str = "item_ids",
    n_splits: int = 5,
    seed: int = SEED,
    bootstrap_samples: int = 1000,
) -> list[dict[str, Any]]:
    """Every site, layer, and position role, scored the same way."""
    sites = [site for site in (sites or trace.sites) if site in trace]
    roles = list(roles or trace.roles)
    rows = []
    for role in roles:
        slot = trace.slot(role)
        try:
            view = analysis_view(trace, slot, group_key=group_key)
        except ValueError:
            continue
        for site in sites:
            for layer_index in trace.layers:
                result = discriminate(
                    activation_features(trace, site, layer_index, slot).take(view.keep),
                    view.labels,
                    view.siblings,
                    view.groups,
                    n_splits=n_splits,
                    seed=seed,
                    bootstrap_samples=bootstrap_samples,
                )
                rows.append(
                    {
                        "site": site,
                        "layer_index": layer_index,
                        "role": role,
                        "slot_coverage": view.coverage,
                        **result.as_dict(),
                    }
                )
    adjusted = benjamini_hochberg([row["p_sign_flip"] for row in rows])
    for row, value in zip(rows, adjusted):
        row["p_sign_flip_bh"] = value
    return rows


def routing_discrimination(
    trace: AtlasTrace,
    *,
    slot: int,
    texts: Sequence[str] | None = None,
    group_key: str = "item_ids",
    n_splits: int = 5,
    seed: int = SEED,
    bootstrap_samples: int = 2000,
) -> dict[str, Any]:
    """Routing-path discrimination against the token controls it must beat."""
    require_routing(trace)
    view = analysis_view(trace, slot, group_key=group_key, texts=texts)
    scoring: dict[str, Any] = {"n_splits": n_splits, "seed": seed, "bootstrap_samples": bootstrap_samples}
    routing = [
        discriminate(features.take(view.keep), view.labels, view.siblings, view.groups, **scoring)
        for features in routing_feature_sets(trace, slot)
    ]
    controls = [
        discriminate(control.take(view.keep), view.labels, view.siblings, view.groups, **scoring)
        for control in control_features(trace, slot, texts)
    ]
    best_routing = max(routing, key=lambda item: item.forced_choice_accuracy)
    best_control = max(controls, key=lambda item: item.forced_choice_accuracy) if controls else None
    return {
        "role": trace.roles[slot],
        "slot_coverage": view.coverage,
        "routing": [item.as_dict() for item in routing],
        "controls": [item.as_dict() for item in controls],
        "path_association": path_label_association(trace, slot, trace["labels"].astype(int), seed=seed),
        "expert_usage_entropy_by_layer": _usage_entropy(expert_usage(trace, [slot])).tolist(),
        "selectivity_over_best_control": (
            best_routing.forced_choice_accuracy - best_control.forced_choice_accuracy if best_control else float("nan")
        ),
        "claim_limit": (
            "routing tracks the token at the position as much as the construct; a routing result that "
            "the token-identity control matches is a tokenization result"
        ),
    }


def _usage_entropy(usage: np.ndarray) -> np.ndarray:
    safe = np.clip(usage, 1e-12, None)
    return -(safe * np.log(safe)).sum(axis=-1) / np.log(usage.shape[-1])


def read_texts(path: Path | None, ids: np.ndarray) -> list[str] | None:
    """Align response text to trace order for the lexical control."""
    if path is None:
        return None
    lookup: dict[str, str] = {}
    with Path(path).open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = str(row.get("record_id", row.get("variant_id", row.get("id", ""))))
            text = row.get("response", row.get("tutor_turn", row.get("completion", "")))
            if key:
                lookup[key] = str(text)
    missing = [str(value) for value in ids if str(value) not in lookup]
    if missing:
        raise ValueError(f"{len(missing)} traced records have no text for the lexical control, first: {missing[:3]}")
    return [lookup[str(value)] for value in ids]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--rows", type=Path, default=None, help="jsonl with response text for the lexical control")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--site", default="mlp_in", choices=SITES)
    # content_last is the only role guaranteed to have read the whole response.
    # prompt_last is shared by quality-matched siblings and cannot separate them.
    parser.add_argument("--role", default="content_last")
    parser.add_argument("--layer", type=int, default=-1, help="traced layer index; -1 uses the deepest traced layer")
    parser.add_argument("--group-key", default="item_ids")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--sweep", action="store_true", help="also score every site, layer, and role")
    args = parser.parse_args()

    trace = load_atlas_trace(args.trace)
    layer_index = trace.layers[-1] if args.layer < 0 else args.layer
    slot = trace.slot(args.role)
    texts = read_texts(args.rows, trace["ids"])
    shared = {
        "group_key": args.group_key,
        "n_splits": args.folds,
        "seed": args.seed,
        "bootstrap_samples": args.bootstrap_samples,
    }

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "trace": str(args.trace),
        "trace_metadata": trace.metadata,
        "sibling_discrimination": quality_matched_sibling_discrimination(
            trace, site=args.site, layer_index=layer_index, slot=slot, texts=texts, **shared
        ),
    }
    try:
        report["named_versus_enacted"] = named_versus_enacted(
            trace, site=args.site, layer_index=layer_index, slot=slot, **shared
        )
    except ValueError as exc:
        report["named_versus_enacted"] = {"unavailable": str(exc)}
    if trace.has_routing:
        report["routing"] = routing_discrimination(trace, slot=slot, texts=texts, **shared)
    if args.sweep:
        report["layer_sweep"] = layer_probe_sweep(trace, group_key=args.group_key, n_splits=args.folds, seed=args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=float) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
