"""Held-out-concept decoding and control-aware RSA for atlas NPZ traces.

These analyses are deliberately exploratory.  They ask whether a direction
learned from other constructs transfers to a construct withheld in its entirety,
and whether distances between construct effect vectors resemble the frozen
ontology after lexical, token, surface, and quality controls are considered.
They do not establish a causal representation, expert specialization, learner
enactment, or a learning outcome.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from projects.learning_science_semantic_atlas import analyze_representations as representation
from projects.learning_science_semantic_atlas.schema import Ontology, load_ontology
from projects.learning_science_semantic_atlas.trace import SITES, AtlasTrace, load_atlas_trace

SCHEMA = "semantic-atlas-geometry-report/v1"
SEED = 1701
CLAIM_LIMITS = (
    "Exploratory, agent-grounded or design-grounded association only: labels are not learner outcomes or "
    "independent causal ground truth.",
    "A held-out concept prevents that construct's rows from training the decoder, but shared templates, domains, "
    "model family, and corpus construction can still induce transferable signal.",
    "Control selectivity reduces specific measured alternatives; it does not prove the activation geometry is "
    "uniquely semantic.",
    "RSA is a correlation between distance structures. It does not identify a mechanism, a causal direction, or "
    "expert specialization.",
)


@dataclass(frozen=True)
class HeldOutFold:
    concept: str
    n_train: int
    n_test: int
    balanced_accuracy: float
    forced_choice_accuracy: float
    n_sibling_groups: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "concept": self.concept,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "balanced_accuracy": self.balanced_accuracy,
            "forced_choice_accuracy": self.forced_choice_accuracy,
            "n_sibling_groups": self.n_sibling_groups,
        }


def _estimator(kind: str, *, seed: int):
    logistic = LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=4000, random_state=seed, solver="liblinear"
    )
    if kind == "text":
        return make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=5000, sublinear_tf=True), logistic
        )
    if kind == "categorical":
        return make_pipeline(OneHotEncoder(handle_unknown="ignore"), logistic)
    return make_pipeline(StandardScaler(), logistic)


def _subset(values: Any, index: np.ndarray, kind: str) -> Any:
    if kind == "text":
        return [values[position] for position in index]
    return np.asarray(values)[index]


def _decision(model: Any, values: Any) -> np.ndarray:
    return np.asarray(model.decision_function(values), dtype=float).reshape(-1)


def held_out_concept_decoding(
    features: representation.FeatureSet,
    labels: np.ndarray,
    concepts: np.ndarray,
    sibling_ids: np.ndarray,
    *,
    valid: np.ndarray | None = None,
    seed: int = SEED,
) -> dict[str, Any]:
    """Fit on every other concept and score the concept withheld in full."""
    labels = np.asarray(labels, dtype=int)
    concepts = np.asarray(concepts).astype(str)
    siblings = np.asarray(sibling_ids).astype(str)
    keep = labels >= 0
    if valid is not None:
        keep &= np.asarray(valid, dtype=bool)
    candidate_concepts = sorted(
        concept
        for concept in np.unique(concepts[keep])
        if concept and len(np.unique(labels[keep & (concepts == concept)])) == 2
    )
    folds: list[HeldOutFold] = []
    unavailable: dict[str, str] = {}
    for concept in candidate_concepts:
        test = np.flatnonzero(keep & (concepts == concept))
        train = np.flatnonzero(keep & (concepts != concept))
        if len(np.unique(labels[train])) < 2:
            unavailable[concept] = "training rows contain only one class"
            continue
        try:
            model = _estimator(features.kind, seed=seed).fit(
                _subset(features.data, train, features.kind), labels[train]
            )
            test_values = _subset(features.data, test, features.kind)
            predictions = np.asarray(model.predict(test_values), dtype=int)
            scores = _decision(model, test_values)
        except ValueError as exc:
            unavailable[concept] = str(exc)
            continue
        margins = representation.sibling_margins(scores, labels[test], siblings[test])
        forced_choice = float(np.mean([margin > 0 for margin in margins.values()])) if margins else float("nan")
        folds.append(
            HeldOutFold(
                concept=concept,
                n_train=len(train),
                n_test=len(test),
                balanced_accuracy=float(balanced_accuracy_score(labels[test], predictions)),
                forced_choice_accuracy=forced_choice,
                n_sibling_groups=len(margins),
            )
        )
    accuracies = np.asarray([fold.balanced_accuracy for fold in folds], dtype=float)
    forced = np.asarray([fold.forced_choice_accuracy for fold in folds], dtype=float)
    return {
        "feature": features.name,
        "role": features.role,
        "n_concepts": len(folds),
        "mean_balanced_accuracy": _finite_mean(accuracies),
        "mean_forced_choice_accuracy": _finite_mean(forced),
        "folds": [fold.as_dict() for fold in folds],
        "unavailable": unavailable,
    }


def _finite_mean(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(finite.mean()) if finite.size else float("nan")


def control_aware_held_out_summary(
    trace: AtlasTrace,
    *,
    site: str,
    layer_index: int,
    slot: int,
    texts: Sequence[str] | None = None,
    seed: int = SEED,
) -> dict[str, Any]:
    labels = trace["labels"].astype(int)
    concepts = trace["constructs"].astype(str)
    siblings = trace["sibling_ids"].astype(str)
    valid = trace.valid(slot)
    activation = representation.activation_features(trace, site, layer_index, slot)
    activation_report = held_out_concept_decoding(
        activation, labels, concepts, siblings, valid=valid, seed=seed
    )
    controls = [
        held_out_concept_decoding(control, labels, concepts, siblings, valid=valid, seed=seed)
        for control in representation.control_features(trace, slot, texts)
    ]
    available_controls = [
        control for control in controls if np.isfinite(float(control["mean_balanced_accuracy"]))
    ]
    best = (
        max(available_controls, key=lambda result: float(result["mean_balanced_accuracy"]))
        if available_controls
        else None
    )
    activation_accuracy = float(activation_report["mean_balanced_accuracy"])
    selectivity = (
        activation_accuracy - float(best["mean_balanced_accuracy"])
        if best is not None and np.isfinite(activation_accuracy)
        else float("nan")
    )
    return {
        "site": site,
        "layer_index": layer_index,
        "role": trace.roles[slot],
        "activation": activation_report,
        "controls": controls,
        "best_control": best["feature"] if best else None,
        "selectivity_over_best_control": selectivity,
        "claim_limit": CLAIM_LIMITS[1],
    }


def concept_effects(
    features: np.ndarray,
    labels: np.ndarray,
    concepts: np.ndarray,
    *,
    valid: np.ndarray | None = None,
) -> tuple[list[str], np.ndarray]:
    """Positive-minus-negative centroid for every concept with both classes."""
    values = np.asarray(features, dtype=float)
    values = values.reshape(len(values), -1)
    labels = np.asarray(labels, dtype=int)
    concepts = np.asarray(concepts).astype(str)
    keep = labels >= 0
    if valid is not None:
        keep &= np.asarray(valid, dtype=bool)
    names: list[str] = []
    effects: list[np.ndarray] = []
    for concept in sorted(np.unique(concepts[keep])):
        positive = keep & (concepts == concept) & (labels == 1)
        negative = keep & (concepts == concept) & (labels == 0)
        if not positive.any() or not negative.any():
            continue
        names.append(concept)
        effects.append(values[positive].mean(axis=0) - values[negative].mean(axis=0))
    width = values.shape[1]
    return names, np.stack(effects) if effects else np.empty((0, width), dtype=float)


def cosine_distance_matrix(vectors: np.ndarray) -> np.ndarray:
    values = np.asarray(vectors, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"vectors must be two-dimensional, got shape {values.shape}")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    unit = np.divide(values, norms, out=np.zeros_like(values), where=norms > 0)
    similarity = np.clip(unit @ unit.T, -1.0, 1.0)
    distance = 1.0 - similarity
    np.fill_diagonal(distance, 0.0)
    return distance


def ontology_distance_matrix(concepts: Sequence[str], ontology: Ontology) -> np.ndarray:
    """Ordinal hypothesis: sibling < same family < different family."""
    names = list(concepts)
    distance = np.ones((len(names), len(names)), dtype=float)
    for left_index, left_name in enumerate(names):
        left = ontology.get(left_name)
        for right_index, right_name in enumerate(names):
            if left_index == right_index:
                distance[left_index, right_index] = 0.0
                continue
            right = ontology.get(right_name)
            if left.sibling_key == right_name or right.sibling_key == left_name:
                distance[left_index, right_index] = 0.25
            elif left.family is right.family:
                distance[left_index, right_index] = 0.5
    return distance


def _triangle(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError(f"distance matrix must be square, got shape {values.shape}")
    return values[np.triu_indices(values.shape[0], k=1)]


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks with ties, implemented here to keep scipy optional."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return ranks


def rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    x = _rankdata(np.asarray(left, dtype=float))
    y = _rankdata(np.asarray(right, dtype=float))
    if len(x) != len(y):
        raise ValueError(f"rank vectors differ in length: {len(x)} and {len(y)}")
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def partial_rank_correlation(target: np.ndarray, hypothesis: np.ndarray, controls: Sequence[np.ndarray]) -> float:
    """Spearman correlation after linear residualization of ranked controls."""
    target_rank = _rankdata(np.asarray(target, dtype=float))
    hypothesis_rank = _rankdata(np.asarray(hypothesis, dtype=float))
    if not controls:
        return rank_correlation(target, hypothesis)
    ranked_controls = np.column_stack([_rankdata(np.asarray(control, dtype=float)) for control in controls])
    design = np.column_stack([np.ones(len(target_rank)), ranked_controls])
    target_residual = target_rank - design @ np.linalg.lstsq(design, target_rank, rcond=None)[0]
    hypothesis_residual = hypothesis_rank - design @ np.linalg.lstsq(design, hypothesis_rank, rcond=None)[0]
    if np.std(target_residual) == 0 or np.std(hypothesis_residual) == 0:
        return float("nan")
    return float(np.corrcoef(target_residual, hypothesis_residual)[0, 1])


def _permutation_p_value(
    activation_rdm: np.ndarray,
    hypothesis_rdm: np.ndarray,
    control_rdms: Sequence[np.ndarray],
    *,
    permutations: int,
    seed: int,
) -> float:
    observed = partial_rank_correlation(_triangle(activation_rdm), _triangle(hypothesis_rdm), control_rdms)
    if not np.isfinite(observed) or permutations <= 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(permutations):
        order = rng.permutation(len(hypothesis_rdm))
        shuffled = hypothesis_rdm[order][:, order]
        value = partial_rank_correlation(_triangle(activation_rdm), _triangle(shuffled), control_rdms)
        exceed += bool(np.isfinite(value) and abs(value) >= abs(observed))
    return (exceed + 1) / (permutations + 1)


def representational_similarity_analysis(
    trace: AtlasTrace,
    ontology: Ontology,
    *,
    site: str,
    layer_index: int,
    slot: int,
    permutations: int = 1000,
    seed: int = SEED,
) -> dict[str, Any]:
    """Compare construct effect geometry with ontology geometry and controls."""
    labels = trace["labels"].astype(int)
    concepts = trace["constructs"].astype(str)
    valid = trace.valid(slot)
    names, effects = concept_effects(trace.features(site, layer_index, slot), labels, concepts, valid=valid)
    names = [name for name in names if name in ontology]
    if len(names) < 3:
        raise ValueError(f"RSA needs at least three ontology concepts with both labels; found {len(names)}")

    # Recompute after excluding controls/open-pool names so all matrices align.
    ontology_rows = valid & np.isin(concepts, names)
    names, effects = concept_effects(
        trace.features(site, layer_index, slot), labels, concepts, valid=ontology_rows
    )
    activation_rdm = cosine_distance_matrix(effects)
    hypothesis_rdm = ontology_distance_matrix(names, ontology)

    control_sources: list[tuple[str, np.ndarray]] = [
        ("surface", trace["surface_features"]),
        ("token_histogram", trace["token_histogram"]),
    ]
    quality = trace["quality"].astype(float)
    if np.isfinite(quality).any():
        control_sources.append(("global_quality", np.nan_to_num(quality).reshape(-1, 1)))
    control_reports: list[dict[str, Any]] = []
    control_vectors: list[np.ndarray] = []
    for name, values in control_sources:
        control_names, control_effect = concept_effects(values, labels, concepts, valid=ontology_rows)
        if control_names != names:
            raise ValueError(f"{name} concept ordering does not align with activation geometry")
        rdm = cosine_distance_matrix(control_effect)
        triangular = _triangle(rdm)
        control_vectors.append(triangular)
        control_reports.append(
            {
                "name": name,
                "spearman_with_ontology": rank_correlation(triangular, _triangle(hypothesis_rdm)),
            }
        )

    raw = rank_correlation(_triangle(activation_rdm), _triangle(hypothesis_rdm))
    partial = partial_rank_correlation(
        _triangle(activation_rdm), _triangle(hypothesis_rdm), control_vectors
    )
    p_value = _permutation_p_value(
        activation_rdm,
        hypothesis_rdm,
        control_vectors,
        permutations=permutations,
        seed=seed,
    )
    return {
        "site": site,
        "layer_index": layer_index,
        "role": trace.roles[slot],
        "concepts": names,
        "n_concepts": len(names),
        "distance": "cosine distance between positive-minus-negative concept centroids",
        "ontology_hypothesis": "siblings=0.25, same family=0.5, different family=1.0",
        "spearman_with_ontology": raw,
        "partial_spearman_controlling_surface_token_quality": partial,
        "permutation_p_two_sided": p_value,
        "permutations": permutations,
        "controls": control_reports,
        "activation_rdm": activation_rdm.tolist(),
        "ontology_rdm": hypothesis_rdm.tolist(),
        "claim_limit": CLAIM_LIMITS[3],
    }


def _row_metadata(path: Path | None, ids: np.ndarray) -> tuple[list[str] | None, list[str]]:
    if path is None:
        return None, []
    lookup: dict[str, Mapping[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError(f"{path}:{number}: expected a JSON object")
            record_id = str(row.get("record_id", row.get("variant_id", row.get("id", ""))))
            if record_id:
                lookup[record_id] = row
    missing = [str(record_id) for record_id in ids if str(record_id) not in lookup]
    if missing:
        raise ValueError(f"{len(missing)} traced records have no row metadata, first: {missing[:3]}")
    ordered = [lookup[str(record_id)] for record_id in ids]
    texts = [str(row.get("response", row.get("tutor_turn", row.get("completion", "")))) for row in ordered]
    label_sources = sorted({str(row.get("label_source", "unspecified")) for row in ordered})
    return texts, label_sources


def analyze_trace(
    trace: AtlasTrace,
    ontology: Ontology,
    *,
    model_tag: str,
    site: str = "mlp_in",
    layer_index: int | None = None,
    role: str = "content_last",
    texts: Sequence[str] | None = None,
    label_sources: Sequence[str] = (),
    permutations: int = 1000,
    seed: int = SEED,
) -> dict[str, Any]:
    if not model_tag.strip():
        raise ValueError("model_tag must be explicit and non-empty")
    chosen_layer = trace.layers[-1] if layer_index is None or layer_index < 0 else layer_index
    slot = trace.slot(role)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "model_tag": model_tag,
        "trace_metadata": trace.metadata,
        "label_sources": list(label_sources) or ["unspecified"],
        "analysis_status": "exploratory",
        "claim_limits": list(CLAIM_LIMITS),
        "held_out_concept": control_aware_held_out_summary(
            trace, site=site, layer_index=chosen_layer, slot=slot, texts=texts, seed=seed
        ),
    }
    try:
        report["rsa"] = representational_similarity_analysis(
            trace,
            ontology,
            site=site,
            layer_index=chosen_layer,
            slot=slot,
            permutations=permutations,
            seed=seed,
        )
    except ValueError as exc:
        report["rsa"] = {"unavailable": str(exc), "claim_limit": CLAIM_LIMITS[3]}
    return report


def _trace_specs(trace_values: Sequence[str], model_tags: Sequence[str]) -> list[tuple[str, Path]]:
    specs: list[tuple[str, Path]] = []
    plain: list[Path] = []
    for value in trace_values:
        if "=" in value:
            tag, path = value.split("=", 1)
            if not tag or not path:
                raise ValueError(f"invalid --trace {value!r}; use MODEL_TAG=PATH")
            specs.append((tag, Path(path)))
        else:
            plain.append(Path(value))
    if plain:
        if specs:
            raise ValueError("do not mix MODEL_TAG=PATH traces with plain paths")
        if len(model_tags) != len(plain):
            raise ValueError("plain --trace paths require one explicit --model-tag each")
        specs = list(zip(model_tags, plain))
    elif model_tags:
        raise ValueError("--model-tag is only used with plain --trace paths")
    if len({tag for tag, _ in specs}) != len(specs):
        raise ValueError("model tags must be unique")
    return specs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--trace",
        action="append",
        required=True,
        help="MODEL_TAG=trace.npz, or a plain path paired with --model-tag",
    )
    parser.add_argument("--model-tag", action="append", default=[], help="explicit tag for a plain --trace path")
    parser.add_argument("--rows", action="append", type=Path, default=[], help="aligned tracer-ready JSONL per trace")
    parser.add_argument("--ontology", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--site", default="mlp_in", choices=SITES)
    parser.add_argument("--role", default="content_last")
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--dry-run", action="store_true", help="print resolved inputs without analyzing")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    specs = _trace_specs(args.trace, args.model_tag)
    if args.rows and len(args.rows) != len(specs):
        raise SystemExit("--rows must be omitted or supplied once per trace")
    plan = {
        "schema": SCHEMA,
        "traces": [{"model_tag": tag, "path": str(path)} for tag, path in specs],
        "rows": [str(path) for path in args.rows],
        "site": args.site,
        "role": args.role,
        "layer": args.layer,
        "permutations": args.permutations,
        "output": str(args.out),
        "dry_run": args.dry_run,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return

    ontology = load_ontology(args.ontology)
    reports: dict[str, Any] = {}
    for index, (model_tag, path) in enumerate(specs):
        trace = load_atlas_trace(path)
        rows_path = args.rows[index] if args.rows else None
        texts, label_sources = _row_metadata(rows_path, trace["ids"])
        reports[model_tag] = analyze_trace(
            trace,
            ontology,
            model_tag=model_tag,
            site=args.site,
            layer_index=args.layer,
            role=args.role,
            texts=texts,
            label_sources=label_sources,
            permutations=args.permutations,
            seed=args.seed,
        )
        reports[model_tag]["trace"] = str(path)
        reports[model_tag]["rows"] = str(rows_path) if rows_path else None

    payload = {
        "schema": SCHEMA,
        "analysis_status": "exploratory",
        "claim_limits": list(CLAIM_LIMITS),
        "models": reports,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
