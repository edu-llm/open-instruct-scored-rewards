"""Behavior gates, lexical controls, grouped activation probes, and candidate freeze.

Layer/site/hyperparameter selection uses discovery validation only. The natural
test is scored once after selection. The sealed Bridge confirmatory set is not
opened by this command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from projects.pedagogy_mech_interp.analyze import bootstrap_ci, paired_differences, sign_flip_permutation_test
from projects.pedagogy_mech_interp.constructs import CONSTRUCTS

ACTIVATION_SITES = ("residual_in", "residual_out", "mlp_in", "mlp_out")
CS = (0.01, 0.1, 1.0, 10.0, 100.0)


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def load_trace(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as blob:
        return {key: blob[key] for key in blob.files}


def load_text(path: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    with path.open() as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[row["variant_id"]] = row
    return rows


def concept_mask(blob: dict[str, np.ndarray], concept: str) -> np.ndarray:
    return (blob["concepts"] == concept) & (blob["labels"] >= 0)


def fit_activation_model(x: np.ndarray, y: np.ndarray, c: float):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c,
            class_weight="balanced",
            dual=True,
            max_iter=4000,
            random_state=1701,
            solver="liblinear",
        ),
    ).fit(x, y)


def choose_activation_candidate(
    train: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
    concept: str,
) -> tuple[dict[str, Any], Any]:
    train_mask = concept_mask(train, concept)
    validation_mask = concept_mask(validation, concept)
    best: dict[str, Any] | None = None
    best_model = None
    for site in ACTIVATION_SITES:
        if site not in train or site not in validation:
            continue
        for layer_position, layer_index in enumerate(train["layers"].tolist()):
            x_train = train[site][train_mask, layer_position].astype(np.float32)
            x_validation = validation[site][validation_mask, layer_position].astype(np.float32)
            y_train = train["labels"][train_mask].astype(int)
            y_validation = validation["labels"][validation_mask].astype(int)
            for c in CS:
                model = fit_activation_model(x_train, y_train, c)
                score = float(balanced_accuracy_score(y_validation, model.predict(x_validation)))
                candidate = {
                    "site": site,
                    "layer_position": layer_position,
                    "layer_index": int(layer_index),
                    "C": c,
                    "validation_balanced_accuracy": score,
                }
                if best is None or (score, -layer_position, site) > (
                    best["validation_balanced_accuracy"],
                    -best["layer_position"],
                    best["site"],
                ):
                    best, best_model = candidate, model
    if best is None:
        raise ValueError(f"no activation sites available for {concept}")
    return best, best_model


def lexical_model(
    train_rows: dict[str, dict[str, Any]],
    validation_rows: dict[str, dict[str, Any]],
    train: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
    concept: str,
):
    train_mask = concept_mask(train, concept)
    validation_mask = concept_mask(validation, concept)
    train_text = [train_rows[str(uid)]["tutor_turn"] for uid in train["ids"][train_mask]]
    validation_text = [validation_rows[str(uid)]["tutor_turn"] for uid in validation["ids"][validation_mask]]
    y_train = train["labels"][train_mask].astype(int)
    y_validation = validation["labels"][validation_mask].astype(int)
    best = None
    for c in CS:
        model = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=5000, sublinear_tf=True),
            LogisticRegression(
                C=c,
                class_weight="balanced",
                dual=True,
                max_iter=4000,
                random_state=1701,
                solver="liblinear",
            ),
        ).fit(train_text, y_train)
        score = float(balanced_accuracy_score(y_validation, model.predict(validation_text)))
        if best is None or score > best[0]:
            best = (score, c, model)
    return best


def behavior_gate(blob: dict[str, np.ndarray], concept: str, bootstrap_samples: int, seed: int) -> dict[str, Any]:
    mask = concept_mask(blob, concept)
    differences = paired_differences(
        blob["response_logprob"][mask], blob["labels"][mask], blob["pair_ids"][mask]
    )
    estimate, low, high = bootstrap_ci(differences, samples=bootstrap_samples, seed=seed)
    accuracy = float(np.mean(np.asarray(list(differences.values())) > 0)) if differences else float("nan")
    return {
        "n_pairs": len(differences),
        "mean_prescribed_minus_alternative_logprob": estimate,
        "ci95": [low, high],
        "forced_choice_accuracy": accuracy,
        "pass": bool(low > 0 and accuracy >= 0.60),
    }


def choose_router_candidate(
    validation: dict[str, np.ndarray], concept: str, permutation_samples: int, seed: int
) -> dict[str, Any] | None:
    if "router_logits" not in validation:
        return None
    mask = concept_mask(validation, concept)
    logits = validation["router_logits"][mask].astype(np.float32)
    logits -= logits.max(axis=-1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    labels = validation["labels"][mask]
    pair_ids = validation["pair_ids"][mask]
    best = None
    candidates = []
    for layer_position, layer_index in enumerate(validation["layers"].tolist()):
        for expert_index in range(probabilities.shape[-1]):
            differences = paired_differences(
                probabilities[:, layer_position, expert_index], labels, pair_ids
            )
            effect = float(np.mean(list(differences.values()))) if differences else 0.0
            candidate = {
                "layer_position": layer_position,
                "layer_index": int(layer_index),
                "expert_index": expert_index,
                "mean_positive_minus_negative_router_probability": effect,
                "absolute_effect": abs(effect),
            }
            candidates.append(candidate)
            if best is None or candidate["absolute_effect"] > best["absolute_effect"]:
                best = {**candidate, "_differences": differences}
    if best is None:
        return None
    differences = best.pop("_differences")
    same_layer = sorted(
        (candidate for candidate in candidates if candidate["layer_index"] == best["layer_index"]),
        key=lambda candidate: candidate["absolute_effect"],
        reverse=True,
    )
    best["coalition_experts"] = [int(candidate["expert_index"]) for candidate in same_layer[:3]]
    estimate, low, high = bootstrap_ci(differences, samples=permutation_samples, seed=seed)
    best["ci95"] = [low, high]
    best["p_sign_flip_unadjusted"] = sign_flip_permutation_test(
        differences, samples=permutation_samples, seed=seed
    )
    best["claim_limit"] = "exploratory routing association; intervention is required for specialization"
    return best


def neuron_atlas(
    validation: dict[str, np.ndarray],
    concept: str,
    site: str,
    layer_position: int,
    top_n: int = 20,
) -> list[dict[str, float | int]]:
    def correlation(first: np.ndarray, second: np.ndarray) -> float:
        if first.std() == 0 or second.std() == 0:
            return 0.0
        return float(np.corrcoef(first, second)[0, 1])

    mask = concept_mask(validation, concept)
    features = validation[site][mask, layer_position].astype(np.float32)
    labels = validation["labels"][mask].astype(float)
    lengths = validation["word_count"][mask].astype(float)
    pair_ids = validation["pair_ids"][mask]
    rows = []
    for neuron_index in range(features.shape[1]):
        differences = paired_differences(features[:, neuron_index], labels.astype(int), pair_ids)
        effect = float(np.mean(list(differences.values()))) if differences else 0.0
        label_correlation = correlation(features[:, neuron_index], labels)
        length_correlation = correlation(features[:, neuron_index], lengths)
        rows.append(
            {
                "neuron_index": neuron_index,
                "mean_positive_minus_negative_activation": effect,
                "label_correlation": label_correlation,
                "word_count_correlation": length_correlation,
            }
        )
    return sorted(
        rows,
        key=lambda row: abs(float(row["mean_positive_minus_negative_activation"])),
        reverse=True,
    )[:top_n]


def permutation_control(
    x_train: np.ndarray,
    y_train: np.ndarray,
    groups: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    c: float,
    repetitions: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    scores = []
    unique_groups = np.unique(groups)
    for _ in range(repetitions):
        permuted = y_train.copy()
        for group in unique_groups:
            if rng.random() < 0.5:
                mask = groups == group
                permuted[mask] = 1 - permuted[mask]
        model = fit_activation_model(x_train, permuted, c)
        scores.append(float(balanced_accuracy_score(y_test, model.predict(x_test))))
    return {
        "mean_balanced_accuracy": float(np.mean(scores)) if scores else float("nan"),
        "max_balanced_accuracy": float(np.max(scores)) if scores else float("nan"),
        "repetitions": len(scores),
    }


def raw_direction(model) -> tuple[np.ndarray, float]:
    scaler = model.named_steps["standardscaler"]
    logistic = model.named_steps["logisticregression"]
    direction = logistic.coef_[0].astype(np.float64) / scaler.scale_
    norm = np.linalg.norm(direction)
    if norm == 0:
        raise ValueError("selected probe has a zero direction")
    direction /= norm
    intercept = float(logistic.intercept_[0] - np.dot(logistic.coef_[0], scaler.mean_ / scaler.scale_))
    return direction.astype(np.float32), intercept


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-trace", type=Path, required=True)
    parser.add_argument("--validation-trace", type=Path, required=True)
    parser.add_argument("--test-trace", type=Path, required=True)
    parser.add_argument("--train-rows", type=Path, required=True)
    parser.add_argument("--validation-rows", type=Path, required=True)
    parser.add_argument("--test-rows", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=50)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1701)
    args = parser.parse_args()

    train, validation, test = map(load_trace, (args.train_trace, args.validation_trace, args.test_trace))
    if not np.array_equal(train["layers"], validation["layers"]) or not np.array_equal(train["layers"], test["layers"]):
        raise SystemExit("trace layer grids differ")
    train_rows, validation_rows, test_rows = map(load_text, (args.train_rows, args.validation_rows, args.test_rows))
    model_metadata = json.loads(str(train["metadata"]))

    report: dict[str, Any] = {
        "schema": "pedagogy-mech-discovery/v1",
        "model": model_metadata["model"],
        "traces": {
            "train": {"path": str(args.train_trace), "sha256": file_sha256(args.train_trace)},
            "validation": {"path": str(args.validation_trace), "sha256": file_sha256(args.validation_trace)},
            "test": {"path": str(args.test_trace), "sha256": file_sha256(args.test_trace)},
        },
        "constructs": {},
    }
    directions: dict[str, np.ndarray] = {}
    direction_metadata = {}

    for concept in CONSTRUCTS:
        candidate, _ = choose_activation_candidate(train, validation, concept)
        layer_position = candidate["layer_position"]
        site = candidate["site"]
        train_mask = concept_mask(train, concept)
        test_mask = concept_mask(test, concept)
        x_train = train[site][train_mask, layer_position].astype(np.float32)
        y_train = train["labels"][train_mask].astype(int)
        x_test = test[site][test_mask, layer_position].astype(np.float32)
        y_test = test["labels"][test_mask].astype(int)
        model = fit_activation_model(x_train, y_train, candidate["C"])
        predictions = model.predict(x_test)
        probabilities = model.predict_proba(x_test)[:, 1]
        activation_score = float(balanced_accuracy_score(y_test, predictions))

        lexical = lexical_model(train_rows, validation_rows, train, validation, concept)
        lexical_score, lexical_c, lexical_fit = lexical
        test_text = [test_rows[str(uid)]["tutor_turn"] for uid in test["ids"][test_mask]]
        lexical_predictions = lexical_fit.predict(test_text)
        lexical_test = float(balanced_accuracy_score(y_test, lexical_predictions))
        selectivity = activation_score - lexical_test
        correctness_delta = (predictions == y_test).astype(float) - (lexical_predictions == y_test).astype(float)
        selectivity_by_item = {
            str(item_id): float(correctness_delta[test["item_ids"][test_mask] == item_id].mean())
            for item_id in np.unique(test["item_ids"][test_mask])
        }
        _, selectivity_low, selectivity_high = bootstrap_ci(
            selectivity_by_item, samples=args.bootstrap_samples, seed=args.seed
        )
        permutation = permutation_control(
            x_train,
            y_train,
            train["pair_ids"][train_mask],
            x_test,
            y_test,
            candidate["C"],
            args.permutations,
            args.seed,
        )
        direction, intercept = raw_direction(model)
        train_projection = x_train @ direction
        positive_projection_mean = float(train_projection[y_train == 1].mean())
        negative_projection_mean = float(train_projection[y_train == 0].mean())
        projection_std = float(train_projection.std(ddof=1))
        direction_key = f"direction_{concept}"
        directions[direction_key] = direction
        direction_metadata[concept] = {
            "array": direction_key,
            "site": site,
            "layer_index": candidate["layer_index"],
            "intercept": intercept,
            "positive_projection_mean": positive_projection_mean,
            "negative_projection_mean": negative_projection_mean,
            "projection_std": projection_std,
        }

        behavior = behavior_gate(test, concept, args.bootstrap_samples, args.seed)
        router_candidate = choose_router_candidate(
            validation, concept, max(args.bootstrap_samples, 1000), args.seed
        )
        top_neurons = neuron_atlas(
            validation, concept, site, layer_position
        )
        label_sources = sorted(
            {
                test_rows[str(uid)].get("label_source", "unknown")
                for uid in test["ids"][test_mask]
            }
        )
        labels_confirmatory = set(label_sources) <= {"dataset_direct", "human_verified"}
        statistical_decode_pass = bool(
            selectivity >= 0.05
            and selectivity_low > 0
            and activation_score > permutation["max_balanced_accuracy"]
        )
        report["constructs"][concept] = {
            "behavior": behavior,
            "candidate": candidate,
            "test_balanced_accuracy": activation_score,
            "lexical_validation_balanced_accuracy": lexical_score,
            "lexical_C": lexical_c,
            "lexical_test_balanced_accuracy": lexical_test,
            "selectivity": selectivity,
            "selectivity_item_cluster_ci95": [selectivity_low, selectivity_high],
            "label_sources": label_sources,
            "labels_confirmatory": labels_confirmatory,
            "random_label_control": permutation,
            "decodability_pass": statistical_decode_pass,
            "confirmatory_decodability_pass": bool(statistical_decode_pass and labels_confirmatory),
            "router_candidate": router_candidate,
            "top_neurons_descriptive": top_neurons,
            "test_predictions": [
                {
                    "variant_id": str(uid),
                    "label": int(label),
                    "prediction": int(prediction),
                    "probability": float(probability),
                }
                for uid, label, prediction, probability in zip(
                    test["ids"][test_mask], y_test, predictions, probabilities
                )
            ],
        }

    args.out.mkdir(parents=True, exist_ok=True)
    freeze = {
        "schema": "pedagogy-mech-candidate-freeze/v1",
        "model": report["model"],
        "input_hashes": report["traces"],
        "candidates": {
            concept: {
                **direction_metadata[concept],
                "behavior_pass": values["behavior"]["pass"],
                "decodability_pass": values["decodability_pass"],
                "confirmatory_decodability_pass": values["confirmatory_decodability_pass"],
                "labels_confirmatory": values["labels_confirmatory"],
                "router_candidate": values["router_candidate"],
            }
            for concept, values in report["constructs"].items()
        },
        "sealed_confirmatory_opened": False,
    }
    (args.out / "discovery_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (args.out / "candidate_freeze.json").write_text(json.dumps(freeze, indent=2) + "\n")
    np.savez_compressed(
        args.out / "directions.npz",
        metadata=np.asarray(json.dumps(direction_metadata)),
        **directions,
    )
    print(f"wrote discovery report and frozen candidates to {args.out}")


if __name__ == "__main__":
    main()
