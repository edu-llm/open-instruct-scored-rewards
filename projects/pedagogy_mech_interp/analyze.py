"""Cluster-aware statistics and claim-gating report generation."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np


def paired_differences(
    values: Iterable[float], labels: Iterable[int], pair_ids: Iterable[str]
) -> dict[str, float]:
    grouped: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for value, label, pair_id in zip(values, labels, pair_ids):
        grouped[str(pair_id)][int(label)].append(float(value))
    return {
        pair_id: float(np.mean(by_label[1]) - np.mean(by_label[0]))
        for pair_id, by_label in grouped.items()
        if by_label[0] and by_label[1]
    }


def bootstrap_ci(
    clustered_values: dict[str, float],
    *,
    statistic: Callable[[np.ndarray], float] = np.mean,
    confidence: float = 0.95,
    samples: int = 5000,
    seed: int = 0,
) -> tuple[float, float, float]:
    values = np.asarray(list(clustered_values.values()), dtype=float)
    if not len(values):
        return float("nan"), float("nan"), float("nan")
    estimate = float(statistic(values))
    rng = np.random.default_rng(seed)
    draws = np.asarray([statistic(rng.choice(values, size=len(values), replace=True)) for _ in range(samples)])
    alpha = (1.0 - confidence) / 2
    low, high = np.quantile(draws, [alpha, 1 - alpha])
    return estimate, float(low), float(high)


def sign_flip_permutation_test(clustered_values: dict[str, float], samples: int = 10000, seed: int = 0) -> float:
    values = np.asarray(list(clustered_values.values()), dtype=float)
    if not len(values):
        return float("nan")
    observed = abs(float(values.mean()))
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(samples):
        signs = rng.choice((-1.0, 1.0), size=len(values))
        exceed += abs(float((values * signs).mean())) >= observed
    return (exceed + 1) / (samples + 1)


def standardized_paired_effect(clustered_values: dict[str, float]) -> float:
    values = np.asarray(list(clustered_values.values()), dtype=float)
    if len(values) < 2 or values.std(ddof=1) == 0:
        return float("nan")
    return float(values.mean() / values.std(ddof=1))


def cluster_rows(rows: list[dict], value_key: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("item_id", row["pair_id"]))].append(float(row[value_key]))
    return {item_id: float(np.mean(values)) for item_id, values in grouped.items()}


def balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    scores = []
    for label in (0, 1):
        mask = labels == label
        if mask.any():
            scores.append(float((predictions[mask] == label).mean()))
    return float(np.mean(scores)) if scores else float("nan")


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 1.0
    for rank_index in range(len(order) - 1, -1, -1):
        original_index = int(order[rank_index])
        rank = rank_index + 1
        running = min(running, p_values[original_index] * len(order) / rank)
        adjusted[original_index] = running
    return adjusted.tolist()


def holm(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    total = len(order)
    for rank_index, original_index in enumerate(order):
        running = max(running, (total - rank_index) * p_values[int(original_index)])
        adjusted[int(original_index)] = min(1.0, running)
    return adjusted.tolist()


def repair_fraction(clean: float, corrupted: float, patched: float, floor: float = 1e-4) -> float | None:
    denominator = clean - corrupted
    if abs(denominator) < floor:
        return None
    return (patched - corrupted) / denominator


def monotonic_in_expected_direction(doses: list[float], effects: list[float], expected_direction: int = 1) -> bool:
    if len(doses) != len(effects) or len(doses) < 3:
        return False
    ordered = [effect for _, effect in sorted(zip(doses, effects))]
    differences = np.diff(ordered) * expected_direction
    return bool(np.all(differences >= -1e-8))


def claim_status(
    *,
    behavior_pass: bool,
    decodability_pass: bool,
    association_pass: bool,
    necessity_pass: bool,
    rescue_pass: bool,
    specificity_pass: bool,
) -> str:
    if not behavior_pass:
        return "failed_behavior"
    if not decodability_pass:
        return "behavior_only"
    if necessity_pass and rescue_pass and specificity_pass:
        return "causally_relevant"
    if association_pass:
        return "controllable" if specificity_pass else "associated_only"
    return "decodable"


def summarize_interventions(rows: list[dict], bootstrap_samples: int, seed: int) -> dict:
    def optional(value: float) -> float | None:
        return float(value) if np.isfinite(value) else None

    by_key: dict[tuple[str, str, str, float | None], list[dict]] = defaultdict(list)
    for row in rows:
        by_key[(row["model"], row["concept"], row["intervention"], row.get("dose"))].append(row)
    summaries = []
    for (model, concept, intervention, dose), group in sorted(
        by_key.items(), key=lambda item: tuple(str(value) for value in item[0])
    ):
        effects = cluster_rows(group, "effect")
        estimate, low, high = bootstrap_ci(effects, samples=bootstrap_samples, seed=seed)
        summaries.append(
            {
                "model": model,
                "concept": concept,
                "intervention": intervention,
                "dose": dose,
                "n_items": len(effects),
                "n_pairs": len({str(row["pair_id"]) for row in group}),
                "mean_effect": optional(estimate),
                "ci95": [optional(low), optional(high)],
                "standardized_paired_effect": optional(standardized_paired_effect(effects)),
                "p_sign_flip": optional(
                    sign_flip_permutation_test(effects, samples=bootstrap_samples, seed=seed)
                ),
            }
        )
    return {"summaries": summaries}


def intervention_interval(by_intervention, name, dose, bootstrap_samples, seed):
    group = by_intervention.get((name, dose), [])
    values = cluster_rows(group, "effect")
    return bootstrap_ci(values, samples=bootstrap_samples, seed=seed)


def gate_claims(rows: list[dict], discovery: dict, bootstrap_samples: int, seed: int) -> dict:
    def optional(value: float) -> float | None:
        return float(value) if np.isfinite(value) else None

    models = sorted({row["model"] for row in rows} | {discovery.get("model", "")})
    output = {}
    for model in models:
        if not model:
            continue
        output[model] = {}
        for concept, discovered in discovery.get("constructs", {}).items():
            subset = [row for row in rows if row["model"] == model and row["concept"] == concept]
            by_intervention = defaultdict(list)
            for row in subset:
                by_intervention[(row["intervention"], row.get("dose"))].append(row)

            _, ablation_low, ablation_high = intervention_interval(
                by_intervention, "direction_ablation", None, bootstrap_samples, seed
            )
            _, rescue_low, _ = intervention_interval(
                by_intervention, "direction_centroid_rescue", None, bootstrap_samples, seed
            )
            steer_estimate, steer_low, _ = intervention_interval(
                by_intervention, "direction_steer", 2.0, bootstrap_samples, seed
            )
            random_estimate, _, random_high = intervention_interval(
                by_intervention, "norm_matched_random_steer", 2.0, bootstrap_samples, seed
            )
            steer_rows = by_intervention.get(("direction_steer", 2.0), [])
            coherence_changes = [
                float(row["coherence_choice_change"])
                for row in steer_rows
                if row.get("coherence_choice_change") is not None
            ]
            humanlikeness_changes = [
                float(row["humanlikeness_choice_change"])
                for row in steer_rows
                if row.get("humanlikeness_choice_change") is not None
            ]
            letter_bias_changes = [
                abs(float(row["changed_letter_bias"]) - float(row["baseline_letter_bias"]))
                for row in steer_rows
            ]
            off_target_pass = bool(
                (not coherence_changes or np.mean(coherence_changes) >= -0.2)
                and (not humanlikeness_changes or np.mean(humanlikeness_changes) >= -0.2)
                and (not letter_bias_changes or np.mean(letter_bias_changes) <= 0.5)
            )
            router = discovered.get("router_candidate")
            router_ci = router.get("ci95", [float("nan"), float("nan")]) if router else [float("nan")] * 2
            exploratory_association = bool(router and (router_ci[0] > 0 or router_ci[1] < 0))
            association_pass = False
            necessity_pass = bool(ablation_high < 0)
            rescue_pass = bool(rescue_low > 0)
            specificity_pass = bool(
                steer_low > 0
                and steer_estimate > random_estimate
                and random_high < steer_estimate
                and off_target_pass
            )
            forced_choice_rows = {}
            for row in subset:
                forced_choice_rows.setdefault(str(row["pair_id"]), row)
            forced_choice = cluster_rows(list(forced_choice_rows.values()), "baseline")
            behavior_estimate, behavior_low, behavior_high = bootstrap_ci(
                forced_choice, samples=bootstrap_samples, seed=seed
            )
            behavior_accuracy = (
                float(np.mean(np.asarray(list(forced_choice.values())) > 0))
                if forced_choice
                else float("nan")
            )
            exploratory_behavior_pass = bool(behavior_low > 0 and behavior_accuracy >= 0.60)
            labels_confirmatory = bool(discovered.get("labels_confirmatory", False))
            behavior_pass = bool(exploratory_behavior_pass and labels_confirmatory)
            exploratory_decodability_pass = bool(discovered.get("decodability_pass"))
            decodability_pass = bool(discovered.get("confirmatory_decodability_pass", False))
            status = claim_status(
                behavior_pass=behavior_pass,
                decodability_pass=decodability_pass,
                association_pass=association_pass,
                necessity_pass=necessity_pass,
                rescue_pass=rescue_pass,
                specificity_pass=specificity_pass,
            )
            if not labels_confirmatory:
                exploratory = claim_status(
                    behavior_pass=exploratory_behavior_pass,
                    decodability_pass=exploratory_decodability_pass,
                    association_pass=exploratory_association,
                    necessity_pass=necessity_pass,
                    rescue_pass=rescue_pass,
                    specificity_pass=specificity_pass,
                )
                status = f"exploratory_{exploratory}"
            output[model][concept] = {
                "status": status,
                "labels_confirmatory": labels_confirmatory,
                "behavior_pass": behavior_pass,
                "exploratory_behavior_pass": exploratory_behavior_pass,
                "behavior_forced_choice": {
                    "n_pairs": len(forced_choice),
                    "mean_signed_logit_difference": optional(behavior_estimate),
                    "ci95": [optional(behavior_low), optional(behavior_high)],
                    "accuracy": optional(behavior_accuracy),
                },
                "teacher_forced_response_likelihood_gate": discovered.get("behavior"),
                "decodability_pass": decodability_pass,
                "exploratory_decodability_pass": exploratory_decodability_pass,
                "routing_association_pass": association_pass,
                "routing_association_exploratory": exploratory_association,
                "necessity_pass": necessity_pass,
                "rescue_pass": rescue_pass,
                "specificity_pass": specificity_pass,
                "off_target_noninferiority_pass": off_target_pass,
                "effects": {
                    "direction_ablation_ci95": [optional(ablation_low), optional(ablation_high)],
                    "direction_rescue_lower95": optional(rescue_low),
                    "direction_steer_dose_2": optional(steer_estimate),
                    "random_steer_dose_2": optional(random_estimate),
                    "mean_coherence_choice_change": optional(float(np.mean(coherence_changes)))
                    if coherence_changes
                    else None,
                    "mean_humanlikeness_choice_change": optional(float(np.mean(humanlikeness_changes)))
                    if humanlikeness_changes
                    else None,
                    "mean_absolute_letter_bias_change": optional(float(np.mean(letter_bias_changes)))
                    if letter_bias_changes
                    else None,
                },
            }
    return output


def markdown_report(report: dict) -> str:
    lines = [
        "# Pedagogy mechanisms pilot report",
        "",
        "Claims are gated in order: behavior, held-out-item decodability, association, necessity, rescue, and specificity.",
        "A failed gate prevents stronger language even when a later exploratory statistic is positive.",
        "",
    ]
    for model, concepts in report.get("claim_gates", {}).items():
        lines += [f"## {model}", ""]
        for concept, values in concepts.items():
            lines += [
                f"### {concept}",
                "",
                f"Claim status: **{values['status']}**.",
                "",
                f"- construct labels confirmatory: {values['labels_confirmatory']}",
                f"- confirmatory forced-choice behavior gate: {values['behavior_pass']}",
                f"- exploratory forced-choice behavior gate: {values['exploratory_behavior_pass']}",
                f"- forced-choice accuracy: {values['behavior_forced_choice']['accuracy']}",
                f"- forced-choice mean logit difference: {values['behavior_forced_choice']['mean_signed_logit_difference']}",
                f"- held-out-item decodability gate: {values['decodability_pass']}",
                f"- exploratory decodability gate: {values['exploratory_decodability_pass']}",
                f"- routing association gate: {values['routing_association_pass']}",
                f"- exploratory routing association: {values['routing_association_exploratory']}",
                f"- necessity gate: {values['necessity_pass']}",
                f"- rescue gate: {values['rescue_pass']}",
                f"- specificity gate: {values['specificity_pass']}",
                f"- off-target non-inferiority gate: {values['off_target_noninferiority_pass']}",
                "",
            ]
    lines += [
        "## Interpretation limits",
        "",
        "- Routing frequency and probe accuracy are not causal evidence.",
        "- MRBench is public and may be contaminated.",
        "- The forced-choice endpoint measures evaluation of tutor responses, not unconstrained generation.",
        "- No result in this pilot measures delayed learner retention or transfer.",
        "- Dense and MoE effects are within-model case studies, not an architecture-controlled comparison.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interventions", type=Path, required=True, help="JSONL per-pair intervention effects")
    parser.add_argument("--discovery-report", type=Path, default=None)
    parser.add_argument(
        "--pair-rows",
        type=Path,
        default=None,
        help="optional JSONL used to restore item_id and label provenance in legacy intervention files",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1701)
    args = parser.parse_args()

    with args.interventions.open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if args.pair_rows:
        with args.pair_rows.open() as handle:
            pair_metadata = {
                row["pair_id"]: {
                    "item_id": row["item_id"],
                    "label_source": row.get("label_source", "unknown"),
                }
                for line in handle
                if line.strip()
                for row in [json.loads(line)]
            }
        rows = [{**pair_metadata.get(row["pair_id"], {}), **row} for row in rows]
    report = summarize_interventions(rows, args.bootstrap_samples, args.seed)
    if args.discovery_report:
        report["discovery"] = json.loads(args.discovery_report.read_text())
        report["claim_gates"] = gate_claims(
            rows, report["discovery"], args.bootstrap_samples, args.seed
        )
    report["schema"] = "pedagogy-mech-report/v1"
    report["seed"] = args.seed
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    args.out.with_suffix(".md").write_text(markdown_report(report))
    print(f"wrote {args.out}: {len(report['summaries'])} intervention summaries")


if __name__ == "__main__":
    main()
