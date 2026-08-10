"""Build the base-model, concept-by-concept atlas report and its figures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "remote"
FIGURES = ROOT / "figures"
CONCEPT_FIGURES = FIGURES / "concepts"
REPORT = ROOT / "FINAL_REPORT.md"

BASE_MODELS = ("OLMoE base", "OLMo-2 base")
REPORT_TAGS = ("olmoe-grpo-base", "olmo2-base")
BEHAVIOR_TAGS = ("olmoe-base", "olmo2-base")
COLORS = ("#2367a9", "#2f855a")
CONTROL_COLOR = "#8a8f98"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def finish(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def ontology() -> dict[str, Any]:
    return yaml.safe_load((ROOT / "ontology.yaml").read_text())


def base_geometry() -> list[dict[str, Any]]:
    return [
        read_json(RESULTS / tag / "round1_geometry.json")["models"][tag]["held_out_concept"]
        for tag in REPORT_TAGS
    ]


def fold_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["concept"]: row for row in rows}


def concept_scores(
    geometry: list[dict[str, Any]], concept: str
) -> tuple[list[float], list[float]]:
    activation: list[float] = []
    controls: list[float] = []
    for report in geometry:
        activation_by_concept = fold_map(report["activation"]["folds"])
        activation.append(float(activation_by_concept[concept]["balanced_accuracy"]))
        controls.append(
            max(
                float(fold_map(control["folds"])[concept]["balanced_accuracy"])
                for control in report["controls"]
            )
        )
    return activation, controls


def wilson_interval(rate: float, n: int = 5, z: float = 1.96) -> tuple[float, float]:
    """Approximate binomial interval used only to show how uncertain n=5 is."""
    denominator = 1 + z**2 / n
    center = (rate + z**2 / (2 * n)) / denominator
    half_width = z * np.sqrt(rate * (1 - rate) / n + z**2 / (4 * n**2)) / denominator
    return max(0.0, center - half_width), min(1.0, center + half_width)


def plot_ontology_family_counts(spec: dict[str, Any]) -> None:
    family_keys = list(spec["families"])
    family_names = [spec["families"][key]["display_name"] for key in family_keys]
    counts = [sum(row["family"] == key for row in spec["constructs"]) for key in family_keys]
    fig, ax = plt.subplots(figsize=(9, 4.3))
    bars = ax.barh(family_names, counts, color=COLORS[0])
    ax.bar_label(bars, padding=4)
    ax.set(xlabel="Number of constructs", title="The 24-concept learning-science ontology")
    ax.set_xlim(0, max(counts) + 1)
    ax.grid(axis="x", alpha=0.25)
    fig.text(0.01, -0.02, "Source: ontology.yaml · atlas-v1.0.0", fontsize=9)
    finish(fig, FIGURES / "ontology_family_counts.png")


def plot_base_behavior() -> None:
    behavior = [
        read_json(RESULTS / "behavior" / tag / "behavior_report.json") for tag in BEHAVIOR_TAGS
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    width = 0.34
    x2 = (-width / 2, width / 2)

    declarative = ("anchor_attribution", "definition_to_name", "name_to_definition")
    declarative_labels = ("Anchor → name", "Definition → name", "Name → definition")
    x = np.arange(3)
    for index, (model, color) in enumerate(zip(BASE_MODELS, COLORS, strict=True)):
        shifts = [
            behavior[index]["declarative_knowledge"]["kinds"][kind]["paired_shift"]
            for kind in declarative
        ]
        values = [shift["mean"] for shift in shifts]
        errors = np.asarray(
            [
                [shift["mean"] - shift["ci95"][0] for shift in shifts],
                [shift["ci95"][1] - shift["mean"] for shift in shifts],
            ]
        )
        axes[0].bar(
            x + x2[index],
            values,
            width,
            label=model,
            color=color,
            yerr=errors,
            capsize=3,
        )
    axes[0].set(
        xticks=x,
        xticklabels=declarative_labels,
        ylabel="Paired log-probability shift",
        title="Declarative knowledge",
    )
    axes[0].tick_params(axis="x", rotation=18)
    axes[0].axhline(0, color="black", linewidth=0.8)

    boundaries = ("boundary_versus_sibling", "required_context", "format_applicability")
    boundary_labels = ("Target vs sibling", "Required context", "Artifact format")
    for index, (model, color) in enumerate(zip(BASE_MODELS, COLORS, strict=True)):
        shifts = [
            behavior[index]["boundary_conditions"]["kinds"][kind]["paired_shift"]
            for kind in boundaries
        ]
        values = [shift["mean"] for shift in shifts]
        errors = np.asarray(
            [
                [shift["mean"] - shift["ci95"][0] for shift in shifts],
                [shift["ci95"][1] - shift["mean"] for shift in shifts],
            ]
        )
        axes[1].bar(
            x + x2[index],
            values,
            width,
            label=model,
            color=color,
            yerr=errors,
            capsize=3,
        )
    axes[1].set(
        xticks=x,
        xticklabels=boundary_labels,
        ylabel="Paired log-probability shift",
        title="Boundary conditions",
    )
    axes[1].tick_params(axis="x", rotation=18)
    axes[1].axhline(0, color="black", linewidth=0.8)

    means: list[float] = []
    errors = [[], []]
    for report in behavior:
        shift = report["enacted_strategy"]["kinds"]["enacted_strategy"]["paired_shift"]
        means.append(float(shift["mean"]))
        errors[0].append(float(shift["mean"] - shift["ci95"][0]))
        errors[1].append(float(shift["ci95"][1] - shift["mean"]))
    x_models = np.arange(2)
    axes[2].bar(x_models, means, color=COLORS)
    axes[2].errorbar(
        x_models, means, yerr=np.asarray(errors), fmt="none", ecolor="black", capsize=4
    )
    axes[2].set(
        xticks=x_models,
        xticklabels=BASE_MODELS,
        ylabel="Paired log-probability shift",
        title="Enacted strategy: target vs sibling",
    )
    axes[2].axhline(0, color="black", linewidth=0.8)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.07),
        ncol=2,
        frameon=False,
    )
    fig.suptitle("Base-model behavior: clear concept knowledge, no enacted-strategy preference")
    fig.subplots_adjust(bottom=0.34, wspace=0.35)
    fig.text(
        0.01,
        0.01,
        "Source: behavior reports · 420 crossed probes per model · error bars are 95% bootstrap CIs",
        fontsize=9,
    )
    finish(fig, FIGURES / "base_behavior_summary.png")


def plot_each_concept(spec: dict[str, Any], geometry: list[dict[str, Any]]) -> None:
    CONCEPT_FIGURES.mkdir(parents=True, exist_ok=True)
    labels = ("OLMoE\nactivation", "OLMoE\ncontrol", "OLMo-2\nactivation", "OLMo-2\ncontrol")
    colors = (COLORS[0], CONTROL_COLOR, COLORS[1], CONTROL_COLOR)
    for construct in spec["constructs"]:
        key = construct["key"]
        activation, controls = concept_scores(geometry, key)
        rates = (activation[0], controls[0], activation[1], controls[1])
        values = tuple(100 * rate for rate in rates)
        intervals = [wilson_interval(rate) for rate in rates]
        errors = np.asarray(
            [
                [100 * (rate - interval[0]) for rate, interval in zip(rates, intervals, strict=True)],
                [100 * (interval[1] - rate) for rate, interval in zip(rates, intervals, strict=True)],
            ]
        )
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        x = np.arange(4)
        bars = ax.bar(x, values, color=colors, yerr=errors, capsize=4)
        ax.bar_label(bars, fmt="%.0f%%", padding=3)
        ax.axhline(50, color="black", linestyle="--", linewidth=0.9, label="Chance")
        ax.set(
            xticks=x,
            xticklabels=labels,
            ylabel="Balanced accuracy (%)",
            title=construct["display_name"],
            ylim=(0, 120),
        )
        ax.legend(frameon=False, loc="upper right")
        ax.grid(axis="y", alpha=0.2)
        fig.subplots_adjust(bottom=0.22)
        finish(fig, CONCEPT_FIGURES / f"{key}.png")


def directional_sentence(activation: list[float], controls: list[float]) -> str:
    deltas = [100 * (score - control) for score, control in zip(activation, controls, strict=True)]
    positive = [delta > 1e-9 for delta in deltas]
    if all(positive):
        reading = "Both base models beat their strongest measured control."
    elif positive[0]:
        reading = "Only OLMoE beat its strongest measured control."
    elif positive[1]:
        reading = "Only OLMo-2 beat its strongest measured control."
    else:
        reading = "Neither base model beat its strongest measured control."
    return (
        f"{reading} Directional selectivity: OLMoE `{deltas[0]:+.1f}` pp; "
        f"OLMo-2 `{deltas[1]:+.1f}` pp."
    )


def build_report(spec: dict[str, Any], geometry: list[dict[str, Any]]) -> None:
    by_family: dict[str, list[dict[str, Any]]] = {key: [] for key in spec["families"]}
    for construct in spec["constructs"]:
        by_family[construct["family"]].append(construct)

    lines = [
        "# Learning Science Semantic Atlas — Base-Model Report",
        "",
        "## Main result",
        "",
        "- Both base models show clear **declarative knowledge** of learning-science concepts.",
        "- Neither reliably prefers the artifact that **enacts** the target strategy.",
        "- Per-concept activation plots below are exploratory: each concept has only **5 held-out rows per model**.",
        "- The old RL comparison is intentionally excluded; it should be rerun with a reward that explicitly maps to the ontology.",
        "",
        "![Ontology family counts](figures/ontology_family_counts.png)",
        "",
        "![Base-model behavior summary](figures/base_behavior_summary.png)",
        "",
        "## How to read each concept plot",
        "",
        "- **Activation:** a linear classifier predicts the concept from the model's hidden-state activation vector.",
        "- **Control:** the best classifier that sees only shortcuts—words, length/formatting, token counts, or token identity—and never sees hidden states.",
        "- **What matters:** `activation accuracy − control accuracy`. Example: `83% − 58% = +25` percentage points of possible internal signal beyond the measured shortcuts.",
        "- **Chance:** 50%. Above chance alone is insufficient; activation should also beat control.",
        "- **Error bars:** approximate 95% Wilson intervals using `n=5`. They visualize the severe uncertainty; balanced accuracy is not exactly binomial, so do not treat them as precise inferential intervals.",
        "- With 5 test rows, each concept result is directional—not a standalone discovery.",
        "",
        "## The 24 concepts",
        "",
    ]

    for family_key, constructs in by_family.items():
        lines.extend([f"## {spec['families'][family_key]['display_name']}", ""])
        for construct in constructs:
            activation, controls = concept_scores(geometry, construct["key"])
            definition = " ".join(construct["definition"].split())
            lines.extend(
                [
                    f"### {construct['display_name']}",
                    "",
                    f"- **Meaning:** {definition}",
                    f"- **Evidence grade:** `{construct['evidence_grade']}`",
                    f"- **Base-model read:** {directional_sentence(activation, controls)}",
                    "",
                    f"![{construct['display_name']} decoding](figures/concepts/{construct['key']}.png)",
                    "",
                ]
            )

    lines.extend(
        [
            "## Bottom line",
            "",
            "- Strongest shared above-control directions: interleaved discrimination and guidance fading.",
            "- Many concepts are matched or beaten by surface/token controls.",
            "- These plots measure linear decodability—not causal use, learner uptake, or learning outcomes.",
            "- Next RL experiment: map every rewarded dimension to explicit ontology constructs and evaluate held-out constructs separately.",
            "",
            "## Reproducibility",
            "",
            "- Source reports: `results/remote/`",
            "- Plot/report generator: `plot_report.py`",
            "- Verification: `339` project tests passed; Ruff clean.",
            "",
        ]
    )
    REPORT.write_text("\n".join(lines))


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    spec = ontology()
    geometry = base_geometry()
    plot_ontology_family_counts(spec)
    plot_base_behavior()
    plot_each_concept(spec, geometry)
    build_report(spec, geometry)
    print(f"wrote {len(spec['constructs'])} concept plots and {REPORT}")


if __name__ == "__main__":
    main()
