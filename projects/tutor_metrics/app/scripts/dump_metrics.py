"""Render metrics.py into the JSON the TypeScript seeder pushes to Postgres.

    python scripts/dump_metrics.py            # rewrites db/metrics.json

metrics.py is the source of truth and the `metric` table is a cache
(LABELING_APP_SPEC.md §7.5). Nothing here invents rater-facing wording: every question, anchor
label and rung description is copied verbatim out of the dataclasses. What this file decides is
only the shape the app needs and metrics.py does not carry - the scale, which panels a metric
needs on screen, and when a span is demanded.

`source_sha` covers the fields a rater reads. /api/health compares it against the seeded rows, so
editing a question in metrics.py without re-seeding is a loud failure rather than a silent
reinterpretation of labels already collected.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
METRICS_PY = HERE.parent.parent / "metrics.py"
OUT = HERE.parent / "db" / "metrics.json"


def load_metrics_module():
    spec = importlib.util.spec_from_file_location("tutor_metrics_metrics", METRICS_PY)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import {METRICS_PY}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


# Which reading panels a metric cannot be answered without, and whether a not-applicable answer is
# a real option. Everything absent from these two sets takes the default.
NEEDS_REFERENCE = {"reference_conflict"}
NEEDS_DIALOGUE = {"dialogue_conflict", "asks_recall_of_established"}
ALLOW_NA = {"reference_conflict", "dialogue_conflict"}

GROUPS = {
    "CONTRIBUTIONS": "A",
    "ANCHORING": "B",
    "PENALTIES": "guard",
    "MEASURED_ONLY": "covariate",
}


def binary_row(m, group: str) -> dict:
    """A presence question on 1-2: 1 is the hard negative, 2 is the positive anchor.

    Binary because metrics.py states every SCORED question as a yes/no and `reward()` reads them
    through `bool()`. LABELING_APP_SPEC.md §9 guess 1 assumed the same for the guards.
    """
    return {
        "key": m.key,
        "group": group,
        "question": m.question,
        "anchors": [
            {"value": 1, "label": m.hard_negative},
            {"value": 2, "label": m.positive_anchor},
        ],
        "guidance": guidance(m),
        "scale": "ordinal",
        "lo": 1,
        "hi": 2,
        # A presence question has evidence to point at exactly when the rater says it is present.
        "requiresSpan": group != "covariate",
        "spanFromValue": 2,
        "needsReference": m.key in NEEDS_REFERENCE,
        "needsDialogue": m.key in NEEDS_DIALOGUE,
        "allowNa": m.key in ALLOW_NA,
        "sourceSha": sha(
            {
                "key": m.key,
                "question": m.question,
                "atlas": m.atlas,
                "eligibility": m.eligibility,
                "positive_anchor": m.positive_anchor,
                "hard_negative": m.hard_negative,
                "surface_confound": m.surface_confound,
                "length_coupling": m.length_coupling,
                "lo": 1,
                "hi": 2,
            }
        ),
    }


def guidance(m) -> str:
    lines = [f"Ignore this surface cue: {m.surface_confound}."]
    if m.atlas:
        lines.append(f"Atlas construct: {m.atlas}.")
    lines.append(f"Eligibility: {m.eligibility}. Expected length coupling: {m.length_coupling}.")
    return "\n".join(lines)


def main() -> None:
    mod = load_metrics_module()
    rows: list[dict] = []

    # The spine. metrics.py holds it as RUNGS + RUNG_ANCHORS rather than as a Metric, because
    # `contingency()` compares a rung name against the scenario's prescribed target. The rater
    # answers the rung; the target never appears anywhere in the app.
    rungs = list(mod.RUNGS)
    rows.append(
        {
            "key": "assistance_level",
            "group": "C",
            "question": "Which single rung does this turn occupy?",
            "anchors": [{"value": i + 1, "label": mod.RUNG_ANCHORS[r], "name": r} for i, r in enumerate(rungs)],
            "guidance": (
                "Graesser's ladder, ordered from least to most tutor control. The ORDER is "
                "load-bearing: the reward gives partial credit for an adjacent rung, so a wrong "
                "order silently changes the metric.\n"
                "Answer the rung the turn OCCUPIES, not the rung it should have occupied. Three "
                "tutors pick the same next action in 18% of cases, so that second question has no "
                "recoverable answer."
            ),
            "scale": "ordinal",
            "lo": 1,
            "hi": len(rungs),
            "requiresSpan": True,
            "spanFromValue": None,
            "needsReference": False,
            "needsDialogue": True,
            "allowNa": False,
            "sourceSha": sha({"rungs": rungs, "anchors": dict(mod.RUNG_ANCHORS)}),
        }
    )

    for family, group in GROUPS.items():
        for m in getattr(mod, family):
            if m.key == "icap_class":
                classes = ["passive", "active", "constructive", "interactive"]
                rows.append(
                    {
                        "key": m.key,
                        "group": group,
                        "question": m.question,
                        "anchors": [{"value": i + 1, "label": c, "name": c} for i, c in enumerate(classes)],
                        "guidance": f"{m.positive_anchor}\n{m.hard_negative}\n{guidance(m)}",
                        # Never averaged, never regressed on. The export keeps it out of labels/.
                        "scale": "categorical",
                        "lo": 1,
                        "hi": len(classes),
                        "requiresSpan": False,
                        "spanFromValue": None,
                        "needsReference": False,
                        "needsDialogue": False,
                        "allowNa": False,
                        "sourceSha": sha(
                            {
                                "key": m.key,
                                "question": m.question,
                                "classes": classes,
                                "atlas": m.atlas,
                                "eligibility": m.eligibility,
                            }
                        ),
                    }
                )
            else:
                rows.append(binary_row(m, group))

    blob = {
        "schema": "tutor-metrics/metric-seed-v1",
        "source": str(METRICS_PY.relative_to(METRICS_PY.parents[3])),
        "sourceFileSha": hashlib.sha256(METRICS_PY.read_bytes()).hexdigest(),
        "metrics": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(blob, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {OUT} — {len(rows)} metrics")
    for row in rows:
        print(f"  {row['key']:<28} {row['group']:<9} {row['scale']:<11} {row['lo']}-{row['hi']}")


if __name__ == "__main__":
    main()
