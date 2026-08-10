"""Validate blinded calibration labels and build an inter-agent disagreement queue."""

from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path

from projects.learning_science_semantic_atlas.schema import LabelRecord


def read_jsonl(path: str | Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(value)
    return rows


def validate_labels(
    label_paths: Iterable[str | Path], expected_rows: Iterable[Mapping[str, object]]
) -> dict[str, dict[str, dict[str, object]]]:
    expected = {str(row["unit_id"]): row for row in expected_rows}
    by_rater: dict[str, dict[str, dict[str, object]]] = {}
    for path in label_paths:
        for raw in read_jsonl(path):
            payload = dict(raw)
            if "construct" in payload:
                if "concept" in payload and payload["concept"] != payload["construct"]:
                    raise ValueError(f"{path}: concept and construct aliases disagree")
                payload["concept"] = payload.pop("construct")
            record = LabelRecord.from_mapping(payload)
            row = record.as_dict()
            unit_id = str(row["unit_id"])
            if unit_id not in expected:
                raise ValueError(f"{path}: unknown calibration unit {unit_id!r}")
            if row["concept"] != expected[unit_id].get("concept"):
                raise ValueError(
                    f"{path}: {unit_id} labels {row['concept']!r}, expected {expected[unit_id].get('concept')!r}"
                )
            rater = str(row["rater"])
            if unit_id in by_rater.setdefault(rater, {}):
                raise ValueError(f"{path}: rater {rater!r} labels {unit_id!r} twice")
            by_rater[rater][unit_id] = row
    for rater, rows in by_rater.items():
        missing = sorted(set(expected) - set(rows))
        if missing:
            raise ValueError(f"rater {rater!r} is missing {len(missing)} calibration units: {missing[:5]}")
    return by_rater


def _pair_agreement(left: Mapping[str, Mapping[str, object]], right: Mapping[str, Mapping[str, object]], field: str):
    pairs = [
        (left[unit_id].get(field), right[unit_id].get(field))
        for unit_id in sorted(set(left) & set(right))
        if left[unit_id].get(field) is not None and right[unit_id].get(field) is not None
    ]
    return {
        "n": len(pairs),
        "exact": sum(a == b for a, b in pairs) / len(pairs) if pairs else None,
    }


def summarize(by_rater: Mapping[str, Mapping[str, Mapping[str, object]]]) -> dict[str, object]:
    raters = sorted(by_rater)
    pairwise = {}
    for left, right in itertools.combinations(raters, 2):
        pairwise[f"{left}__{right}"] = {
            field: _pair_agreement(by_rater[left], by_rater[right], field)
            for field in ("applicable", "presence", "fidelity", "quality")
        }
    return {"raters": raters, "n_raters": len(raters), "pairwise": pairwise}


def disagreement_rows(
    by_rater: Mapping[str, Mapping[str, Mapping[str, object]]],
    expected_rows: Iterable[Mapping[str, object]],
) -> list[dict[str, object]]:
    expected = {str(row["unit_id"]): row for row in expected_rows}
    out = []
    for unit_id in sorted(expected):
        labels = {rater: rows[unit_id] for rater, rows in by_rater.items()}
        fields = {}
        for field in ("applicable", "presence", "fidelity", "quality"):
            values = [row.get(field) for row in labels.values()]
            if len(set(values)) > 1:
                fields[field] = dict(Counter(str(value) for value in values))
        if fields:
            out.append(
                {
                    "unit_id": unit_id,
                    "item_id": expected[unit_id].get("item_id"),
                    "construct": expected[unit_id].get("concept"),
                    "disagreements": fields,
                    "labels": labels,
                }
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--disagreements-out", required=True)
    args = parser.parse_args()

    expected = read_jsonl(args.calibration)
    labels = validate_labels(args.labels, expected)
    summary = summarize(labels)
    disagreements = disagreement_rows(labels, expected)
    summary["n_disagreement_items"] = len(disagreements)

    Path(args.summary_out).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with Path(args.disagreements_out).open("w", encoding="utf-8") as handle:
        for row in disagreements:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    print(f"validated {len(labels)} raters; {len(disagreements)} of {len(expected)} items need calibration review")


if __name__ == "__main__":
    main()
