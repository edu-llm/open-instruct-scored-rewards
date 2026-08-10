from __future__ import annotations

import pytest
from projects.learning_science_semantic_atlas import calibration_review

EXPECTED = [
    {"unit_id": "u1", "item_id": "i1", "concept": "c1"},
    {"unit_id": "u2", "item_id": "i2", "concept": "c2"},
]


def label(unit_id: str, item_id: str, concept: str, rater: str, *, presence: bool) -> dict[str, object]:
    return {
        "unit_id": unit_id,
        "item_id": item_id,
        "concept": concept,
        "rater": rater,
        "applicable": True,
        "applicability": "applicable",
        "presence": presence,
        "fidelity": 2 if presence else None,
        "quality": 2,
        "span": "quoted move" if presence else None,
        "flags": [],
        "quarantined": False,
    }


def test_summary_and_disagreement_queue() -> None:
    labels = {
        "a": {
            "u1": label("u1", "i1", "c1", "a", presence=True),
            "u2": label("u2", "i2", "c2", "a", presence=False),
        },
        "b": {
            "u1": label("u1", "i1", "c1", "b", presence=True),
            "u2": label("u2", "i2", "c2", "b", presence=True),
        },
    }
    summary = calibration_review.summarize(labels)
    assert summary["pairwise"]["a__b"]["presence"] == {"n": 2, "exact": 0.5}

    disagreements = calibration_review.disagreement_rows(labels, EXPECTED)
    assert len(disagreements) == 1
    assert disagreements[0]["unit_id"] == "u2"
    assert set(disagreements[0]["disagreements"]) == {"presence", "fidelity"}


def test_validation_requires_every_calibration_item(tmp_path) -> None:
    path = tmp_path / "labels.jsonl"
    path.write_text(
        __import__("json").dumps(label("u1", "i1", "c1", "a", presence=True)) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing 1 calibration units"):
        calibration_review.validate_labels([path], EXPECTED)


def test_validation_accepts_construct_as_a_concept_alias(tmp_path) -> None:
    rows = [
        {**label("u1", "i1", "c1", "a", presence=True), "construct": "c1"},
        {**label("u2", "i2", "c2", "a", presence=False), "construct": "c2"},
    ]
    for row in rows:
        row.pop("concept")
    path = tmp_path / "labels.jsonl"
    path.write_text("\n".join(__import__("json").dumps(row) for row in rows) + "\n", encoding="utf-8")
    validated = calibration_review.validate_labels([path], EXPECTED)
    assert set(validated["a"]) == {"u1", "u2"}
