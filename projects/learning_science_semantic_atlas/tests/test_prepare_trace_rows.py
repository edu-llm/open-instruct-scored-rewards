"""CPU tests for the withheld-artifact to tracer-row join."""

from __future__ import annotations

import json

import pytest
from projects.learning_science_semantic_atlas import prepare_trace_rows as prepare


def prompt(unit_id: str, sha: str = "sha-a") -> dict[str, object]:
    return {
        "schema": prepare.PROMPT_SCHEMA,
        "unit_id": unit_id,
        "prompt_sha": sha,
        "system": "System line.\nKeep this spacing.",
        "user": "User line.\n\nSecond paragraph.",
    }


def generation(
    unit_id: str, text: str, *, sha: str = "sha-a", refused: list[str] | None = None
) -> dict[str, object]:
    return {
        "schema": prepare.GENERATION_SCHEMA,
        "unit_id": unit_id,
        "prompt_sha": sha,
        "text": text,
        "source_model": "allenai/example",
        "source_policy": "sample",
        "seed": 17,
        "refusal_reasons": refused or [],
    }


def key(unit_id: str, sha: str = "sha-a") -> dict[str, object]:
    return {
        "unit_id": unit_id,
        "item_id": "lineage-1",
        "designed_target_construct": "retrieval_practice_opportunity",
        "domain": "chemistry",
        "expected_presence": True,
        "names": False,
        "enacts": True,
        "prompt_sha": sha,
        "block": "round1",
        "pool": "targeted",
    }


def test_join_preserves_exact_text_and_tracer_fields():
    rows = prepare.prepare_trace_rows(
        [prompt("u1")],
        [generation("u1", "First response."), generation("u1", "  Exact response.\n")],
        [key("u1")],
        model_tag="olmoe-base",
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["record_id"] == "u1"
    assert row["item_id"] == row["sibling_id"] == "lineage-1"
    assert row["concept"] == "retrieval_practice_opportunity"
    assert row["domain"] == "chemistry"
    assert row["response"] == "  Exact response.\n"
    assert row["messages"] == [
        {"role": "system", "content": "System line.\nKeep this spacing."},
        {"role": "user", "content": "User line.\n\nSecond paragraph."},
    ]
    assert row["label"] == row["designed_label"] == 1
    assert row["named_label"] == 0
    assert row["enacted_label"] == 1
    assert row["label_source"] == "designed"
    assert row["provenance"]["source_model"] == "allenai/example"
    assert row["provenance"]["model_tag"] == "olmoe-base"


def test_newest_accepted_generation_ignores_a_later_refusal():
    rows = [
        generation("u1", "accepted one"),
        generation("u1", "accepted two"),
        generation("u1", "refused retry", refused=["echoed brief"]),
    ]
    newest = prepare.newest_accepted_generations(rows)
    assert newest["u1"]["text"] == "accepted two"


def test_stale_prompt_generation_pair_is_rejected():
    with pytest.raises(ValueError, match="prompt_sha values disagree"):
        prepare.prepare_trace_rows(
            [prompt("u1", "new")],
            [generation("u1", "old response", sha="old")],
            [key("u1", "new")],
            model_tag="dense-base",
        )


def test_key_order_selects_a_subset_and_jsonl_round_trips(tmp_path):
    rows = prepare.prepare_trace_rows(
        [prompt("unused"), prompt("selected")],
        [generation("unused", "no"), generation("selected", "yes")],
        [key("selected")],
        model_tag="olmoe-base",
    )
    target = tmp_path / "trace_rows.jsonl"
    assert prepare.write_jsonl(rows, target) == 1
    stored = json.loads(target.read_text())
    assert stored["record_id"] == "selected"
    assert stored["response"] == "yes"


def test_missing_accepted_generation_fails_instead_of_dropping_a_key_row():
    with pytest.raises(ValueError, match="lack accepted generations"):
        prepare.prepare_trace_rows(
            [prompt("u1")],
            [generation("u1", "bad", refused=["identifier leak"])],
            [key("u1")],
            model_tag="olmoe-base",
        )
