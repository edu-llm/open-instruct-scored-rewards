import json

import pytest

from projects.pedagogy_mech_interp.build_pairs import (
    assert_no_cross_split_leakage,
    assign_split,
    family_errors,
    flatten,
)


def family():
    return {
        "item_id": "item-1",
        "pair_id": "pair-1",
        "template_id": "template-1",
        "domain": "mathdial",
        "source": "test",
        "concept": "contingent_scaffolding",
        "student_state": "attempt",
        "prescribed_action": "strategy_hint",
        "natural_or_synthetic": "natural",
        "question": "What is 2 + 2?",
        "student_before": "I think it is 5.",
        "variants": [
            {
                "variant_id": "positive",
                "tutor_turn": "Which addition step would you check first?",
                "candidate_action": "strategy_hint",
                "label": 1,
                "expected_direction": 1,
            },
            {
                "variant_id": "negative",
                "tutor_turn": "The answer is four.",
                "candidate_action": "direct_answer",
                "label": 0,
                "expected_direction": -1,
            },
        ],
    }


def test_confirmatory_domain_is_held_out_entirely():
    assert assign_split("anything", 1701, "bridge", "bridge") == "confirmatory"
    assert assign_split("anything", 1701, "mathdial", "bridge") != "confirmatory"


def test_family_flattens_and_validates():
    value = family()
    assert family_errors(value) == []
    rows = flatten(value, "discovery_train")
    assert {row["label"] for row in rows} == {0, 1}
    assert len({row["context_hash"] for row in rows}) == 1


def test_cross_split_text_leakage_is_rejected():
    first = flatten(family(), "discovery_train")
    second_family = json.loads(json.dumps(family()))
    second_family["item_id"] = "item-2"
    second_family["pair_id"] = "pair-2"
    for variant in second_family["variants"]:
        variant["variant_id"] += "-2"
    second = flatten(second_family, "natural_test")
    with pytest.raises(ValueError, match="normalized response"):
        assert_no_cross_split_leakage([*first, *second])
