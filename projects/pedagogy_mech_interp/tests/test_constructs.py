import pytest

from projects.pedagogy_mech_interp.constructs import (
    scaffolding_distance,
    score_contingent_scaffolding,
    score_generative_elicitation,
    score_record,
    validate_label,
)


def test_contingent_scaffolding_is_scenario_conditioned():
    assert score_contingent_scaffolding("strategy_hint", "strategy_hint") == 1
    assert score_contingent_scaffolding("direct_answer", "strategy_hint") == 0
    assert scaffolding_distance("strategy_hint", "direct_answer") == 2
    assert scaffolding_distance("worked_step", "question") == -2


def test_generative_actions_are_explicit_not_keyword_inferred():
    assert score_generative_elicitation("self_explain") == 1
    assert score_generative_elicitation("tutor_explanation") == 0
    with pytest.raises(ValueError, match="unknown generative action"):
        score_generative_elicitation("sounds_socratic")


def test_diagnostic_label_requires_both_correctness_and_location():
    base = {
        "concept": "diagnostic_localization",
        "student_before": "I divided 8 by 2 and got 2.",
        "reference": "8 / 2 = 4",
        "candidate_action": "diagnose_first_error",
    }
    assert score_record({**base, "diagnosis_correct": True, "first_error_located": True}) == 1
    assert score_record({**base, "diagnosis_correct": True, "first_error_located": False}) == 0
    with pytest.raises(ValueError, match="label must be 0 or 1"):
        validate_label(
            {
                **base,
                "diagnosis_correct": True,
                "first_error_located": True,
                "label": True,
            }
        )
