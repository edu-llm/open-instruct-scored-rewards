from projects.tutor_metrics.metrics import probe_reward


def test_absent_predictions_do_not_count_as_positive():
    score, terms = probe_reward({"assistance_level": 2.0}, "prompt")

    assert score == terms["contingency"] == 2.0
    assert terms["contributions"] == 0.0
    assert terms["locates_student_object"] == 0.0


def test_assistance_target_moves_with_student_state():
    low, _ = probe_reward({"assistance_level": 1.0}, "hint")
    high, _ = probe_reward({"assistance_level": 1.0}, "tell_step")

    assert low == 2.0
    assert high == 0.0


def test_partial_predictions_give_continuous_credit():
    labels = {
        "assistance_level": 2.0,
        "diagnoses_the_error": 2.0,
        "locates_student_object": 2.0,
        "reference_conflict": 2.0,
    }
    score, terms = probe_reward(labels, "prompt")

    assert terms["contributions"] == 0.375
    assert terms["locates_student_object"] == 0.25
    assert terms["reference_conflict"] == -0.5
    assert score == 2.125
