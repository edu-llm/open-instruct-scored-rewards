from __future__ import annotations

import asyncio
import json

import numpy as np

from open_instruct.scored_rewards import Sample
from projects.tutor_metrics.generate import tutor_messages_neutral
from projects.tutor_metrics.plugin import TutorMetricsHead


def head_file(tmp_path):
    dimensions = {
        "assistance_level": {"pooling": "eot", "layer": 1, "lo": 1.0, "hi": 3.0},
        "diagnoses_the_error": {"pooling": "eot", "layer": 1, "lo": 1.0, "hi": 3.0},
    }
    arrays = {
        "meta": np.array(
            json.dumps(
                {
                    "schema": "tutor-metrics/reward-head-v1",
                    "model": "unit/model",
                    "revision": "abc123",
                    "prompt_scheme": "tutor_metrics_neutral",
                    "dimensions": dimensions,
                }
            )
        )
    }
    for key, intercept in (("assistance_level", 2.0), ("diagnoses_the_error", 3.0)):
        arrays[f"{key}/mean"] = np.zeros(2, dtype=np.float32)
        arrays[f"{key}/scale"] = np.ones(2, dtype=np.float32)
        arrays[f"{key}/coef"] = np.zeros(2, dtype=np.float32)
        arrays[f"{key}/intercept"] = np.float32(intercept)
    path = tmp_path / "head.npz"
    np.savez_compressed(path, **arrays)
    return path


def sample() -> Sample:
    item = {
        "question": "What is 2 + 2?",
        "student_before": "I think it is 5.",
        "target_level": "prompt",
    }
    return Sample(
        completion="Check the addition in your ones place. What should 2 + 2 be?",
        prompt=item["question"],
        label=json.dumps(item),
    )


def test_context_matches_neutral_policy_prompt(tmp_path):
    scorer = TutorMetricsHead(head=str(head_file(tmp_path)))
    messages, _ = scorer.context(sample())

    assert messages == tutor_messages_neutral(
        {"question": "What is 2 + 2?", "choices": None},
        "I think it is 5.",
    )


def test_metric_heads_compose_into_moving_target_reward(tmp_path):
    scorer = TutorMetricsHead(head=str(head_file(tmp_path)))
    scorer.states = lambda contexts: {("eot", 1): [np.zeros(2, dtype=np.float32) for _ in contexts]}  # type: ignore[method-assign]
    result = asyncio.run(scorer.score_group([sample()]))[0]

    assert result.info["raw"] == {"assistance_level": 2.0, "diagnoses_the_error": 3.0}
    assert result.dimensions["contingency"] == 2.0
    assert result.dimensions["contributions"] == 0.75
    assert result.score == 2.75
