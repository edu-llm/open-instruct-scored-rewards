from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace

import numpy as np
import torch

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
    item = {"question": "What is 2 + 2?", "student_before": "I think it is 5.", "target_level": "prompt"}
    return Sample(
        completion="Check the addition in your ones place. What should 2 + 2 be?",
        prompt=item["question"],
        label=json.dumps(item),
    )


def test_context_matches_neutral_policy_prompt(tmp_path):
    scorer = TutorMetricsHead(head=str(head_file(tmp_path)))
    messages, _ = scorer.context(sample())

    assert messages == tutor_messages_neutral({"question": "What is 2 + 2?", "choices": None}, "I think it is 5.")


def test_metric_heads_compose_into_moving_target_reward(tmp_path):
    scorer = TutorMetricsHead(head=str(head_file(tmp_path)))
    scorer.states = lambda contexts: {("eot", 1): [np.zeros(2, dtype=np.float32) for _ in contexts]}  # type: ignore[method-assign]
    result = asyncio.run(scorer.score_group([sample()]))[0]

    assert result.info["raw"] == {"assistance_level": 2.0, "diagnoses_the_error": 3.0}
    assert result.dimensions["contingency"] == 2.0
    assert result.dimensions["contributions"] == 0.75
    assert result.score == 2.75


def test_states_requests_tensor_chat_template_output(tmp_path):
    scorer = TutorMetricsHead(head=str(head_file(tmp_path)))
    return_dict_values = []

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 0

        def apply_chat_template(self, *_args, return_dict=None, **_kwargs):
            return_dict_values.append(return_dict)
            return torch.tensor([[1, 2, 3]])

    class Model:
        def __call__(self, input_ids, **_kwargs):
            shape = (*input_ids.shape, 2)
            return SimpleNamespace(hidden_states=(torch.zeros(shape), torch.ones(shape)))

    scorer._model = Model()
    scorer._tokenizer = Tokenizer()
    scorer._input_device = torch.device("cpu")

    states = scorer.states([([{"role": "user", "content": "Question"}], "Try one step.")])

    assert return_dict_values == [False, False, False]
    assert states[("eot", 1)][0].shape == (2,)


def test_score_group_keeps_event_loop_live_and_serializes_worker(tmp_path):
    scorer = TutorMetricsHead(head=str(head_file(tmp_path)))
    release = threading.Event()
    lock = threading.Lock()
    worker_threads = []
    active = 0
    max_active = 0

    def fake_score(_group):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
            worker_threads.append(threading.get_ident())
        assert release.wait(timeout=1.0)
        with lock:
            active -= 1
        return []

    scorer._score_group_sync = fake_score  # type: ignore[method-assign]

    async def run_scores():
        event_loop_thread = threading.get_ident()
        loop = asyncio.get_running_loop()
        loop.call_later(0.05, release.set)
        await asyncio.gather(scorer.score_group([]), scorer.score_group([]))
        return event_loop_thread

    try:
        event_loop_thread = asyncio.run(run_scores())
    finally:
        scorer._score_executor.shutdown(wait=True)

    assert max_active == 1
    assert len(set(worker_threads)) == 1
    assert worker_threads[0] != event_loop_thread
