from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from projects.tutor_metrics.generate import tutor_messages_neutral
from projects.tutor_metrics.upgrade_probe_head import upgrade_head


def write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = tmp_path / "flat.npz"
    np.savez_compressed(
        source,
        schema=np.array("tutor-metrics/linear-head-v1"),
        model=np.array("Qwen/test"),
        revision=np.array("abc123"),
        prompt_scheme=np.array("stored"),
        metrics=np.array(["assistance_level", "reference_conflict"]),
        poolings=np.array(["eot", "last"]),
        layers=np.array([4, 3]),
        coefficients=np.arange(6, dtype=np.float32).reshape(2, 3),
        intercepts=np.array([1.5, 2.5], dtype=np.float32),
        feature_means=np.ones((2, 3), dtype=np.float32),
        feature_scales=np.full((2, 3), 2.0, dtype=np.float32),
        alphas=np.array([10.0, 20.0], dtype=np.float32),
        deployable=np.array([True, False]),
        label_min=np.array(1.0, dtype=np.float32),
        label_max=np.array(3.0, dtype=np.float32),
    )
    problem = {"question": "What is 1 + 1?", "choices": ["1", "2"]}
    units = tmp_path / "units.json"
    units.write_text(
        json.dumps(
            {
                "units": [
                    {
                        **problem,
                        "student_before": "I think it is 1.",
                        "encoder_messages": tutor_messages_neutral(problem, "I think it is 1."),
                    }
                ]
            }
        )
    )
    results = tmp_path / "results.json"
    results.write_text(
        json.dumps(
            {
                "metrics": {
                    "assistance_level": {"ridgeNested": {"pearson": 0.9, "mae": 0.1}},
                    "reference_conflict": {"ridgeNested": {"pearson": 0.2, "mae": 0.4}},
                }
            }
        )
    )
    return source, units, results


def test_flat_probe_head_is_upgraded_losslessly(tmp_path: Path) -> None:
    source, units, results = write_inputs(tmp_path)
    destination = tmp_path / "reward.npz"

    upgrade_head(
        source,
        destination,
        prompt_scheme="tutor_metrics_neutral",
        results_path=results,
        units_path=units,
    )

    with np.load(destination, allow_pickle=False) as head:
        meta = json.loads(str(head["meta"]))
        assert meta["schema"] == "tutor-metrics/reward-head-v1"
        assert meta["prompt_scheme"] == "tutor_metrics_neutral"
        assert meta["dimensions"]["assistance_level"]["deployable"] is True
        assert meta["dimensions"]["reference_conflict"]["deployable"] is False
        np.testing.assert_array_equal(head["assistance_level/coef"], np.arange(3, dtype=np.float32))


def test_stored_prompt_context_must_match_declared_scheme(tmp_path: Path) -> None:
    source, units, results = write_inputs(tmp_path)

    with pytest.raises(ValueError, match="does not prove stored context"):
        upgrade_head(
            source,
            tmp_path / "reward.npz",
            prompt_scheme="tutor_metrics",
            results_path=results,
            units_path=units,
        )
