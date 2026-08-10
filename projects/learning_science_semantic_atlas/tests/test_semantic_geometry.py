"""CPU-only tests for held-out-concept geometry and RSA."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from projects.learning_science_semantic_atlas import semantic_geometry as geometry
from projects.learning_science_semantic_atlas.schema import load_ontology
from projects.learning_science_semantic_atlas.trace import AtlasTrace


def synthetic_trace(*, seed: int = 0) -> AtlasTrace:
    rng = np.random.default_rng(seed)
    ontology = load_ontology()
    concepts = [ontology.constructs[index].key for index in (0, 1, 5, 6, 10, 15)]
    pairs_per_concept = 6
    labels = np.tile([0, 1], len(concepts) * pairs_per_concept)
    concept_values = np.repeat(concepts, pairs_per_concept * 2)
    siblings = np.asarray(
        [
            f"{concept}-pair-{pair}"
            for concept in concepts
            for pair in range(pairs_per_concept)
            for _ in range(2)
        ]
    )
    n_records = len(labels)
    hidden = 10
    states = rng.normal(scale=0.15, size=(n_records, 1, 1, hidden))
    states[..., 0] += labels[:, None, None] * 5.0
    arrays = {
        "ids": np.asarray([f"r{index}" for index in range(n_records)]),
        "item_ids": siblings.copy(),
        "sibling_ids": siblings,
        "constructs": concept_values,
        "labels": labels.astype(np.int32),
        "named_labels": labels.astype(np.int32),
        "enacted_labels": labels.astype(np.int32),
        "quality": rng.normal(size=n_records).astype(np.float32),
        "layers": np.asarray([3], dtype=np.int16),
        "position_roles": np.asarray(["content_last"]),
        "position_bases": np.asarray(["content_last"]),
        "position_ordinals": np.asarray([0], dtype=np.int16),
        "position_mask": np.ones((n_records, 1), dtype=bool),
        "position_index": np.zeros((n_records, 1), dtype=np.int32),
        "position_token_id": rng.integers(0, 1000, size=(n_records, 1), dtype=np.int32),
        "surface_features": rng.normal(size=(n_records, 5)).astype(np.float32),
        "token_histogram": rng.normal(size=(n_records, 12)).astype(np.float32),
        "mlp_in": states.astype(np.float16),
    }
    return AtlasTrace(
        arrays=arrays,
        metadata={
            "schema": "semantic-atlas-trace/v1",
            "model": "toy",
            "layers": [3],
            "slots": ["content_last"],
            "mixture_of_experts": False,
        },
    )


def test_shared_activation_direction_generalizes_to_wholly_held_out_concepts():
    trace = synthetic_trace()
    report = geometry.control_aware_held_out_summary(
        trace, site="mlp_in", layer_index=3, slot=0, seed=4
    )
    assert report["activation"]["n_concepts"] == 6
    assert report["activation"]["mean_balanced_accuracy"] > 0.99
    assert report["activation"]["mean_forced_choice_accuracy"] == 1.0
    assert report["selectivity_over_best_control"] > 0.2
    assert {control["feature"] for control in report["controls"]} == {
        "surface",
        "global_quality",
        "token_histogram",
        "token_identity",
    }


def test_concept_effects_are_positive_minus_negative_centroids():
    features = np.asarray([[1.0, 4.0], [5.0, 8.0], [3.0, 1.0], [7.0, 5.0]])
    labels = np.asarray([0, 1, 0, 1])
    concepts = np.asarray(["a", "a", "b", "b"])
    names, effects = geometry.concept_effects(features, labels, concepts)
    assert names == ["a", "b"]
    assert np.array_equal(effects, np.asarray([[4.0, 4.0], [4.0, 4.0]]))


def test_ontology_rdm_orders_siblings_before_other_families():
    ontology = load_ontology()
    first = ontology.constructs[0]
    sibling = ontology.get(first.sibling_key)
    other = next(construct for construct in ontology if construct.family is not first.family)
    matrix = geometry.ontology_distance_matrix([first.key, sibling.key, other.key], ontology)
    assert matrix[0, 1] == 0.25
    assert matrix[0, 2] == 1.0
    assert np.array_equal(np.diag(matrix), np.zeros(3))


def test_cosine_rdm_and_rank_statistics_have_pinned_behavior():
    vectors = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    matrix = geometry.cosine_distance_matrix(vectors)
    assert matrix[0, 1] == pytest.approx(1.0)
    assert matrix[0, 2] == pytest.approx(2.0)
    assert geometry.rank_correlation(np.arange(8), np.arange(8) ** 2) == pytest.approx(1.0)
    assert geometry.partial_rank_correlation(np.arange(8), np.arange(8) ** 2, []) == pytest.approx(1.0)


def test_full_report_marks_agent_grounded_geometry_as_exploratory():
    trace = synthetic_trace()
    report = geometry.analyze_trace(
        trace,
        load_ontology(),
        model_tag="toy-dense",
        site="mlp_in",
        layer_index=3,
        role="content_last",
        label_sources=["agent_consensus"],
        permutations=20,
    )
    assert report["model_tag"] == "toy-dense"
    assert report["analysis_status"] == "exploratory"
    assert report["label_sources"] == ["agent_consensus"]
    assert report["rsa"]["n_concepts"] == 6
    assert any("agent-grounded" in limit for limit in report["claim_limits"])
    assert "causal" in report["rsa"]["claim_limit"]


def test_trace_specs_require_explicit_model_tags():
    with pytest.raises(ValueError, match="explicit --model-tag"):
        geometry._trace_specs(["trace.npz"], [])
    assert geometry._trace_specs(["dense=trace.npz"], []) == [("dense", Path("trace.npz"))]
