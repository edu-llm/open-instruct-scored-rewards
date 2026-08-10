import numpy as np

from projects.pedagogy_mech_interp.analyze import (
    benjamini_hochberg,
    bootstrap_ci,
    claim_status,
    holm,
    monotonic_in_expected_direction,
    paired_differences,
    repair_fraction,
)


def test_paired_differences_use_pair_as_sampling_unit():
    differences = paired_differences(
        values=[0.2, 0.8, 1.5, 1.0],
        labels=[0, 1, 1, 0],
        pair_ids=["a", "a", "b", "b"],
    )
    assert np.isclose(differences["a"], 0.6)
    assert np.isclose(differences["b"], 0.5)
    estimate, low, high = bootstrap_ci(differences, samples=500, seed=3)
    assert low <= estimate <= high
    assert np.isclose(estimate, 0.55)


def test_multiplicity_adjustments_are_monotone_in_rank():
    values = [0.01, 0.04, 0.02]
    bh = benjamini_hochberg(values)
    corrected_holm = holm(values)
    assert all(adjusted >= raw for adjusted, raw in zip(bh, values))
    assert all(adjusted >= raw for adjusted, raw in zip(corrected_holm, values))


def test_repair_and_claim_gates():
    assert repair_fraction(clean=2.0, corrupted=0.0, patched=1.0) == 0.5
    assert repair_fraction(clean=1.0, corrupted=1.0, patched=1.0) is None
    assert monotonic_in_expected_direction([-1, 0, 1], [-0.2, 0.0, 0.4])
    assert (
        claim_status(
            behavior_pass=True,
            decodability_pass=True,
            association_pass=True,
            necessity_pass=True,
            rescue_pass=True,
            specificity_pass=True,
        )
        == "causally_relevant"
    )
