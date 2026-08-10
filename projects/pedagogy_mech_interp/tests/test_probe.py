import numpy as np

from projects.pedagogy_mech_interp.probe import (
    choose_activation_candidate,
    choose_router_candidate,
    fit_activation_model,
    raw_direction,
)


def synthetic_blob(seed=1):
    rng = np.random.default_rng(seed)
    pairs = 12
    labels = np.tile([0, 1], pairs)
    concepts = np.asarray(["diagnostic_localization"] * (pairs * 2))
    pair_ids = np.repeat([f"p{i}" for i in range(pairs)], 2)
    signal = labels[:, None, None] * 2.0
    residual = rng.normal(size=(pairs * 2, 2, 5)).astype(np.float32)
    residual[:, 1:2, :1] += signal
    router = rng.normal(size=(pairs * 2, 2, 4)).astype(np.float32)
    router[:, 1, 2] += labels * 2.0
    return {
        "labels": labels,
        "concepts": concepts,
        "pair_ids": pair_ids,
        "layers": np.asarray([3, 7]),
        "residual_in": residual,
        "residual_out": residual,
        "mlp_in": residual,
        "mlp_out": residual,
        "router_logits": router,
    }


def test_activation_and_router_candidates_find_injected_signal():
    train = synthetic_blob(2)
    validation = synthetic_blob(3)
    candidate, _ = choose_activation_candidate(train, validation, "diagnostic_localization")
    assert candidate["layer_index"] == 7
    router = choose_router_candidate(validation, "diagnostic_localization", 100, 4)
    assert router["layer_index"] == 7
    assert router["expert_index"] == 2


def test_raw_direction_is_unit_norm():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(30, 6))
    y = (x[:, 0] > 0).astype(int)
    model = fit_activation_model(x, y, 1.0)
    direction, intercept = raw_direction(model)
    assert np.isclose(np.linalg.norm(direction), 1.0)
    assert np.isfinite(intercept)
