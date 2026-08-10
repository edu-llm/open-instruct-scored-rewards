import torch

from projects.pedagogy_mech_interp.intervene import ActivationEditor, ExpertOutputScaler, RouterEditor
from projects.pedagogy_mech_interp.modeling import normalize_gate_output, route, tensor_from_output
from projects.pedagogy_mech_interp.tests.test_modeling import ToyMoe4


class ToyLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = torch.nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            self.mlp.weight.copy_(torch.eye(4))

    def forward(self, value):
        return value + self.mlp(value)


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([ToyLayer()])

    def forward(self, value):
        return self.model.layers[0](value)


def test_zero_dose_activation_hook_has_exact_parity():
    model = ToyModel()
    value = torch.randn(1, 3, 4)
    baseline = model(value)
    with ActivationEditor(model, 0, "residual_out", [2], torch.tensor([1.0, 0.0, 0.0, 0.0]), "add", 0.0):
        changed = model(value)
    assert torch.equal(baseline, changed)


def test_projection_ablation_changes_only_selected_position():
    model = ToyModel()
    value = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]]])
    with ActivationEditor(model, 0, "residual_out", [1], torch.tensor([1.0, 0.0, 0.0, 0.0]), "remove"):
        changed = model(value)
    baseline = model(value)
    assert torch.equal(changed[:, 0], baseline[:, 0])
    assert changed[0, 1, 0] == 0
    assert torch.equal(changed[0, 1, 1:], baseline[0, 1, 1:])


def test_router_force_renormalizes_and_selects_expert():
    block = ToyMoe4()
    hidden = torch.randn(1, 2, 4)
    captured = {}

    def capture(_module, _args, output):
        captured["routing"] = normalize_gate_output(block, output)

    with RouterEditor(block, [1], expert_index=2, mode="force"):
        capture_handle = block.gate.register_forward_hook(capture)
        block(hidden)
        capture_handle.remove()
    routing = captured["routing"]
    assert 2 in routing.indices[1].tolist()
    assert torch.allclose(routing.weights.sum(dim=-1), torch.ones(2), atol=1e-6)


def test_expert_ablation_subtracts_selected_contribution():
    block = ToyMoe4()
    hidden = torch.randn(1, 2, 4)
    baseline = tensor_from_output(block(hidden))
    routing = route(block, hidden)
    target = int(routing.indices[1, 0])
    with ExpertOutputScaler(block, [1], target, 0.0):
        changed = tensor_from_output(block(hidden))
    assert torch.equal(changed[:, 0], baseline[:, 0])
    assert not torch.allclose(changed[:, 1], baseline[:, 1])
