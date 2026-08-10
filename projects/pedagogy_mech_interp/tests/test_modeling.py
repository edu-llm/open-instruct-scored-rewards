import torch

from projects.pedagogy_mech_interp.modeling import (
    normalize_gate_output,
    route,
    selected_expert_contributions,
    verify_moe_decomposition,
)


class ScaleExpert(torch.nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, value):
        return value * self.scale


class ToyMoe4(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.top_k = 2
        self.norm_topk_prob = True
        self.num_experts = 3
        self.gate = torch.nn.Linear(4, 3, bias=False)
        self.experts = torch.nn.ModuleList([ScaleExpert(1.0), ScaleExpert(2.0), ScaleExpert(-1.0)])
        with torch.no_grad():
            self.gate.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                    ]
                )
            )

    def forward(self, hidden_states):
        routing = route(self, hidden_states)
        value = selected_expert_contributions(self, hidden_states, routing).sum(dim=1)
        return value.reshape_as(hidden_states), routing.logits


class ToyGate5(torch.nn.Module):
    def __init__(self, top_k=2):
        super().__init__()
        self.linear = torch.nn.Linear(4, 3, bias=False)
        self.top_k = top_k

    @property
    def weight(self):
        return self.linear.weight

    def forward(self, hidden_states):
        logits = self.linear(hidden_states)
        probabilities = logits.softmax(dim=-1)
        weights, indices = probabilities.topk(self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        return logits, weights, indices


class ToyGate54(ToyGate5):
    def forward(self, hidden_states):
        probabilities = self.linear(hidden_states).softmax(dim=-1)
        weights, indices = probabilities.topk(self.top_k, dim=-1)
        return probabilities, weights, indices


def test_transformers4_style_decomposition_reconstructs_block():
    block = ToyMoe4()
    hidden = torch.randn(2, 3, 4)
    error = verify_moe_decomposition(block, hidden, atol=1e-6, rtol=1e-6)
    assert error < 1e-6


def test_transformers5_style_gate_tuple_is_normalized():
    block = ToyMoe4()
    gate = ToyGate5()
    hidden = torch.randn(5, 4)
    routing = normalize_gate_output(block, gate(hidden))
    assert routing.indices.shape == (5, 2)
    assert torch.allclose(routing.weights.sum(dim=-1), torch.ones(5))


def test_transformers54_probability_tuple_recovers_true_logits():
    block = ToyMoe4()
    block.gate = ToyGate54()
    hidden = torch.randn(5, 4)
    routing = route(block, hidden)
    expected = torch.nn.functional.linear(hidden, block.gate.weight)
    assert torch.allclose(routing.logits, expected)
    assert torch.allclose(routing.logits.softmax(dim=-1), expected.softmax(dim=-1))


def test_native_unnormalized_topk_mass_is_preserved():
    block = ToyMoe4()
    block.norm_topk_prob = False
    hidden = torch.zeros(5, 4)
    routing = route(block, hidden)
    assert torch.all(routing.weights.sum(dim=-1) < 1)
