"""End-to-end mixed particle swarm regression coverage."""

import pytest
import torch

from robotorchan.optim import optimize_acqf
from robotorchan.optim.domains.variable_space import MixedVariableSpace


class SumAcquisition(torch.nn.Module):
    def forward(self, candidates):
        return candidates.sum(dim=(-1, -2))


@pytest.mark.parametrize("q", [1, 2])
def test_mixed_pso_legal_candidates_and_reproducibility(q):
    bounds = torch.tensor([[0.0, 1.0, 10.0], [1.0, 4.0, 90.0]], dtype=torch.double)
    space = MixedVariableSpace(
        bounds,
        integer_dims=(1,),
        categorical_values={2: (10.0, 90.0, 30.0)},
    )
    kwargs = {
        "optimizer": "pso",
        "variable_space": space,
        "seed": 13,
        "optimizer_options": {"swarm_size": 24, "iterations": 8},
    }
    first, value = optimize_acqf(SumAcquisition(), bounds, q, **kwargs)
    second, second_value = optimize_acqf(SumAcquisition(), bounds, q, **kwargs)
    assert first.shape == (q, 3)
    assert torch.all(first >= bounds[0])
    assert torch.all(first <= bounds[1])
    assert torch.all(first[:, 1] == first[:, 1].round())
    assert torch.isin(first[:, 2], torch.tensor([10.0, 90.0, 30.0])).all()
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(value, second_value)
    torch.testing.assert_close(value.reshape(()), first.sum())


def test_mixed_pso_fixed_discrete_features():
    bounds = torch.tensor([[0.0, 1.0, 10.0], [1.0, 4.0, 90.0]], dtype=torch.double)
    space = MixedVariableSpace(
        bounds,
        integer_dims=(1,),
        categorical_values={2: (10.0, 90.0, 30.0)},
    )
    candidate, value = optimize_acqf(
        SumAcquisition(),
        bounds,
        2,
        optimizer="pso",
        variable_space=space,
        fixed_features={1: 2.0, 2: 30.0},
        seed=7,
        optimizer_options={"swarm_size": 16, "iterations": 5},
    )
    assert torch.all(candidate[:, 1] == 2.0)
    assert torch.all(candidate[:, 2] == 30.0)
    torch.testing.assert_close(value.reshape(()), candidate.sum())
