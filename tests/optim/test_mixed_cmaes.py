"""Mixed CMA-ES integration tests with the optional upstream implementation."""

import pytest
import torch

from robotorchan.optim.backends.mixed_cmaes import optimize_acqf_mixed_cmaes
from robotorchan.optim.domains.variable_space import MixedVariableSpace

pytest.importorskip("cmaes")


class SumAcquisition(torch.nn.Module):
    def forward(self, candidates):
        return candidates.sum(dim=(-1, -2))


@pytest.mark.parametrize("q", [1, 2])
def test_mixed_cmaes_returns_legal_values_and_consistent_acquisition(q):
    bounds = torch.tensor([[0.0, 1.0, 10.0], [1.0, 3.0, 90.0]], dtype=torch.double)
    space = MixedVariableSpace(
        bounds,
        integer_dims=(1,),
        categorical_values={2: (90.0, 10.0, 30.0)},
    )
    candidate, value = optimize_acqf_mixed_cmaes(
        SumAcquisition(),
        bounds,
        q,
        variable_space=space,
        population_size=8,
        max_generations=3,
        seed=11,
    )
    assert candidate.shape == (q, 3)
    assert torch.all((candidate[:, 0] >= 0.0) & (candidate[:, 0] <= 1.0))
    assert all(v in {1.0, 2.0, 3.0} for v in candidate[:, 1].tolist())
    assert all(v in {10.0, 30.0, 90.0} for v in candidate[:, 2].tolist())
    torch.testing.assert_close(value, SumAcquisition()(candidate.unsqueeze(0)).reshape(()))


def test_mixed_cmaes_respects_fixed_categorical_values():
    bounds = torch.tensor([[0.0, 10.0], [1.0, 90.0]], dtype=torch.double)
    space = MixedVariableSpace(bounds, categorical_values={1: (10.0, 90.0, 30.0)})
    candidate, _ = optimize_acqf_mixed_cmaes(
        SumAcquisition(),
        bounds,
        1,
        variable_space=space,
        fixed_features={1: 30.0},
        population_size=8,
        max_generations=2,
        seed=5,
    )
    assert candidate[0, 1].item() == 30.0
