"""Mixed-domain vector NSGA-II runtime contracts."""

import pytest
import torch

from robotorchan.optim.backends import optimize_vector_nsga2_mixed
from robotorchan.optim.domains.variable_space import MixedVariableSpace


def _objective(x):
    return torch.stack((x[:, 0] + x[:, 1], -x[:, 0] + (x[:, 2] == 30).double()), dim=-1)


@pytest.mark.parametrize("generations", [1, 4])
def test_mixed_nsga2_returns_legal_nondominated_candidates(generations):
    bounds = torch.tensor([[0.0, 1.0, 10.0], [1.0, 3.0, 90.0]], dtype=torch.double)
    space = MixedVariableSpace(
        bounds, integer_dims=(1,), categorical_values={2: (90.0, 10.0, 30.0)}
    )
    candidates, values = optimize_vector_nsga2_mixed(
        _objective,
        bounds,
        variable_space=space,
        population_size=24,
        generations=generations,
        seed=7,
    )
    assert candidates.shape[1] == 3
    assert values.shape == (candidates.shape[0], 2)
    assert torch.all((candidates >= bounds[0]) & (candidates <= bounds[1]))
    assert set(candidates[:, 1].tolist()) <= {1.0, 2.0, 3.0}
    assert set(candidates[:, 2].tolist()) <= {10.0, 30.0, 90.0}
    torch.testing.assert_close(values, _objective(candidates))
    dominates = (values.unsqueeze(1) >= values.unsqueeze(0)).all(dim=-1) & (
        values.unsqueeze(1) > values.unsqueeze(0)
    ).any(dim=-1)
    assert not dominates.any()


def test_mixed_nsga2_seed_is_reproducible():
    bounds = torch.tensor([[0.0, 1.0, 10.0], [1.0, 3.0, 90.0]], dtype=torch.double)
    space = MixedVariableSpace(
        bounds, integer_dims=(1,), categorical_values={2: (10.0, 30.0, 90.0)}
    )
    options = {"variable_space": space, "population_size": 16, "generations": 4, "seed": 9}
    first = optimize_vector_nsga2_mixed(_objective, bounds, **options)
    second = optimize_vector_nsga2_mixed(_objective, bounds, **options)
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])


def test_mixed_nsga2_rejects_mismatched_bounds():
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    space = MixedVariableSpace(torch.tensor([[0.0], [2.0]], dtype=torch.double))
    with pytest.raises(ValueError, match="bounds must match"):
        optimize_vector_nsga2_mixed(
            lambda x: torch.cat((x, -x), dim=1), bounds, variable_space=space
        )
