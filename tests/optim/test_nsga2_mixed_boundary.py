"""NSGA-II mixed-domain boundary tests before implementing discrete operators."""

import torch

from robotorchan.optim.backends import optimize_vector_nsga2
from robotorchan.optim.capabilities import NSGA2_OPTIMIZER_CAPABILITIES
from robotorchan.optim.domains.variable_space import MixedVariableSpace


def test_nsga2_declares_continuous_only_support():
    capabilities = NSGA2_OPTIMIZER_CAPABILITIES
    assert capabilities.continuous
    assert not capabilities.integer
    assert not capabilities.categorical
    assert not capabilities.mixed
    assert not capabilities.q_batch


def test_mixed_domain_requires_discrete_aware_genetic_operators():
    bounds = torch.tensor([[0.0, 1.0, 10.0], [1.0, 3.0, 90.0]], dtype=torch.double)
    space = MixedVariableSpace(
        bounds,
        integer_dims=(1,),
        categorical_values={2: (10.0, 30.0, 90.0)},
    )
    assert space.continuous_dims == (0,)
    assert space.integer_dims == (1,)
    assert space.categorical_dims == (2,)
    assert space.is_mixed
    assert not NSGA2_OPTIMIZER_CAPABILITIES.mixed


def test_continuous_nsga2_remains_a_vector_optimizer():
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)

    def objective(candidates):
        x = candidates[:, 0]
        return torch.stack([-x.square(), -(x - 1).square()], dim=-1)

    candidates, values = optimize_vector_nsga2(
        objective,
        bounds,
        population_size=12,
        generations=3,
        seed=17,
    )
    assert candidates.ndim == 2
    assert values.shape == (candidates.shape[0], 2)
    assert torch.all((candidates >= bounds[0]) & (candidates <= bounds[1]))
