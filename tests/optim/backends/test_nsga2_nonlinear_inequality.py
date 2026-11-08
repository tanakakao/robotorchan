"""Focused NSGA-II nonlinear inequality constraint tests."""

import pytest
import torch

from robotorchan.optim.backends.nsga2 import optimize_vector_nsga2
from robotorchan.optim.constraints import CandidateConstraints


def test_nsga2_intrapoint_nonlinear_inequality() -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)

    def objective(x: torch.Tensor) -> torch.Tensor:
        return torch.cat((x, 1 - x), dim=-1)

    candidates, values = optimize_vector_nsga2(
        objective,
        bounds,
        population_size=16,
        generations=2,
        seed=7,
        nonlinear_inequality_constraints=((lambda x: x[0] - 0.2, True),),
    )
    assert candidates.shape[0] > 0
    assert values.shape == (candidates.shape[0], 2)
    assert bool((candidates[:, 0] >= 0.2).all())


def test_nsga2_rejects_interpoint_nonlinear_inequality() -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    constraints = CandidateConstraints(
        nonlinear_inequality_constraints=((lambda x: x.sum() - 0.2, False),)
    )
    with pytest.raises(NotImplementedError, match="intra-point"):
        optimize_vector_nsga2(
            lambda x: torch.cat((x, 1 - x), dim=-1),
            bounds,
            population_size=8,
            generations=1,
            constraints=constraints,
        )
