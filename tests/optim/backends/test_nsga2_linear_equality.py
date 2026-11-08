"""Focused linear-equality coverage for continuous and mixed NSGA-II."""

import torch

from robotorchan.optim.backends.nsga2 import optimize_vector_nsga2


def test_nsga2_linear_equality_projects_population() -> None:
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)

    def objective(x: torch.Tensor) -> torch.Tensor:
        return torch.stack((x[:, 0], x[:, 1]), dim=-1)

    candidates, values = optimize_vector_nsga2(
        objective,
        bounds,
        population_size=16,
        generations=2,
        seed=3,
        equality_constraints=((torch.tensor([0, 1]), torch.tensor([1.0, 1.0]), 1.0),),
    )
    assert candidates.shape[0] > 0
    assert values.shape == (candidates.shape[0], 2)
    torch.testing.assert_close(
        candidates.sum(dim=-1),
        torch.ones(candidates.shape[0], dtype=bounds.dtype),
        atol=1e-6,
        rtol=0,
    )
