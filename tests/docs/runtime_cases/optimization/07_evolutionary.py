"""Runtime inputs retained when usage examples were removed from Guide."""

import torch

from robotorchan.optim.backends.nsga2 import optimize_vector_nsga2


def objectives(x: torch.Tensor) -> torch.Tensor:
    return torch.cat((-((x - 0.2) ** 2), -((x - 0.8) ** 2)), dim=-1)


bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
pareto_x, pareto_y = optimize_vector_nsga2(
    objectives, bounds, population_size=16, generations=5, seed=7
)
assert pareto_x.ndim == 2 and pareto_x.shape[1] == 1
assert pareto_y.shape == (pareto_x.shape[0], 2)
assert torch.all(pareto_x >= bounds[0])
assert torch.all(pareto_x <= bounds[1])
assert torch.isfinite(pareto_y).all()
