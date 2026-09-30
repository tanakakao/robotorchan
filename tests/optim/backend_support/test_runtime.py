"""Runtime contract tests for optimizer backends."""

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from torch import Tensor

from robotorchan.optim.backend_support.runtime import validate_bounds
from robotorchan.optim.backends import (
    optimize_acqf_ga,
    optimize_acqf_pso,
    optimize_acqf_sampling,
    optimize_vector_nsga2,
)


class _Quadratic(AcquisitionFunction):
    def __init__(self) -> None:
        torch.nn.Module.__init__(self)

    def forward(self, X: Tensor) -> Tensor:
        return -((X - 0.4) ** 2).sum(dim=(-1, -2))


def test_validate_bounds_rejects_integer_nonfinite_and_degenerate_bounds() -> None:
    for bounds, error in [
        (torch.tensor([[0], [1]]), TypeError),
        (torch.tensor([[0.0], [float("inf")]]), ValueError),
        (torch.tensor([[1.0], [1.0]]), ValueError),
    ]:
        try:
            validate_bounds(bounds)
        except error:
            pass
        else:
            raise AssertionError(f"Expected {error.__name__}.")


def test_seeded_torch_backends_do_not_mutate_global_rng() -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    optimizers = [
        lambda: optimize_acqf_sampling(_Quadratic(), bounds, 1, num_samples=16, seed=7),
        lambda: optimize_acqf_ga(
            _Quadratic(), bounds, 1, population_size=16, generations=3, seed=7
        ),
        lambda: optimize_acqf_pso(_Quadratic(), bounds, 1, swarm_size=16, iterations=3, seed=7),
    ]
    for optimize in optimizers:
        torch.manual_seed(123)
        expected = torch.rand(4)
        torch.manual_seed(123)
        optimize()
        actual = torch.rand(4)
        assert torch.equal(actual, expected)


def test_seeded_backends_are_reproducible() -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    first = optimize_acqf_sampling(_Quadratic(), bounds, 2, num_samples=32, seed=5)
    second = optimize_acqf_sampling(_Quadratic(), bounds, 2, num_samples=32, seed=5)
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])


def test_nsga2_preserves_float64_dtype() -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)

    def objective(X: Tensor) -> Tensor:
        return torch.cat((-(X - 0.2).square(), -(X - 0.8).square()), dim=1)

    candidates, values = optimize_vector_nsga2(
        objective, bounds, population_size=16, generations=3, seed=9
    )
    assert candidates.dtype == torch.double
    assert values.dtype == torch.double
    assert candidates.device == bounds.device
