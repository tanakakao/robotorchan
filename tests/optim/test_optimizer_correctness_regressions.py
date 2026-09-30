"""Cross-cutting optimizer correctness regressions found by E2E review."""

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from torch import Tensor

from robotorchan.optim.backends import optimize_acqf_cmaes, optimize_acqf_mixed_ga
from robotorchan.optim.constraints.evaluation import candidate_is_feasible
from robotorchan.optim.constraints import CandidateConstraints


class _PreferZero(AcquisitionFunction):
    def __init__(self) -> None:
        torch.nn.Module.__init__(self)

    def forward(self, X: Tensor) -> Tensor:
        return -(X.square()).sum(dim=(-1, -2))


def _lower_bound_constraint() -> CandidateConstraints:
    return CandidateConstraints(
        inequality_constraints=(
            (
                torch.tensor([0]),
                torch.tensor([1.0], dtype=torch.double),
                0.8,
            ),
        )
    )


def test_mixed_ga_returns_raw_acquisition_value_under_constraints() -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    acq = _PreferZero()
    constraints = _lower_bound_constraint()

    candidate, value = optimize_acqf_mixed_ga(
        acq,
        bounds,
        q=1,
        integer_dims=(0,),
        population_size=32,
        generations=8,
        seed=8,
        constraints=constraints,
    )

    expected = acq(candidate.unsqueeze(0)).reshape(())
    assert torch.allclose(value, expected)


def test_cmaes_returns_feasible_candidate_under_strong_penalty() -> None:
    try:
        import cmaes  # noqa: F401
    except ImportError:
        return

    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    constraints = _lower_bound_constraint()
    candidate, value = optimize_acqf_cmaes(
        _PreferZero(),
        bounds,
        q=1,
        population_size=16,
        max_generations=20,
        seed=5,
        constraints=constraints,
        constraint_penalty=1e6,
    )

    assert candidate_is_feasible(candidate.unsqueeze(0), constraints).item()
    assert torch.allclose(value, _PreferZero()(candidate.unsqueeze(0)).reshape(()))
