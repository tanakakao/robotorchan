"""Tests for candidate-constraint evaluation and derivative-free optimizers."""

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from torch import Tensor

from robotorchan.optim.backends import optimize_acqf_de, optimize_acqf_ga, optimize_acqf_mixed_ga
from robotorchan.optim.constraint_evaluation import (
    candidate_constraint_violation,
    candidate_is_feasible,
    feasibility_first_ranks,
)
from robotorchan.optim.constraints import CandidateConstraints


class _Target(AcquisitionFunction):
    def __init__(self, target: float = 0.9) -> None:
        torch.nn.Module.__init__(self)
        self.target = target

    def forward(self, X: Tensor) -> Tensor:
        return -((X - self.target) ** 2).sum(dim=(-2, -1))


def test_constraint_evaluation_supports_linear_and_nonlinear_constraints() -> None:
    candidates = torch.tensor([[[0.8, 0.3]], [[0.2, 0.3]]], dtype=torch.double)
    constraints = CandidateConstraints(
        inequality_constraints=((torch.tensor([0]), torch.tensor([1.0]), 0.5),),
        nonlinear_inequality_constraints=((lambda x: 0.5 - x[..., 1], True),),
    )

    violation = candidate_constraint_violation(candidates, constraints)

    assert torch.equal(candidate_is_feasible(candidates, constraints), torch.tensor([True, False]))
    assert violation[0] == 0
    assert violation[1] > 0


def test_constraint_evaluation_supports_interpoint_linear_constraint() -> None:
    candidates = torch.tensor([[[0.2], [0.9]], [[0.8], [0.9]]], dtype=torch.double)
    constraints = CandidateConstraints(
        inequality_constraints=(
            (
                torch.tensor([[0, 0], [1, 0]]),
                torch.tensor([1.0, -1.0]),
                -0.3,
            ),
        )
    )

    feasible = candidate_is_feasible(candidates, constraints)

    assert torch.equal(feasible, torch.tensor([False, True]))


def test_ga_returns_feasible_candidate_with_linear_constraint() -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    constraints = CandidateConstraints(
        inequality_constraints=((torch.tensor([0]), torch.tensor([-1.0]), -0.4),)
    )

    candidates, _ = optimize_acqf_ga(
        _Target(), bounds, q=1, population_size=80, generations=40, seed=4, constraints=constraints
    )

    assert float(candidates[0, 0]) <= 0.4 + 1e-6


def test_de_returns_feasible_candidate_with_nonlinear_constraint() -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    constraints = CandidateConstraints(
        nonlinear_inequality_constraints=((lambda x: 0.4 - x[..., 0], True),)
    )

    candidates, _ = optimize_acqf_de(
        _Target(),
        bounds,
        q=1,
        seed=3,
        options={"maxiter": 30, "popsize": 8},
        constraints=constraints,
    )

    assert float(candidates[0, 0]) <= 0.4 + 1e-3


def test_mixed_ga_preserves_domain_while_enforcing_constraint() -> None:
    bounds = torch.tensor([[0.0, 0.0], [1.0, 4.0]], dtype=torch.double)
    constraints = CandidateConstraints(
        inequality_constraints=((torch.tensor([0]), torch.tensor([-1.0]), -0.4),)
    )

    candidates, _ = optimize_acqf_mixed_ga(
        _Target(),
        bounds,
        q=1,
        integer_dims=[1],
        population_size=80,
        generations=30,
        seed=8,
        constraints=constraints,
    )

    assert float(candidates[0, 0]) <= 0.4 + 1e-6
    assert candidates[0, 1] == candidates[0, 1].round()


def test_feasibility_first_ranks_prioritize_feasibility_over_acquisition_scale() -> None:
    values = torch.tensor([1.0e12, 3.0, 2.0, -1.0e12], dtype=torch.double)
    violation = torch.tensor([0.2, 0.0, 0.0, 0.1], dtype=torch.double)

    ranks = feasibility_first_ranks(values, violation)

    assert ranks[1] > ranks[2]
    assert ranks[2] > ranks[3]
    assert ranks[3] > ranks[0]


def test_mixed_ga_feasibility_first_is_independent_of_acquisition_scale() -> None:
    class _HugeTarget(AcquisitionFunction):
        def __init__(self) -> None:
            torch.nn.Module.__init__(self)

        def forward(self, X: Tensor) -> Tensor:
            return 1.0e12 * X[..., 0, 0]

    bounds = torch.tensor([[0.0, 0.0], [1.0, 4.0]], dtype=torch.double)
    constraints = CandidateConstraints(
        inequality_constraints=((torch.tensor([0]), torch.tensor([-1.0]), -0.4),)
    )

    candidates, _ = optimize_acqf_mixed_ga(
        _HugeTarget(),
        bounds,
        q=1,
        integer_dims=[1],
        population_size=80,
        generations=30,
        seed=8,
        constraints=constraints,
    )

    assert float(candidates[0, 0]) <= 0.4 + 1e-6
    assert candidates[0, 1] == candidates[0, 1].round()


def test_mixed_ga_supports_variable_dependent_nonlinear_constraint() -> None:
    bounds = torch.tensor([[0.0, 0.0, 0.0], [1.0, 4.0, 2.0]], dtype=torch.double)
    constraints = CandidateConstraints(
        nonlinear_inequality_constraints=(
            (
                lambda x: 0.55 + 0.10 * x[..., 1] - 0.15 * x[..., 2] - x[..., 0],
                True,
            ),
        )
    )

    candidates, _ = optimize_acqf_mixed_ga(
        _Target(target=0.9),
        bounds,
        q=1,
        integer_dims=[1],
        categorical_values={2: [0.0, 1.0, 2.0]},
        population_size=120,
        generations=50,
        seed=17,
        constraints=constraints,
    )

    residual = 0.55 + 0.10 * candidates[0, 1] - 0.15 * candidates[0, 2] - candidates[0, 0]
    assert residual >= -1e-8
    assert candidates[0, 1] == candidates[0, 1].round()
    assert float(candidates[0, 2]) in {0.0, 1.0, 2.0}


def test_mixed_ga_supports_q_batch_interpoint_nonlinear_constraint() -> None:
    bounds = torch.tensor([[0.0, 0.0], [1.0, 3.0]], dtype=torch.double)
    constraints = CandidateConstraints(
        nonlinear_inequality_constraints=((lambda x: 0.7 - x[:, 0].sum(), False),)
    )

    candidates, _ = optimize_acqf_mixed_ga(
        _Target(target=0.9),
        bounds,
        q=2,
        integer_dims=[1],
        population_size=140,
        generations=60,
        seed=23,
        constraints=constraints,
    )

    assert candidates[:, 0].sum() <= 0.7 + 1e-8
    assert torch.equal(candidates[:, 1], candidates[:, 1].round())


def test_mixed_ga_supports_q_batch_interpoint_linear_constraint() -> None:
    bounds = torch.tensor([[0.0, 0.0], [1.0, 3.0]], dtype=torch.double)
    constraints = CandidateConstraints(
        inequality_constraints=(
            (
                torch.tensor([[0, 0], [1, 0]]),
                torch.tensor([-1.0, -1.0]),
                -0.7,
            ),
        )
    )

    candidates, _ = optimize_acqf_mixed_ga(
        _Target(target=0.9),
        bounds,
        q=2,
        integer_dims=[1],
        population_size=140,
        generations=60,
        seed=29,
        constraints=constraints,
    )

    assert candidates[:, 0].sum() <= 0.7 + 1e-8
    assert torch.equal(candidates[:, 1], candidates[:, 1].round())
