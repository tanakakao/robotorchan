"""Cross-layer E2E tests for nonlinear candidate constraints."""

import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.acquisition.multi_objective.parego import qLogNParEGO

from robotorchan.models import ModelListGP, SingleTaskGP
from robotorchan.optim import CandidateConstraints, optimize_acqf


def _single_objective_problem() -> tuple[SingleTaskGP, torch.Tensor, torch.Tensor]:
    train_x = torch.linspace(0.0, 1.0, 10, dtype=torch.double).unsqueeze(-1)
    train_y = 1.0 - (train_x - 0.72).square()
    model = SingleTaskGP(train_x, train_y)
    model.eval()
    return model, train_x, train_y


def test_public_optimizer_returns_feasible_single_objective_candidate() -> None:
    model, _, train_y = _single_objective_problem()
    acquisition = qLogExpectedImprovement(model=model, best_f=train_y.max())

    def nonlinear_constraint(x: torch.Tensor) -> torch.Tensor:
        return x.new_tensor(0.65) - x[0]

    constraints = CandidateConstraints(
        nonlinear_inequality_constraints=((nonlinear_constraint, True),),
    )
    initial_conditions = torch.tensor(
        [[[0.2]], [[0.4]], [[0.6]]],
        dtype=torch.double,
    )

    candidate, value = optimize_acqf(
        acquisition,
        torch.tensor([[0.0], [1.0]], dtype=torch.double),
        q=1,
        num_restarts=3,
        raw_samples=None,
        constraints=constraints,
        batch_initial_conditions=initial_conditions,
    )

    assert candidate.shape == torch.Size([1, 1])
    assert torch.isfinite(value).all()
    assert nonlinear_constraint(candidate[0]) >= -1e-6


def test_public_optimizer_composes_linear_nonlinear_and_fixed_features() -> None:
    train_x = torch.rand(12, 2, dtype=torch.double)
    train_y = 1.0 - (train_x[:, :1] - 0.6).square() - (train_x[:, 1:] - 0.4).square()
    model = SingleTaskGP(train_x, train_y)
    model.eval()
    acquisition = qLogExpectedImprovement(model=model, best_f=train_y.max())

    def nonlinear_constraint(x: torch.Tensor) -> torch.Tensor:
        return x.new_tensor(0.8) - x[0]

    constraints = CandidateConstraints(
        inequality_constraints=(
            (
                torch.tensor([0]),
                torch.tensor([1.0], dtype=torch.double),
                0.25,
            ),
        ),
        nonlinear_inequality_constraints=((nonlinear_constraint, True),),
    )
    initial_conditions = torch.tensor(
        [[[0.3, 0.4]], [[0.5, 0.4]], [[0.7, 0.4]]],
        dtype=torch.double,
    )

    candidate, value = optimize_acqf(
        acquisition,
        torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double),
        q=1,
        num_restarts=3,
        raw_samples=None,
        fixed_features={1: 0.4},
        constraints=constraints,
        batch_initial_conditions=initial_conditions,
    )

    assert torch.isfinite(value).all()
    assert candidate[0, 0] >= 0.25 - 1e-6
    assert nonlinear_constraint(candidate[0]) >= -1e-6
    torch.testing.assert_close(candidate[0, 1], torch.tensor(0.4, dtype=torch.double))


def test_public_optimizer_preserves_joint_interpoint_q_semantics() -> None:
    model, _, train_y = _single_objective_problem()
    acquisition = qLogExpectedImprovement(model=model, best_f=train_y.max())

    def separation_constraint(X: torch.Tensor) -> torch.Tensor:
        return (X[0] - X[1]).square().sum() - X.new_tensor(0.04)

    constraints = CandidateConstraints(
        nonlinear_inequality_constraints=((separation_constraint, False),),
    )
    initial_conditions = torch.tensor(
        [
            [[0.2], [0.6]],
            [[0.3], [0.7]],
            [[0.4], [0.8]],
        ],
        dtype=torch.double,
    )

    candidate, value = optimize_acqf(
        acquisition,
        torch.tensor([[0.0], [1.0]], dtype=torch.double),
        q=2,
        num_restarts=3,
        raw_samples=None,
        constraints=constraints,
        batch_initial_conditions=initial_conditions,
    )

    assert candidate.shape == torch.Size([2, 1])
    assert torch.isfinite(value).all()
    assert separation_constraint(candidate) >= -1e-6


def test_qlognparego_runs_end_to_end_with_nonlinear_candidate_constraint() -> None:
    train_x = torch.linspace(0.0, 1.0, 10, dtype=torch.double).unsqueeze(-1)
    first = 1.0 - (train_x - 0.25).square()
    second = 1.0 - (train_x - 0.75).square()
    model = ModelListGP(
        SingleTaskGP(train_x, first),
        SingleTaskGP(train_x, second),
    )
    model.eval()
    acquisition = qLogNParEGO(
        model=model,
        X_baseline=train_x,
        scalarization_weights=torch.tensor([0.4, 0.6], dtype=torch.double),
    )

    def nonlinear_constraint(x: torch.Tensor) -> torch.Tensor:
        return x.new_tensor(0.8) - x[0]

    constraints = CandidateConstraints(
        nonlinear_inequality_constraints=((nonlinear_constraint, True),),
    )
    initial_conditions = torch.tensor(
        [[[0.2]], [[0.5]], [[0.7]]],
        dtype=torch.double,
    )

    candidate, value = optimize_acqf(
        acquisition,
        torch.tensor([[0.0], [1.0]], dtype=torch.double),
        q=1,
        num_restarts=3,
        raw_samples=None,
        constraints=constraints,
        batch_initial_conditions=initial_conditions,
    )

    assert candidate.shape == torch.Size([1, 1])
    assert torch.isfinite(value).all()
    assert nonlinear_constraint(candidate[0]) >= -1e-6
