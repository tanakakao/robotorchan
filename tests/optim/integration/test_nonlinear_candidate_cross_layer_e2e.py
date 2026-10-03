"""Cross-layer E2E contracts for nonlinear candidate constraints."""

import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.acquisition.multi_objective.parego import qLogNParEGO
from botorch.sampling.normal import SobolQMCNormalSampler

from robotorchan.models import ModelListGP, SingleTaskGP
from robotorchan.optim import CandidateConstraints, optimize_acqf


def _single_output_problem() -> tuple[SingleTaskGP, torch.Tensor, torch.Tensor]:
    train_x = torch.linspace(0.0, 1.0, 10, dtype=torch.double).unsqueeze(-1)
    train_y = 1.0 - (train_x - 0.72).square()
    model = SingleTaskGP(train_x, train_y)
    model.eval()
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    return model, train_y, bounds


def test_public_optimizer_runs_model_acquisition_and_nonlinear_candidate_constraint() -> None:
    model, train_y, bounds = _single_output_problem()
    acquisition = qLogExpectedImprovement(
        model=model,
        best_f=train_y.max(),
        sampler=SobolQMCNormalSampler(torch.Size([32]), seed=2201),
    )

    def upper_limit(x: torch.Tensor) -> torch.Tensor:
        return x.new_tensor(0.65) - x[0]

    constraints = CandidateConstraints(
        nonlinear_inequality_constraints=((upper_limit, True),),
    )
    initial_conditions = torch.tensor(
        [[[0.2]], [[0.4]], [[0.6]]],
        dtype=torch.double,
    )

    candidate, value = optimize_acqf(
        acquisition,
        bounds,
        q=1,
        num_restarts=3,
        raw_samples=None,
        constraints=constraints,
        batch_initial_conditions=initial_conditions,
    )

    assert candidate.shape == torch.Size([1, 1])
    assert torch.isfinite(candidate).all()
    assert torch.isfinite(value).all()
    assert upper_limit(candidate[0]) >= -1e-6


def test_public_optimizer_preserves_joint_q_interpoint_nonlinear_feasibility() -> None:
    model, train_y, bounds = _single_output_problem()
    acquisition = qLogExpectedImprovement(
        model=model,
        best_f=train_y.max(),
        sampler=SobolQMCNormalSampler(torch.Size([32]), seed=2202),
    )

    def upper_limit(x: torch.Tensor) -> torch.Tensor:
        return x.new_tensor(0.9) - x[0]

    def separation(X: torch.Tensor) -> torch.Tensor:
        return (X[0] - X[1]).square().sum() - X.new_tensor(0.2**2)

    constraints = CandidateConstraints(
        nonlinear_inequality_constraints=(
            (upper_limit, True),
            (separation, False),
        ),
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
        bounds,
        q=2,
        num_restarts=3,
        raw_samples=None,
        constraints=constraints,
        batch_initial_conditions=initial_conditions,
    )

    assert candidate.shape == torch.Size([2, 1])
    assert torch.isfinite(candidate).all()
    assert torch.isfinite(value).all()
    assert torch.all(candidate[:, 0] <= 0.9 + 1e-6)
    assert separation(candidate) >= -1e-6


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
        sampler=SobolQMCNormalSampler(torch.Size([32]), seed=2203),
    )

    def candidate_constraint(x: torch.Tensor) -> torch.Tensor:
        return x.new_tensor(0.8) - x[0]

    constraints = CandidateConstraints(
        nonlinear_inequality_constraints=((candidate_constraint, True),),
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
    assert torch.isfinite(candidate).all()
    assert torch.isfinite(value).all()
    assert candidate_constraint(candidate[0]) >= -1e-6
