"""Full Objective-to-acquisition-optimization E2E contracts."""

import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.acquisition.objective import GenericMCObjective
from botorch.optim import optimize_acqf, optimize_acqf_mixed
from botorch.sampling.normal import SobolQMCNormalSampler

from robotorchan.models import ModelListGP, MixedSingleTaskGP, SingleTaskGP


def _two_output_model() -> tuple[ModelListGP, torch.Tensor, torch.Tensor]:
    train_x = torch.linspace(0.0, 1.0, 10, dtype=torch.double).unsqueeze(-1)
    train_y = torch.cat((torch.sin(3.0 * train_x), torch.cos(3.0 * train_x)), dim=-1)
    model = ModelListGP(
        SingleTaskGP(train_x, train_y[:, :1]),
        SingleTaskGP(train_x, train_y[:, 1:2]),
    )
    return model, train_x, train_y


def test_objective_acquisition_and_optimize_acqf_run_end_to_end() -> None:
    model, _, train_y = _two_output_model()
    weights = torch.tensor([0.7, 0.3], dtype=torch.double)
    objective = GenericMCObjective(lambda Y, X=None: Y @ weights)
    acquisition = qLogExpectedImprovement(
        model=model,
        best_f=(train_y @ weights).max(),
        sampler=SobolQMCNormalSampler(torch.Size([16]), seed=163),
        objective=objective,
    )

    candidate, value = optimize_acqf(
        acq_function=acquisition,
        bounds=torch.tensor([[0.1], [0.9]], dtype=torch.double),
        q=1,
        num_restarts=2,
        raw_samples=16,
        options={"maxiter": 20},
    )

    assert candidate.shape == torch.Size([1, 1])
    assert 0.1 <= candidate.item() <= 0.9
    assert torch.isfinite(candidate).all()
    assert torch.isfinite(value).all()


def test_objective_and_known_candidate_constraint_optimize_end_to_end() -> None:
    model, _, train_y = _two_output_model()
    weights = torch.tensor([0.7, 0.3], dtype=torch.double)
    objective = GenericMCObjective(lambda Y, X=None: Y @ weights)
    acquisition = qLogExpectedImprovement(
        model=model,
        best_f=(train_y @ weights).max(),
        sampler=SobolQMCNormalSampler(torch.Size([16]), seed=167),
        objective=objective,
    )
    # BoTorch inequality convention is sum(coefficients * X[indices]) >= rhs.
    constraint = (
        torch.tensor([0], dtype=torch.long),
        torch.tensor([1.0], dtype=torch.double),
        0.4,
    )

    candidate, value = optimize_acqf(
        acq_function=acquisition,
        bounds=torch.tensor([[0.1], [0.9]], dtype=torch.double),
        q=1,
        num_restarts=2,
        raw_samples=32,
        inequality_constraints=[constraint],
        options={"maxiter": 20},
    )

    assert candidate.shape == torch.Size([1, 1])
    assert candidate.item() >= 0.4 - 1e-6
    assert torch.isfinite(value).all()


def test_objective_and_mixed_search_optimize_end_to_end() -> None:
    train_x = torch.tensor(
        [
            [0.1, 0.0],
            [0.2, 1.0],
            [0.35, 0.0],
            [0.5, 1.0],
            [0.7, 0.0],
            [0.9, 1.0],
        ],
        dtype=torch.double,
    )
    first = torch.sin(3.0 * train_x[:, :1]) + 0.1 * train_x[:, 1:2]
    second = torch.cos(3.0 * train_x[:, :1]) - 0.1 * train_x[:, 1:2]
    train_y = torch.cat((first, second), dim=-1)
    model = MixedSingleTaskGP(train_x, train_y, cat_dims=[1])
    weights = torch.tensor([0.6, 0.4], dtype=torch.double)
    objective = GenericMCObjective(lambda Y, X=None: Y @ weights)
    acquisition = qLogExpectedImprovement(
        model=model,
        best_f=(train_y @ weights).max(),
        sampler=SobolQMCNormalSampler(torch.Size([16]), seed=173),
        objective=objective,
    )

    candidate, value = optimize_acqf_mixed(
        acq_function=acquisition,
        bounds=torch.tensor([[0.1, 0.0], [0.9, 1.0]], dtype=torch.double),
        q=1,
        num_restarts=2,
        raw_samples=16,
        fixed_features_list=[{1: 0.0}, {1: 1.0}],
        options={"maxiter": 20},
    )

    assert candidate.shape == torch.Size([1, 2])
    assert 0.1 <= candidate[0, 0].item() <= 0.9
    assert candidate[0, 1].item() in {0.0, 1.0}
    assert torch.isfinite(candidate).all()
    assert torch.isfinite(value).all()
