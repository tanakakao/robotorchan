"""Sampling E2E contracts for multi-objective and outcome-constrained MC acquisition."""

import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.acquisition.multi_objective.logei import qLogExpectedHypervolumeImprovement
from botorch.acquisition.objective import GenericMCObjective
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)

from robotorchan.models import ModelListGP, SingleTaskGP
from robotorchan.optim.backends import optimize_acqf_botorch


def _multi_output_model() -> tuple[ModelListGP, torch.Tensor]:
    train_x = torch.linspace(0.0, 1.0, 10, dtype=torch.double).unsqueeze(-1)
    first = -(train_x - 0.25).square() + 1.0
    second = -(train_x - 0.75).square() + 1.0
    model = ModelListGP(
        SingleTaskGP(train_x, first),
        SingleTaskGP(train_x, second),
    )
    model.eval()
    return model, train_x


def test_joint_sampling_qlogehvi_runs_through_optimizer() -> None:
    model, train_x = _multi_output_model()
    with torch.no_grad():
        train_y = model.posterior(train_x).mean
    ref_point = train_y.min(dim=0).values - 0.1
    partitioning = FastNondominatedPartitioning(ref_point=ref_point, Y=train_y)
    sampler = SobolQMCNormalSampler(torch.Size([32]), seed=123)
    acquisition = qLogExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point.tolist(),
        partitioning=partitioning,
        sampler=sampler,
    )
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)

    candidate, value = optimize_acqf_botorch(
        acquisition,
        bounds,
        q=2,
        num_restarts=3,
        raw_samples=32,
    )

    assert candidate.shape == torch.Size([2, 1])
    assert value.numel() == 1
    assert torch.isfinite(candidate).all()
    assert torch.isfinite(value).all()


def test_outcome_constraint_sampling_runs_through_optimizer() -> None:
    train_x = torch.linspace(0.0, 1.0, 10, dtype=torch.double).unsqueeze(-1)
    objective_y = -(train_x - 0.7).square() + 1.0
    constraint_y = train_x - 0.8
    model = ModelListGP(
        SingleTaskGP(train_x, objective_y),
        SingleTaskGP(train_x, constraint_y),
    )
    model.eval()
    sampler = SobolQMCNormalSampler(torch.Size([32]), seed=456)
    objective = GenericMCObjective(lambda samples, X=None: samples[..., 0])
    acquisition = qLogExpectedImprovement(
        model=model,
        best_f=objective_y.max(),
        sampler=sampler,
        objective=objective,
        constraints=[lambda samples: samples[..., 1]],
    )
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)

    candidate, value = optimize_acqf_botorch(
        acquisition,
        bounds,
        q=1,
        num_restarts=3,
        raw_samples=32,
    )

    assert candidate.shape == torch.Size([1, 1])
    assert value.numel() == 1
    assert torch.isfinite(candidate).all()
    assert torch.isfinite(value).all()
