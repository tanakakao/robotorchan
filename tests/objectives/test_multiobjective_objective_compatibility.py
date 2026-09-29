"""Multi-objective Objective contracts for native BoTorch acquisitions."""

import pytest
import torch
from botorch.acquisition.multi_objective.logei import (
    qLogExpectedHypervolumeImprovement,
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
from botorch.acquisition.multi_objective.parego import qLogNParEGO
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)

from robotorchan.models import ModelListGP, SingleTaskGP


def _three_output_model() -> tuple[ModelListGP, torch.Tensor, torch.Tensor]:
    train_X = torch.linspace(0.0, 1.0, 8, dtype=torch.double).unsqueeze(-1)
    train_Y = torch.cat(
        (
            torch.sin(4.0 * train_X),
            torch.cos(4.0 * train_X),
            0.5 * train_X,
        ),
        dim=-1,
    )
    model = ModelListGP(
        SingleTaskGP(train_X, train_Y[:, :1]),
        SingleTaskGP(train_X, train_Y[:, 1:2]),
        SingleTaskGP(train_X, train_Y[:, 2:3]),
    )
    return model, train_X, train_Y


def test_qlogehvi_output_subset_uses_objective_space_reference_point() -> None:
    model, _, train_Y = _three_output_model()
    objective = IdentityMCMultiOutputObjective(outcomes=[2, 0])
    objective_Y = objective(train_Y)
    ref_point = objective_Y.min(dim=0).values - 0.1
    partitioning = FastNondominatedPartitioning(ref_point=ref_point, Y=objective_Y)
    acquisition = qLogExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point.tolist(),
        partitioning=partitioning,
        sampler=SobolQMCNormalSampler(torch.Size([16]), seed=61),
        objective=objective,
    )
    X = torch.tensor([[[0.4]]], dtype=torch.double)

    value = acquisition(X)

    assert objective_Y.shape[-1] == 2
    assert ref_point.numel() == 2
    assert value.shape == torch.Size([1])
    assert torch.isfinite(value).all()


def test_qlognehvi_output_subset_uses_objective_space_reference_point() -> None:
    model, train_X, train_Y = _three_output_model()
    objective = IdentityMCMultiOutputObjective(outcomes=[1, 2])
    objective_Y = objective(train_Y)
    ref_point = objective_Y.min(dim=0).values - 0.1
    acquisition = qLogNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point.tolist(),
        X_baseline=train_X,
        sampler=SobolQMCNormalSampler(torch.Size([16]), seed=67),
        objective=objective,
        prune_baseline=False,
    )
    X = torch.tensor([[[0.6]]], dtype=torch.double)

    value = acquisition(X)

    assert ref_point.numel() == objective_Y.shape[-1] == 2
    assert value.shape == torch.Size([1])
    assert torch.isfinite(value).all()


def test_qlogehvi_rejects_reference_point_dimension_outside_objective_space() -> None:
    model, _, train_Y = _three_output_model()
    objective = IdentityMCMultiOutputObjective(outcomes=[2, 0])
    objective_Y = objective(train_Y)
    partitioning = FastNondominatedPartitioning(
        ref_point=objective_Y.min(dim=0).values - 0.1,
        Y=objective_Y,
    )

    with pytest.raises(ValueError):
        qLogExpectedHypervolumeImprovement(
            model=model,
            ref_point=[-1.0, -1.0, -1.0],
            partitioning=partitioning,
            objective=objective,
        )


def test_qlognparego_owns_scalarization_of_the_multi_output_posterior() -> None:
    model, train_X, _ = _three_output_model()
    acquisition = qLogNParEGO(
        model=model,
        X_baseline=train_X,
        scalarization_weights=torch.tensor([0.2, 0.3, 0.5], dtype=torch.double),
        sampler=SobolQMCNormalSampler(torch.Size([16]), seed=71),
    )
    X = torch.tensor([[[0.5]]], dtype=torch.double)

    value = acquisition(X)

    assert value.shape == torch.Size([1])
    assert torch.isfinite(value).all()
