from __future__ import annotations

import torch
from botorch.acquisition.logei import qLogExpectedImprovement, qLogNoisyExpectedImprovement
from botorch.acquisition.monte_carlo import (
    qExpectedImprovement,
    qNoisyExpectedImprovement,
    qUpperConfidenceBound,
)
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
from botorch.acquisition.objective import GenericMCObjective
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.non_dominated import NondominatedPartitioning

from robotorchan.models import ReducedGP
from robotorchan.reduction import OutputPCAReducer, PCAInputReducer


def _training_data() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(31)
    train_X = torch.rand(20, 6, dtype=torch.double)
    x0, x1, x2, x3, x4, x5 = train_X.unbind(dim=-1)
    train_Y = torch.stack(
        [
            torch.sin(2.0 * torch.pi * x0) + 0.2 * x1,
            torch.cos(2.0 * torch.pi * x1) - 0.1 * x2,
            x0 + x2.square(),
            x3 - 0.5 * x4,
            1.0 - (x4 - 0.3).square() - 0.2 * x5,
            0.5 + x5 - 0.4 * x0,
        ],
        dim=-1,
    )
    return train_X, train_Y


def _combined_model() -> tuple[ReducedGP, torch.Tensor, torch.Tensor]:
    train_X, train_Y = _training_data()
    model = ReducedGP(
        train_X=train_X,
        train_Y=train_Y,
        input_reducer=PCAInputReducer(n_components=4),
        output_reducer=OutputPCAReducer(n_components=3),
    )
    model.eval()
    model.likelihood.eval()
    return model, train_X, train_Y


def _sampler() -> SobolQMCNormalSampler:
    return SobolQMCNormalSampler(sample_shape=torch.Size([16]), seed=7)


def _original_space_scalar_objective() -> GenericMCObjective:
    return GenericMCObjective(lambda samples, X=None: samples[..., 4])


def test_scalar_mc_acquisitions_use_restored_original_outputs() -> None:
    model, train_X, train_Y = _combined_model()
    objective = _original_space_scalar_objective()
    sampler = _sampler()
    best_f = train_Y[:, 4].max()
    X = torch.rand(2, 6, dtype=torch.double)

    acquisitions = [
        qExpectedImprovement(
            model=model,
            best_f=best_f,
            sampler=sampler,
            objective=objective,
        ),
        qLogExpectedImprovement(
            model=model,
            best_f=best_f,
            sampler=sampler,
            objective=objective,
        ),
        qUpperConfidenceBound(
            model=model,
            beta=0.2,
            sampler=sampler,
            objective=objective,
        ),
        qNoisyExpectedImprovement(
            model=model,
            X_baseline=train_X,
            sampler=sampler,
            objective=objective,
            prune_baseline=False,
            cache_root=False,
        ),
        qLogNoisyExpectedImprovement(
            model=model,
            X_baseline=train_X,
            sampler=sampler,
            objective=objective,
            prune_baseline=False,
            cache_root=False,
        ),
    ]

    for acquisition in acquisitions:
        value = acquisition(X)
        assert value.numel() == 1
        assert torch.isfinite(value).all()


def test_scalar_acquisition_gradient_flows_to_original_input_space() -> None:
    model, _, train_Y = _combined_model()
    objective = _original_space_scalar_objective()
    acquisition = qLogExpectedImprovement(
        model=model,
        best_f=train_Y[:, 4].max(),
        sampler=_sampler(),
        objective=objective,
    )
    X = torch.rand(2, 6, dtype=torch.double, requires_grad=True)

    value = acquisition(X)
    value.sum().backward()

    assert X.grad is not None
    assert X.grad.shape == X.shape
    assert torch.isfinite(X.grad).all()


def test_multiobjective_acquisitions_use_original_output_indices() -> None:
    model, train_X, train_Y = _combined_model()
    objective = IdentityMCMultiOutputObjective(outcomes=[4, 5])
    ref_point = train_Y[:, [4, 5]].min(dim=0).values - 0.1
    partitioning = NondominatedPartitioning(
        ref_point=ref_point,
        Y=train_Y[:, [4, 5]],
    )
    sampler = _sampler()
    X = torch.rand(2, 6, dtype=torch.double)

    qehvi = qExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point,
        partitioning=partitioning,
        sampler=sampler,
        objective=objective,
    )
    qnehvi = qNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point,
        X_baseline=train_X,
        sampler=sampler,
        objective=objective,
        prune_baseline=False,
        cache_root=False,
    )

    for acquisition in (qehvi, qnehvi):
        value = acquisition(X)
        assert value.numel() == 1
        assert torch.isfinite(value).all()


def test_multiobjective_acquisition_gradient_flows_through_both_reducers() -> None:
    model, _, train_Y = _combined_model()
    objective = IdentityMCMultiOutputObjective(outcomes=[4, 5])
    ref_point = train_Y[:, [4, 5]].min(dim=0).values - 0.1
    partitioning = NondominatedPartitioning(
        ref_point=ref_point,
        Y=train_Y[:, [4, 5]],
    )
    acquisition = qExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point,
        partitioning=partitioning,
        sampler=_sampler(),
        objective=objective,
    )
    X = torch.rand(2, 6, dtype=torch.double, requires_grad=True)

    value = acquisition(X)
    value.sum().backward()

    assert X.grad is not None
    assert X.grad.shape == X.shape
    assert torch.isfinite(X.grad).all()
