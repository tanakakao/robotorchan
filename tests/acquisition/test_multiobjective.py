import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.acquisition.multi_objective.logei import (
    qLogExpectedHypervolumeImprovement,
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.parego import qLogNParEGO
from botorch.acquisition.multi_objective.utils import get_default_partitioning_alpha
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)

from robotorchan.models import ModelListGP, SingleTaskGP


def _model() -> tuple[ModelListGP, torch.Tensor]:
    train_X = torch.linspace(0.0, 1.0, 8, dtype=torch.double).unsqueeze(-1)
    y1 = torch.sin(train_X * 5.0)
    y2 = torch.cos(train_X * 5.0)
    return ModelListGP(SingleTaskGP(train_X, y1), SingleTaskGP(train_X, y2)), train_X


def test_native_qlogehvi_is_compatible() -> None:
    model, train_X = _model()
    with torch.no_grad():
        train_Y = model.posterior(train_X).mean
    ref_point = train_Y.min(dim=0).values - 0.1
    partitioning = FastNondominatedPartitioning(ref_point=ref_point, Y=train_Y)
    acquisition = qLogExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point.tolist(),
        partitioning=partitioning,
    )
    X = torch.tensor([[[0.4]]], dtype=torch.double)

    value = acquisition(X)

    assert value.shape == torch.Size([1])
    assert torch.isfinite(value).all()


def test_native_qlognehvi_is_compatible() -> None:
    model, train_X = _model()
    with torch.no_grad():
        train_Y = model.posterior(train_X).mean
    ref_point = train_Y.min(dim=0).values - 0.1
    acquisition = qLogNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point.tolist(),
        X_baseline=train_X,
        prune_baseline=False,
    )
    X = torch.tensor([[[0.4]]], dtype=torch.double)

    value = acquisition(X)

    assert value.shape == torch.Size([1])
    assert torch.isfinite(value).all()


def test_native_qlognparego_is_compatible() -> None:
    model, train_X = _model()
    acquisition = qLogNParEGO(
        model=model,
        X_baseline=train_X,
        scalarization_weights=torch.tensor([0.4, 0.6], dtype=torch.double),
    )
    X = torch.tensor([[[0.4]]], dtype=torch.double)

    value = acquisition(X)

    assert value.shape == torch.Size([1])
    assert torch.isfinite(value).all()
