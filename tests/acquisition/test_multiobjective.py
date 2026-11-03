import pytest
import torch
from botorch.acquisition.multi_objective.logei import (
    qLogExpectedHypervolumeImprovement,
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.parego import qLogNParEGO
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)

from robotorchan.models import ModelListGP, SingleTaskGP
from robotorchan.optim.backends import (
    optimize_acqf_botorch,
    optimize_acqf_ga,
    optimize_acqf_pso,
    optimize_acqf_sampling,
)


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


@pytest.mark.parametrize("acquisition_name", ["qLogEHVI", "qLogNEHVI", "qLogNParEGO"])
@pytest.mark.parametrize("optimizer_name", ["botorch", "ga", "pso", "sobol"])
def test_multiobjective_acquisition_optimizer_q1_matrix(
    acquisition_name: str, optimizer_name: str
) -> None:
    model, train_x = _model()
    with torch.no_grad():
        train_y = model.posterior(train_x).mean
    ref_point = train_y.min(dim=0).values - 0.1
    if acquisition_name == "qLogEHVI":
        partitioning = FastNondominatedPartitioning(ref_point=ref_point, Y=train_y)
        acquisition = qLogExpectedHypervolumeImprovement(
            model=model, ref_point=ref_point.tolist(), partitioning=partitioning
        )
    elif acquisition_name == "qLogNEHVI":
        acquisition = qLogNoisyExpectedHypervolumeImprovement(
            model=model, ref_point=ref_point.tolist(), X_baseline=train_x, prune_baseline=False
        )
    else:
        acquisition = qLogNParEGO(
            model=model,
            X_baseline=train_x,
            scalarization_weights=torch.tensor([0.4, 0.6], dtype=torch.double),
        )
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    optimizers = {
        "botorch": (optimize_acqf_botorch, {"num_restarts": 2, "raw_samples": 16}),
        "ga": (optimize_acqf_ga, {"population_size": 16, "generations": 3, "seed": 3}),
        "pso": (optimize_acqf_pso, {"swarm_size": 16, "iterations": 3, "seed": 3}),
        "sobol": (optimize_acqf_sampling, {"num_samples": 32, "method": "sobol", "seed": 3}),
    }
    optimizer, kwargs = optimizers[optimizer_name]
    candidate, value = optimizer(acquisition, bounds, q=1, **kwargs)
    assert candidate.shape == (1, 1)
    assert torch.isfinite(candidate).all()
    assert torch.isfinite(value).all()
    assert torch.all(candidate >= bounds[0])
    assert torch.all(candidate <= bounds[1])
