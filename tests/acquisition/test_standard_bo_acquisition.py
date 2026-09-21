import torch
from botorch.acquisition.analytic import (
    LogExpectedImprovement,
    LogProbabilityOfImprovement,
    UpperConfidenceBound,
)
from botorch.acquisition.logei import (
    qLogExpectedImprovement,
    qLogNoisyExpectedImprovement,
)
from botorch.acquisition.monte_carlo import qSimpleRegret, qUpperConfidenceBound
from botorch.sampling.normal import SobolQMCNormalSampler

from robotorchan.models import SingleTaskGP


def _model() -> tuple[SingleTaskGP, torch.Tensor, torch.Tensor]:
    train_X = torch.linspace(0.0, 1.0, 8, dtype=torch.double).unsqueeze(-1)
    train_Y = torch.sin(train_X * 5.0)
    return SingleTaskGP(train_X, train_Y), train_X, train_Y


def test_native_analytic_standard_bo_acquisitions() -> None:
    model, _, train_Y = _model()
    X = torch.tensor([[0.35]], dtype=torch.double)
    acquisitions = [
        LogExpectedImprovement(model=model, best_f=train_Y.max()),
        LogProbabilityOfImprovement(model=model, best_f=train_Y.max()),
        UpperConfidenceBound(model=model, beta=0.2),
    ]

    for acquisition in acquisitions:
        value = acquisition(X)
        assert value.shape == torch.Size([1])
        assert torch.isfinite(value).all()


def test_native_mc_standard_bo_acquisitions() -> None:
    model, train_X, train_Y = _model()
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([16]), seed=0)
    X = torch.tensor([[[0.25], [0.75]]], dtype=torch.double)
    acquisitions = [
        qLogExpectedImprovement(model=model, best_f=train_Y.max(), sampler=sampler),
        qLogNoisyExpectedImprovement(model=model, X_baseline=train_X, sampler=sampler),
        qUpperConfidenceBound(model=model, beta=0.2, sampler=sampler),
        qSimpleRegret(model=model, sampler=sampler),
    ]

    for acquisition in acquisitions:
        value = acquisition(X)
        assert value.shape == torch.Size([1])
        assert torch.isfinite(value).all()
