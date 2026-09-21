import pytest
import torch
from botorch.acquisition.active_learning import qNegIntegratedPosteriorVariance

from robotorchan.acquisition import PosteriorStd, PosteriorVariance
from robotorchan.models import SingleTaskGP


def _model() -> SingleTaskGP:
    train_X = torch.linspace(0.0, 1.0, 8, dtype=torch.double).unsqueeze(-1)
    train_Y = torch.sin(train_X * 5.0)
    return SingleTaskGP(train_X, train_Y)


def test_posterior_variance_and_std_are_consistent() -> None:
    model = _model()
    X = torch.tensor([[[0.25]], [[0.75]]], dtype=torch.double)

    variance = PosteriorVariance(model)(X)
    std = PosteriorStd(model)(X)

    assert variance.shape == torch.Size([2])
    assert torch.isfinite(variance).all()
    assert torch.allclose(std.square(), variance)


def test_posterior_variance_rejects_batch_q() -> None:
    model = _model()
    X = torch.tensor([[[0.25], [0.75]]], dtype=torch.double)

    with pytest.raises(ValueError, match="q=1"):
        PosteriorVariance(model)(X)


def test_native_negative_integrated_posterior_variance() -> None:
    model = _model()
    mc_points = torch.linspace(0.0, 1.0, 16, dtype=torch.double).unsqueeze(-1)
    acquisition = qNegIntegratedPosteriorVariance(model=model, mc_points=mc_points)
    X = torch.tensor([[[0.3], [0.7]]], dtype=torch.double)

    value = acquisition(X)

    assert value.shape == torch.Size([1])
    assert torch.isfinite(value).all()
