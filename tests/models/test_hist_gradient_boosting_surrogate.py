import pytest
import torch
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.posteriors.ensemble import EnsemblePosterior

pytest.importorskip("sklearn")

from robotorchan.models import HistGradientBoostingSurrogate
from robotorchan.models.base import UnsupportedModelOperationError


def make_training_data() -> tuple[torch.Tensor, torch.Tensor]:
    train_X = torch.linspace(0.0, 1.0, 24, dtype=torch.double).unsqueeze(-1)
    train_Y = torch.sin(train_X * 6.0)
    return train_X, train_Y


def test_hist_gradient_boosting_training_contract() -> None:
    train_X, train_Y = make_training_data()
    model = HistGradientBoostingSurrogate(train_X, train_Y, n_members=4, random_state=0)

    assert model.supports_fit is True
    assert model.supports_mll is False
    assert model.supports_input_gradients is False
    assert torch.equal(model.raw_train_X, train_X)
    assert torch.equal(model.raw_train_Y, train_Y)
    with pytest.raises(UnsupportedModelOperationError):
        model.make_mll()

    model.fit()
    assert model.is_fitted
    assert len(model._members) == 4


def test_hist_gradient_boosting_posterior_uses_complete_models() -> None:
    train_X, train_Y = make_training_data()
    model = HistGradientBoostingSurrogate(train_X, train_Y, n_members=5, random_state=1)
    model.fit()

    posterior = model.posterior(torch.tensor([[0.2], [0.8]], dtype=torch.double))

    assert isinstance(posterior, EnsemblePosterior)
    assert posterior.values.shape == torch.Size([5, 2, 1])
    assert torch.isfinite(posterior.values).all()


def test_hist_gradient_boosting_integrates_with_mc_qei() -> None:
    train_X, train_Y = make_training_data()
    model = HistGradientBoostingSurrogate(train_X, train_Y, n_members=5, random_state=2)
    model.fit()
    acqf = qExpectedImprovement(model=model, best_f=train_Y.max())

    value = acqf(torch.tensor([[[0.25], [0.75]]], dtype=torch.double))

    assert value.shape == torch.Size([1])
    assert torch.isfinite(value).all()


def test_hist_gradient_boosting_requires_fit_and_multiple_members() -> None:
    train_X, train_Y = make_training_data()
    model = HistGradientBoostingSurrogate(train_X, train_Y, n_members=2)
    with pytest.raises(RuntimeError, match="fit"):
        model.posterior(train_X[:2])
    with pytest.raises(ValueError, match="at least 2"):
        HistGradientBoostingSurrogate(train_X, train_Y, n_members=1)
