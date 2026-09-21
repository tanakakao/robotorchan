import pytest
import torch
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.posteriors.ensemble import EnsemblePosterior

pytest.importorskip("sklearn")

from robotorchan.models import RandomForestSurrogate
from robotorchan.models.base import UnsupportedModelOperationError


def make_training_data() -> tuple[torch.Tensor, torch.Tensor]:
    train_X = torch.linspace(0.0, 1.0, 12, dtype=torch.double).unsqueeze(-1)
    train_Y = torch.sin(train_X * 6.0)
    return train_X, train_Y


def test_random_forest_training_contract_and_raw_data() -> None:
    train_X, train_Y = make_training_data()
    model = RandomForestSurrogate(train_X, train_Y, n_estimators=8, random_state=0)

    assert model.supports_fit is True
    assert model.supports_mll is False
    assert model.supports_input_gradients is False
    assert not model.is_fitted
    assert torch.equal(model.raw_train_X, train_X)
    assert torch.equal(model.raw_train_Y, train_Y)
    with pytest.raises(UnsupportedModelOperationError):
        model.make_mll()

    model.fit()
    assert model.is_fitted


def test_random_forest_posterior_uses_tree_ensemble() -> None:
    train_X, train_Y = make_training_data()
    model = RandomForestSurrogate(train_X, train_Y, n_estimators=7, random_state=1)
    model.fit()
    X = torch.tensor([[0.2], [0.8]], dtype=torch.double)

    posterior = model.posterior(X)

    assert isinstance(posterior, EnsemblePosterior)
    assert posterior.values.shape == torch.Size([7, 2, 1])
    assert posterior.mean.shape == torch.Size([2, 1])
    assert posterior.dtype == torch.double
    assert torch.isfinite(posterior.values).all()


def test_random_forest_posterior_supports_batched_candidates() -> None:
    train_X, train_Y = make_training_data()
    model = RandomForestSurrogate(train_X, train_Y, n_estimators=5, random_state=2)
    model.fit()
    X = torch.rand(3, 4, 1, dtype=torch.double)

    posterior = model.posterior(X)

    assert posterior.values.shape == torch.Size([3, 5, 4, 1])


def test_random_forest_integrates_with_mc_qei() -> None:
    train_X, train_Y = make_training_data()
    model = RandomForestSurrogate(train_X, train_Y, n_estimators=8, random_state=3)
    model.fit()
    acqf = qExpectedImprovement(model=model, best_f=train_Y.max())
    X = torch.tensor([[[0.25], [0.75]]], dtype=torch.double)

    value = acqf(X)

    assert value.shape == torch.Size([1])
    assert torch.isfinite(value).all()


def test_random_forest_requires_fit_and_single_output() -> None:
    train_X, train_Y = make_training_data()
    model = RandomForestSurrogate(train_X, train_Y, n_estimators=4, random_state=4)
    with pytest.raises(RuntimeError, match="fit"):
        model.posterior(train_X[:2])
    with pytest.raises(ValueError, match="one output"):
        RandomForestSurrogate(train_X, torch.cat([train_Y, train_Y], dim=-1))
