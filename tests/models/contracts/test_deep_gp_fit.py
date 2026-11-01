"""Fit contract for variational deep Gaussian process models."""

import pytest
import torch

from robotorchan.models.expressive.deep_gp import SingleTaskDeepGP


def test_deep_gp_fit_updates_parameters_and_posterior() -> None:
    torch.manual_seed(0)
    X = torch.rand(10, 2, dtype=torch.double)
    Y = torch.sin(X[:, :1])
    model = SingleTaskDeepGP(X, Y, hidden_dims=(2,), num_inducing=4, posterior_samples=8)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    assert model.supports_fit
    assert model.fit(epochs=2, lr=0.01, num_likelihood_samples=4) is None
    assert not model.training
    assert not model.likelihood.training
    assert any(
        not torch.equal(old, new) for old, new in zip(before, model.parameters(), strict=True)
    )
    assert torch.isfinite(model.posterior(X[:2]).mean).all()


@pytest.mark.parametrize(
    "kwargs",
    [{"epochs": 0}, {"lr": 0.0}, {"num_likelihood_samples": 0}],
)
def test_deep_gp_fit_rejects_invalid_settings(kwargs: dict) -> None:
    X = torch.rand(8, 2)
    model = SingleTaskDeepGP(X, X[:, :1], hidden_dims=(2,), num_inducing=4)
    with pytest.raises(ValueError):
        model.fit(**kwargs)
