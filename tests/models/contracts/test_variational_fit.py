"""Variational fit contract tests."""

import pytest
import torch

from robotorchan.models import MixedSingleTaskVariationalGP, SingleTaskVariationalGP


@pytest.mark.parametrize("mixed", [False, True])
def test_variational_fit_updates_parameters(mixed: bool) -> None:
    train_X = torch.linspace(0.0, 1.0, 12).unsqueeze(-1)
    train_Y = torch.sin(train_X)
    if mixed:
        categories = (torch.arange(12) % 2).to(train_X).unsqueeze(-1)
        train_X = torch.cat([train_X, categories], dim=-1)
        model = MixedSingleTaskVariationalGP(train_X, train_Y, cat_dims=[1], inducing_points=4)
    else:
        model = SingleTaskVariationalGP(train_X, train_Y, inducing_points=4)

    before = [parameter.detach().clone() for parameter in model.parameters()]
    model.fit(epochs=2, lr=0.01, batch_size=6)

    assert model.supports_fit is True
    assert not model.training
    assert any(
        not torch.equal(old, new) for old, new in zip(before, model.parameters(), strict=True)
    )
    assert torch.isfinite(model.posterior(train_X[:2]).mean).all()


def test_variational_fit_requires_training_targets() -> None:
    train_X = torch.rand(8, 1)
    model = SingleTaskVariationalGP(train_X, inducing_points=4)

    with pytest.raises(ValueError, match="requires train_Y"):
        model.fit()
