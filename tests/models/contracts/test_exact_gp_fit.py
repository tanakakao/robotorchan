"""Contracts for opt-in native fitting of standard exact GP wrappers."""

import torch
from botorch.fit import fit_gpytorch_mll

from robotorchan.models import MixedSingleTaskGP, SingleTaskGP, SingleTaskVariationalGP
from robotorchan.models.non_gp.random_forest import RandomForestSurrogate


def _regression_data() -> tuple[torch.Tensor, torch.Tensor]:
    X = torch.linspace(0.1, 0.9, 8, dtype=torch.double).unsqueeze(-1)
    return X, torch.sin(X * 4.0)


def test_standard_exact_gp_fit_preserves_native_mll_and_modes() -> None:
    X, Y = _regression_data()
    model = SingleTaskGP(X, Y)
    assert model.supports_fit
    assert model.make_mll().model is model
    assert model.fit(optimizer_kwargs={"options": {"maxiter": 2}}) is None
    assert not model.training
    with torch.no_grad():
        posterior = model.posterior(X[:2])
    assert posterior.mean.shape == (2, 1)
    assert torch.isfinite(posterior.mean).all()
    assert torch.equal(model.raw_train_X, X)
    assert torch.equal(model.raw_train_Y, Y)


def test_mixed_exact_gp_fit_supports_native_categorical_kernel() -> None:
    X = torch.tensor(
        [
            [0.1, 0.0],
            [0.2, 1.0],
            [0.3, 0.0],
            [0.4, 1.0],
            [0.5, 0.0],
            [0.6, 1.0],
            [0.7, 0.0],
            [0.8, 1.0],
        ],
        dtype=torch.double,
    )
    Y = torch.sin(X[:, :1] * 3.0) + X[:, 1:] * 0.15
    model = MixedSingleTaskGP(X, Y, cat_dims=[1])
    assert model.supports_fit
    assert model.fit(optimizer_kwargs={"options": {"maxiter": 2}}) is None
    with torch.no_grad():
        posterior = model.posterior(X[:2])
    assert posterior.mean.shape == (2, 1)
    assert torch.isfinite(posterior.mean).all()


def test_direct_botorch_mll_training_remains_available() -> None:
    X, Y = _regression_data()
    model = SingleTaskGP(X, Y)
    fit_gpytorch_mll(model.make_mll(), optimizer_kwargs={"options": {"maxiter": 2}})
    with torch.no_grad():
        posterior = model.posterior(X[:2])
    assert torch.isfinite(posterior.mean).all()


def test_specialized_families_do_not_inherit_exact_fit() -> None:
    X, Y = _regression_data()
    variational = SingleTaskVariationalGP(X, Y)
    forest = RandomForestSurrogate(X, Y, n_estimators=4, random_state=0)
    assert not variational.supports_fit
    assert not hasattr(variational, "fit")
    assert forest.supports_fit
    assert forest.fit() is None
