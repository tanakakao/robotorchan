"""Representative executable checks complementing static Phase 2 API inspection."""

import pytest
import torch
from botorch.fit import fit_gpytorch_mll

from robotorchan.models import SingleTaskGP, SingleTaskVariationalGP
from robotorchan.models.classification.binary.standard.single_task import (
    BinarySingleTaskGPClassifier,
)
from robotorchan.models.non_gp.random_forest import RandomForestSurrogate


def _data() -> tuple[torch.Tensor, torch.Tensor]:
    X = torch.linspace(0.1, 0.9, 8, dtype=torch.double).unsqueeze(-1)
    Y = torch.sin(2 * torch.pi * X)
    return X, Y


def test_exact_gp_mll_and_posterior_are_directly_usable() -> None:
    X, Y = _data()
    model = SingleTaskGP(X, Y)
    mll = model.make_mll()
    assert mll.model is model
    assert model.raw_train_X.shape == X.shape
    assert model.raw_train_Y.shape == Y.shape
    assert model.supports_fit
    fit_gpytorch_mll(mll, optimizer_kwargs={"options": {"maxiter": 2}})
    model.eval()
    with torch.no_grad():
        posterior = model.posterior(X[:2])
    assert posterior.mean.shape == (2, 1)
    assert torch.isfinite(posterior.mean).all()


def test_variational_gp_uses_elbo_with_dataset_size() -> None:
    X, Y = _data()
    model = SingleTaskVariationalGP(X, Y)
    mll = model.make_mll(num_data=len(X))
    assert mll.num_data == len(X)
    with torch.no_grad():
        posterior = model.posterior(X[:2])
    assert posterior.mean.shape == (2, 1)


def test_binary_classifier_keeps_latent_and_label_probability_separate() -> None:
    X, _ = _data()
    Y = (X[:, 0] >= 0.5).to(dtype=X.dtype)
    model = BinarySingleTaskGPClassifier(X, Y)
    latent = model.posterior(X[:2])
    probabilities = model.predict_proba(X[:2])
    classes = model.predict_class(X[:2])
    assert latent.mean.shape[-1] == 1
    assert probabilities.shape == (2, 2)
    torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(2, dtype=X.dtype))
    assert classes.shape == (2,)


def test_non_gp_fit_is_a_real_training_operation() -> None:
    pytest.importorskip("sklearn")
    X, Y = _data()
    model = RandomForestSurrogate(X, Y, n_estimators=6, random_state=0)
    with pytest.raises(RuntimeError, match="fit"):
        model.posterior(X[:2])
    assert model.fit() is None
    posterior = model.posterior(X[:2])
    assert posterior.mean.shape == (2, 1)
    assert torch.isfinite(posterior.mean).all()
