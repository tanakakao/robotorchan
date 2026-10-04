"""Tests for SAAS binary GP classification."""

import pytest
import torch

from robotorchan.models.classification import SaasBinarySingleTaskGPClassifier


def _data() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(11)
    X = torch.rand(10, 16, dtype=torch.double)
    Y = (X[:, 0] - X[:, 3] > 0.0).to(dtype=torch.double)
    return X, Y


def test_saas_binary_classifier_preserves_classification_contract() -> None:
    train_X, train_Y = _data()
    model = SaasBinarySingleTaskGPClassifier(train_X, train_Y, global_scale=0.05)
    probabilities = model.predict_proba(train_X[:4])
    assert probabilities.shape == torch.Size([4, 2])
    torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(4, dtype=torch.double))
    assert model.global_scale == 0.05


def test_saas_binary_classifier_prior_targets_inverse_lengthscale() -> None:
    train_X, train_Y = _data()
    model = SaasBinarySingleTaskGPClassifier(train_X, train_Y)
    base_kernel = model.model.covar_module.base_kernel
    priors = {
        name: closure(base_kernel)
        for name, _, _prior, closure, _ in base_kernel.named_priors()
    }
    assert "saas_inv_lengthscale_prior" in priors
    torch.testing.assert_close(
        priors["saas_inv_lengthscale_prior"],
        base_kernel.lengthscale.reciprocal(),
    )


def test_saas_binary_classifier_elbo_has_finite_gradient() -> None:
    train_X, train_Y = _data()
    model = SaasBinarySingleTaskGPClassifier(train_X, train_Y)
    model.train()
    model.likelihood.train()
    output = model.model(train_X)
    loss = -model.make_mll()(output, train_Y)
    loss.backward()
    gradient = model.model.covar_module.base_kernel.raw_lengthscale.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()


def test_saas_binary_classifier_rejects_nonpositive_global_scale() -> None:
    train_X, train_Y = _data()
    with pytest.raises(ValueError, match="global_scale must be positive"):
        SaasBinarySingleTaskGPClassifier(train_X, train_Y, global_scale=0.0)
