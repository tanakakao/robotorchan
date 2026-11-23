"""Smoke tests for sparse-ARD multiclass and ordinal GP classifiers."""

import pytest
import torch
from gpytorch.kernels import ScaleKernel

from robotorchan.models.classification import (
    MapSaasMulticlassGPClassifier,
    MapSaasOrdinalGPClassifier,
    SaasMulticlassGPClassifier,
    SaasOrdinalGPClassifier,
)

MODELS = (
    SaasMulticlassGPClassifier,
    SaasOrdinalGPClassifier,
    MapSaasMulticlassGPClassifier,
    MapSaasOrdinalGPClassifier,
)


@pytest.mark.parametrize("model_type", MODELS)
def test_sparse_ard_fit_predict_and_entropy_candidate(model_type):
    torch.manual_seed(23)
    train_X = torch.rand(15, 5, dtype=torch.double)
    train_Y = torch.arange(15) % 3
    model = model_type(train_X, train_Y, num_classes=3)
    kernel = (
        model.class_models[0].model.covar_module
        if isinstance(model, (SaasMulticlassGPClassifier, MapSaasMulticlassGPClassifier))
        else model.model.covar_module
    )
    base_kernel = kernel.base_kernel if isinstance(kernel, ScaleKernel) else kernel
    assert base_kernel.ard_num_dims == 5
    assert any(
        name == "saas_inv_lengthscale_prior"
        for name, *_ in base_kernel.named_priors()
    )
    model.fit(epochs=2, lr=0.005)
    candidate_X = torch.rand(9, 5, dtype=torch.double)
    with torch.no_grad():
        probabilities = model.predict_proba(candidate_X)
        entropy = -torch.special.xlogy(probabilities, probabilities).sum(-1)
        candidate = candidate_X[entropy.argmax()]
    assert probabilities.shape == (9, 3)
    assert torch.isfinite(probabilities).all()
    torch.testing.assert_close(
        probabilities.sum(-1), torch.ones(9, dtype=torch.double), atol=1e-5, rtol=1e-5
    )
    assert candidate.shape == (5,)


@pytest.mark.parametrize("model_type", MODELS)
def test_sparse_ard_rejects_nonpositive_scale(model_type):
    train_X = torch.rand(9, 4, dtype=torch.double)
    train_Y = torch.arange(9) % 3
    kwargs = (
        {"tau": 0.0}
        if model_type in (MapSaasMulticlassGPClassifier, MapSaasOrdinalGPClassifier)
        else {"global_scale": 0.0}
    )
    with pytest.raises(ValueError, match="must be positive"):
        model_type(train_X, train_Y, num_classes=3, **kwargs)


@pytest.mark.parametrize(
    "model_type", [SaasMulticlassGPClassifier, MapSaasMulticlassGPClassifier]
)
def test_multiclass_saas_prior_contributes_gradient(model_type):
    """Sparse prior must enter the multiclass objective, not only be registered."""
    x = torch.rand(9, 4, dtype=torch.double)
    y = torch.arange(9) % 3
    model = model_type(x, y, num_classes=3)
    log_prior = model._log_kernel_prior()
    assert torch.isfinite(log_prior)
    assert log_prior.requires_grad
    log_prior.backward()
    grads = [
        (
            gp.model.covar_module.base_kernel
            if isinstance(gp.model.covar_module, ScaleKernel)
            else gp.model.covar_module
        ).raw_lengthscale.grad
        for gp in model.class_models
    ]
    assert all(grad is not None and torch.isfinite(grad).all() for grad in grads)
