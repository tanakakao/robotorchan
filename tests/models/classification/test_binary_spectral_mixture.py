"""Focused checks for spectral-mixture binary GP classification."""

import pytest
import torch
from gpytorch.kernels import SpectralMixtureKernel
from gpytorch.mlls import VariationalELBO

from robotorchan.models.classification import (
    SpectralMixtureBinaryGPClassifier,
    get_classification_model_class,
)


def _training_data():
    X = torch.linspace(0.05, 0.95, 16, dtype=torch.double).unsqueeze(-1)
    Y = (torch.sin(4 * torch.pi * X.squeeze(-1)) > 0).long()
    return X, Y


def test_spectral_mixture_binary_training_and_prediction():
    X, Y = _training_data()
    model = SpectralMixtureBinaryGPClassifier(X, Y, num_mixtures=2, inducing_points=8)
    assert isinstance(model.make_mll(), VariationalELBO)
    assert isinstance(model.model.covar_module.base_kernel, SpectralMixtureKernel)
    model.fit(epochs=2, lr=0.01)
    probabilities = model.predict_proba(X[:3])
    assert probabilities.shape == (3, 2)
    assert torch.isfinite(probabilities).all()
    torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(3, dtype=torch.double))
    assert torch.isfinite(model.posterior(X[:3]).mean).all()


def test_spectral_mixture_binary_registry():
    assert (
        get_classification_model_class("binary.spectral_mixture")
        is SpectralMixtureBinaryGPClassifier
    )


@pytest.mark.parametrize("num_mixtures", [0, -1, 1.5, True])
def test_spectral_mixture_rejects_invalid_mixture_count(num_mixtures):
    X, Y = _training_data()
    with pytest.raises(ValueError):
        SpectralMixtureBinaryGPClassifier(X, Y, num_mixtures=num_mixtures)
