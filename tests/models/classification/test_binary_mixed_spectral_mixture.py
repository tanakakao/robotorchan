"""Focused mixed spectral-mixture binary classification checks."""

import pytest
import torch
from gpytorch.kernels import AdditiveKernel, SpectralMixtureKernel
from gpytorch.mlls import VariationalELBO

from robotorchan.models.classification import (
    MixedSpectralMixtureBinaryGPClassifier,
    get_classification_model_class,
)


def _training_data():
    x = torch.linspace(0.05, 0.95, 16, dtype=torch.double)
    category = torch.tensor([0, 1] * 8, dtype=torch.double)
    X = torch.stack((x, category), dim=-1)
    Y = (torch.sin(4 * torch.pi * x) + category * 0.3 > 0).long()
    return X, Y


def test_mixed_spectral_mixture_training_and_prediction():
    X, Y = _training_data()
    model = MixedSpectralMixtureBinaryGPClassifier(
        X, Y, cat_dims=[1], num_mixtures=2, inducing_points=8
    )
    assert isinstance(model.make_mll(), VariationalELBO)
    assert isinstance(model.model.covar_module, AdditiveKernel)
    assert any(
        isinstance(kernel, SpectralMixtureKernel)
        for kernel in model.model.covar_module.kernels
    )
    model.fit(epochs=2, lr=0.01)
    probabilities = model.predict_proba(X[:3])
    assert probabilities.shape == (3, 2)
    assert torch.isfinite(probabilities).all()
    torch.testing.assert_close(probabilities.sum(-1), torch.ones(3, dtype=torch.double))
    assert torch.isfinite(model.posterior(X[:3]).mean).all()
    assert model.cat_dims == (1,)


def test_mixed_spectral_mixture_registry():
    assert (
        get_classification_model_class("binary.mixed_spectral_mixture")
        is MixedSpectralMixtureBinaryGPClassifier
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cat_dims": []},
        {"cat_dims": [0, 1]},
        {"cat_dims": [1], "num_mixtures": 0},
    ],
)
def test_mixed_spectral_mixture_rejects_invalid_configuration(kwargs):
    X, Y = _training_data()
    with pytest.raises(ValueError):
        MixedSpectralMixtureBinaryGPClassifier(X, Y, **kwargs)
