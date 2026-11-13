"""Focused tests for infinite-width ReLU binary GP classifiers."""

import pytest
import torch
from gpytorch.kernels import AdditiveKernel, ScaleKernel
from gpytorch.mlls import VariationalELBO

from robotorchan.models.classification import (
    InfiniteWidthBNNBinaryGPClassifier,
    MixedInfiniteWidthBNNBinaryGPClassifier,
    get_classification_model_class,
)
from robotorchan.models.expressive.infinite_width_bnn import InfiniteWidthReLUKernel


def _data():
    x = torch.linspace(0.05, 0.95, 16, dtype=torch.double)
    category = torch.tensor([0, 1] * 8, dtype=torch.double)
    X = torch.stack((x, category), dim=-1)
    Y = (torch.sin(4 * torch.pi * x) + 0.3 * category > 0).long()
    return X, Y


@pytest.mark.parametrize(
    ("model_class", "kwargs", "registry_id"),
    [
        (InfiniteWidthBNNBinaryGPClassifier, {}, "binary.infinite_width_bnn"),
        (
            MixedInfiniteWidthBNNBinaryGPClassifier,
            {"cat_dims": [1]},
            "binary.mixed_infinite_width_bnn",
        ),
    ],
)
def test_infinite_width_binary_training_and_registry(model_class, kwargs, registry_id):
    X, Y = _data()
    model = model_class(X, Y, depth=2, inducing_points=8, **kwargs)
    assert isinstance(model.make_mll(), VariationalELBO)
    assert get_classification_model_class(registry_id) is model_class
    assert isinstance(model.model.covar_module, AdditiveKernel | ScaleKernel)
    model.fit(epochs=2, lr=0.01)
    probabilities = model.predict_proba(X[:3])
    assert probabilities.shape == (3, 2)
    assert torch.isfinite(probabilities).all()
    torch.testing.assert_close(probabilities.sum(-1), torch.ones(3, dtype=torch.double))
    assert torch.isfinite(model.posterior(X[:3]).mean).all()
    if model_class is InfiniteWidthBNNBinaryGPClassifier:
        assert isinstance(model.model.covar_module.base_kernel, InfiniteWidthReLUKernel)
    else:
        assert model.cat_dims == (1,)


def test_infinite_width_binary_rejects_invalid_kernel_configuration():
    X, Y = _data()
    with pytest.raises(ValueError, match="depth"):
        InfiniteWidthBNNBinaryGPClassifier(X, Y, depth=0)
    with pytest.raises(ValueError, match="continuous"):
        MixedInfiniteWidthBNNBinaryGPClassifier(X, Y, cat_dims=[0, 1])
