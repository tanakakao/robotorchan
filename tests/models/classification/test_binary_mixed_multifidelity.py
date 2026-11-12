"""Focused Phase 3 checks for mixed multi-fidelity binary classification."""

import pytest
import torch
from botorch.models.kernels.downsampling import DownsamplingKernel
from gpytorch.mlls import VariationalELBO

from robotorchan.models.classification import (
    MixedMultiFidelityBinaryGPClassifier,
    get_classification_model_class,
)


def _training_data():
    x = torch.linspace(0.1, 0.9, 12, dtype=torch.double)
    category = torch.tensor([0, 1] * 6, dtype=torch.double)
    fidelity = torch.tensor([0.3, 1.0] * 6, dtype=torch.double)
    X = torch.stack((x, category, fidelity), dim=-1)
    Y = (x + 0.2 * category + 0.1 * fidelity > 0.6).long()
    return X, Y


def test_mixed_multifidelity_fit_and_predict():
    X, Y = _training_data()
    model = MixedMultiFidelityBinaryGPClassifier(
        X, Y, cat_dims=[1], data_fidelities=[2], inducing_points=6
    )
    assert isinstance(model.make_mll(), VariationalELBO)
    model.fit(epochs=2, lr=0.01)
    query = torch.tensor([[0.4, 0.0, 0.3], [0.4, 1.0, 1.0]], dtype=torch.double)
    assert torch.isfinite(model.posterior(query).mean).all()
    probabilities = model.predict_proba(query)
    assert probabilities.shape == (2, 2)
    assert torch.isfinite(probabilities).all()
    torch.testing.assert_close(probabilities.sum(-1), torch.ones(2, dtype=torch.double))
    assert model.cat_dims == (1,)
    assert model.fidelity_dims == (2,)
    assert any(
        isinstance(kernel, DownsamplingKernel)
        for kernel in model.model.covar_module.base_kernel.kernels
    )


def test_mixed_multifidelity_registry():
    assert (
        get_classification_model_class("binary.mixed_multifidelity")
        is MixedMultiFidelityBinaryGPClassifier
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cat_dims": [2], "data_fidelities": [2]},
        {"cat_dims": [], "data_fidelities": [2]},
        {"cat_dims": [1]},
        {"cat_dims": [1], "data_fidelities": [2], "linear_truncated": True},
    ],
)
def test_mixed_multifidelity_rejects_invalid_dimensions(kwargs):
    X, Y = _training_data()
    with pytest.raises(ValueError):
        MixedMultiFidelityBinaryGPClassifier(X, Y, **kwargs)
