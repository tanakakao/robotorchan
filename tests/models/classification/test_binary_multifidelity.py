"""Focused checks for the binary multi-fidelity variational GP."""

import pytest
import torch
from gpytorch.mlls import VariationalELBO

from robotorchan.models.classification import (
    MultiFidelityBinaryGPClassifier,
    get_classification_model_class,
)


def _training_data():
    x = torch.linspace(0.05, 0.95, 12, dtype=torch.double)
    fidelity = torch.tensor([0.25, 1.0] * 6, dtype=torch.double)
    X = torch.stack((x, fidelity), dim=-1)
    Y = ((x + 0.15 * fidelity) > 0.55).long()
    return X, Y


def test_multifidelity_binary_training_and_predictions():
    X, Y = _training_data()
    model = MultiFidelityBinaryGPClassifier(X, Y, data_fidelities=[1], inducing_points=6)
    assert isinstance(model.make_mll(), VariationalELBO)
    model.fit(epochs=2, lr=0.01)
    query = torch.tensor([[0.4, 0.25], [0.4, 1.0]], dtype=torch.double)
    latent = model.posterior(query)
    assert torch.isfinite(latent.mean).all()
    probabilities = model.predict_proba(query)
    assert probabilities.shape == (2, 2)
    assert torch.isfinite(probabilities).all()
    torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(2, dtype=torch.double))
    assert model.fidelity_dims == (1,)
    assert model.design_dims == (0,)


def test_multifidelity_binary_registry():
    assert get_classification_model_class("binary.multifidelity") is MultiFidelityBinaryGPClassifier


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"data_fidelities": [0, 1]},
        {"iteration_fidelity": 1, "data_fidelities": [1]},
        {"data_fidelities": [2]},
        {"data_fidelities": [1], "linear_truncated": True},
    ],
)
def test_multifidelity_binary_rejects_invalid_fidelity(kwargs):
    X, Y = _training_data()
    with pytest.raises(ValueError):
        MultiFidelityBinaryGPClassifier(X, Y, **kwargs)
