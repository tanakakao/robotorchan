"""Tests for sparse high-dimensional binary classification."""

import pytest
import torch

from robotorchan.models.classification import MapSaasBinarySingleTaskGPClassifier


def _data() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(7)
    X = torch.rand(8, 12, dtype=torch.double)
    Y = (X[:, 0] + X[:, 1] > 1.0).to(dtype=torch.double)
    return X, Y


def test_map_saas_binary_classifier_preserves_prediction_contract() -> None:
    train_X, train_Y = _data()
    model = MapSaasBinarySingleTaskGPClassifier(train_X, train_Y, tau=0.05)
    probabilities = model.predict_proba(train_X[:3])
    assert probabilities.shape == torch.Size([3, 2])
    torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(3, dtype=torch.double))
    assert model.tau == 0.05


def test_map_saas_binary_classifier_registers_sparse_prior() -> None:
    train_X, train_Y = _data()
    model = MapSaasBinarySingleTaskGPClassifier(train_X, train_Y)
    names = {name for name, *_ in model.model.covar_module.named_priors()}
    assert any("saas_inv_lengthscale_prior" in name for name in names)


def test_map_saas_binary_classifier_rejects_nonpositive_tau() -> None:
    train_X, train_Y = _data()
    with pytest.raises(ValueError, match="tau must be positive"):
        MapSaasBinarySingleTaskGPClassifier(train_X, train_Y, tau=0.0)
