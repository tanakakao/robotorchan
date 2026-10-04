"""Tests for mixed-input binary GP classification."""

import pytest
import torch
from botorch.models.kernels.categorical import CategoricalKernel
from gpytorch.kernels import AdditiveKernel, ProductKernel, ScaleKernel

from robotorchan.models.classification import MixedBinarySingleTaskGPClassifier


def _training_data() -> tuple[torch.Tensor, torch.Tensor]:
    train_X = torch.tensor(
        [
            [0.0, 0.0],
            [0.2, 1.0],
            [0.4, 0.0],
            [0.6, 1.0],
            [0.8, 0.0],
            [1.0, 1.0],
        ],
        dtype=torch.double,
    )
    train_Y = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.double)
    return train_X, train_Y


def test_mixed_binary_classifier_uses_native_categorical_kernel() -> None:
    train_X, train_Y = _training_data()
    model = MixedBinarySingleTaskGPClassifier(train_X, train_Y, cat_dims=[1])
    covariance = model.model.covar_module
    assert isinstance(covariance, AdditiveKernel)
    assert any(isinstance(module, CategoricalKernel) for module in covariance.modules())
    assert any(isinstance(module, ProductKernel) for module in covariance.modules())
    assert any(isinstance(module, ScaleKernel) for module in covariance.modules())
    assert model.cat_dims == (1,)


def test_mixed_binary_classifier_supports_negative_categorical_dims() -> None:
    train_X, train_Y = _training_data()
    model = MixedBinarySingleTaskGPClassifier(train_X, train_Y, cat_dims=[-1])
    assert model.cat_dims == (1,)


def test_mixed_binary_classifier_rejects_invalid_categorical_dims() -> None:
    train_X, train_Y = _training_data()
    with pytest.raises(ValueError, match="duplicate"):
        MixedBinarySingleTaskGPClassifier(train_X, train_Y, cat_dims=[1, -1])
    with pytest.raises(ValueError, match="at least one"):
        MixedBinarySingleTaskGPClassifier(train_X, train_Y, cat_dims=[])


def test_mixed_binary_classifier_runs_prediction_and_sampling_contracts() -> None:
    train_X, train_Y = _training_data()
    model = MixedBinarySingleTaskGPClassifier(train_X, train_Y, cat_dims=[1])
    X = torch.tensor([[0.3, 0.0], [0.7, 1.0]], dtype=torch.double)
    probabilities = model.predict_proba(X)
    classes = model.predict_class(X)
    samples = model.sample_class_probabilities(X, torch.Size([3]))
    assert probabilities.shape == torch.Size([2, 2])
    assert classes.shape == torch.Size([2])
    assert samples.shape == torch.Size([3, 2, 2])
    torch.testing.assert_close(
        probabilities.sum(dim=-1),
        torch.ones(2, dtype=torch.double),
    )
    torch.testing.assert_close(
        samples.sum(dim=-1),
        torch.ones(3, 2, dtype=torch.double),
    )


def test_mixed_binary_classifier_preserves_raw_inputs_and_labels() -> None:
    train_X, train_Y = _training_data()
    model = MixedBinarySingleTaskGPClassifier(train_X, train_Y.bool(), cat_dims=[1])
    torch.testing.assert_close(model.raw_train_X, train_X)
    torch.testing.assert_close(model.raw_train_Y, train_Y.bool())
    assert model.raw_train_Y.dtype == torch.bool
