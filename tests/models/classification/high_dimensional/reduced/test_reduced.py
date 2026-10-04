"""Tests for reduced-space binary GP classifiers."""

import pytest
import torch

from robotorchan.models.classification import (
    PCABinarySingleTaskGPClassifier,
    PLSBinarySingleTaskGPClassifier,
    RandomProjectionBinarySingleTaskGPClassifier,
)


def _data() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(13)
    X = torch.rand(12, 8, dtype=torch.double)
    Y = (X[:, 0] + 0.5 * X[:, 1] > 0.75).to(dtype=torch.double)
    return X, Y


@pytest.mark.parametrize(
    "model_type",
    [
        PCABinarySingleTaskGPClassifier,
        PLSBinarySingleTaskGPClassifier,
        RandomProjectionBinarySingleTaskGPClassifier,
    ],
)
def test_reduced_classifier_accepts_original_and_latent_inputs(model_type) -> None:
    train_X, train_Y = _data()
    model = model_type(train_X, train_Y, n_components=3)
    reduced_X = model.input_reducer.transform(train_X[:4])
    original_probabilities = model.predict_proba(train_X[:4])
    reduced_probabilities = model.predict_proba(reduced_X)
    assert model.original_input_dim == 8
    assert model.reduced_input_dim == 3
    assert original_probabilities.shape == torch.Size([4, 2])
    torch.testing.assert_close(original_probabilities, reduced_probabilities)


def test_reduced_classifier_preserves_raw_training_inputs() -> None:
    train_X, train_Y = _data()
    model = PCABinarySingleTaskGPClassifier(train_X, train_Y, n_components=3)
    torch.testing.assert_close(model.raw_train_X, train_X)
    torch.testing.assert_close(model.raw_train_Y, train_Y)


def test_reduced_classifier_rejects_wrong_input_dimension() -> None:
    train_X, train_Y = _data()
    model = PCABinarySingleTaskGPClassifier(train_X, train_Y, n_components=3)
    with pytest.raises(ValueError, match="Expected final input dimension"):
        model.predict_proba(torch.rand(2, 5, dtype=torch.double))
