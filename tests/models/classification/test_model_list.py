"""Tests for heterogeneous classification model composition."""

import pytest
import torch
from botorch.posteriors import PosteriorList

from robotorchan.models.classification import (
    BinarySingleTaskGPClassifier,
    ClassificationModelList,
    MixedBinarySingleTaskGPClassifier,
)


def _models() -> tuple[BinarySingleTaskGPClassifier, MixedBinarySingleTaskGPClassifier]:
    X = torch.tensor([[0.0, 0.0], [0.4, 1.0], [0.8, 0.0]], dtype=torch.double)
    Y = torch.tensor([0.0, 1.0, 1.0], dtype=torch.double)
    return BinarySingleTaskGPClassifier(X, Y), MixedBinarySingleTaskGPClassifier(X, Y, cat_dims=[1])


def test_classification_model_list_composes_latent_posteriors() -> None:
    first, second = _models()
    models = ClassificationModelList(first, second)
    X = torch.tensor([[0.2, 0.0], [0.6, 1.0]], dtype=torch.double)
    posterior = models.latent_posterior(X)
    assert isinstance(posterior, PosteriorList)
    assert len(posterior.posteriors) == 2


def test_classification_model_list_keeps_child_prediction_shapes() -> None:
    first, second = _models()
    models = ClassificationModelList(first, second)
    X = torch.tensor([[0.2, 0.0], [0.6, 1.0]], dtype=torch.double)
    probabilities = models.predict_proba(X)
    classes = models.predict_class(X)
    assert len(probabilities) == 2
    assert all(value.shape == torch.Size([2, 2]) for value in probabilities)
    assert all(value.shape == torch.Size([2]) for value in classes)


def test_classification_model_list_exposes_independent_training_contracts() -> None:
    first, second = _models()
    models = ClassificationModelList(first, second)
    assert len(models.make_mlls()) == 2
    assert models.raw_train_Xs == (first.raw_train_X, second.raw_train_X)
    assert models.raw_train_Ys == (first.raw_train_Y, second.raw_train_Y)
    assert len(models.classification_metadata) == 2


def test_classification_model_list_rejects_empty_container() -> None:
    with pytest.raises(ValueError, match="at least one model"):
        ClassificationModelList()
