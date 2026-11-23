"""Smoke tests for reduced-input multiclass and ordinal classification."""

import pytest
import torch

from robotorchan.models.classification import (
    PCAMulticlassGPClassifier,
    PCAOrdinalGPClassifier,
    PLSMulticlassGPClassifier,
    PLSOrdinalGPClassifier,
    ReducedMulticlassGPClassifier,
    ReducedOrdinalGPClassifier,
    RandomProjectionMulticlassGPClassifier,
    RandomProjectionOrdinalGPClassifier,
)
from robotorchan.reduction.input import PCAInputReducer


@pytest.mark.parametrize(
    "model_type",
    [
        PCAMulticlassGPClassifier,
        PCAOrdinalGPClassifier,
        PLSMulticlassGPClassifier,
        PLSOrdinalGPClassifier,
        RandomProjectionMulticlassGPClassifier,
        RandomProjectionOrdinalGPClassifier,
    ],
)
def test_reduced_classifier_fit_predict_and_candidate(model_type):
    torch.manual_seed(7)
    x = torch.randn(18, 5, dtype=torch.double)
    y = torch.arange(18) % 3
    model = model_type(x, y, num_classes=3, n_components=2)
    assert model.original_input_dim == 5
    assert model.reduced_input_dim == 2
    model.fit(epochs=2, lr=0.01)
    candidates = torch.randn(7, 5, dtype=torch.double)
    with torch.no_grad():
        p = model.predict_proba(candidates)
        entropy = -torch.special.xlogy(p, p).sum(dim=-1)
        selected = candidates[entropy.argmax()]
    assert p.shape == (7, 3)
    torch.testing.assert_close(p.sum(-1), torch.ones(7, dtype=p.dtype), atol=1e-5, rtol=1e-5)
    assert selected.shape == (5,)
    with pytest.raises(ValueError, match="Expected input dimension"):
        model.posterior(torch.randn(2, 4, dtype=torch.double))


@pytest.mark.parametrize("model_type", [ReducedMulticlassGPClassifier, ReducedOrdinalGPClassifier])
def test_reuse_pretrained_reducer(model_type):
    x = torch.randn(12, 4, dtype=torch.double)
    y = torch.arange(12) % 3
    reducer = PCAInputReducer(n_components=2)
    reducer.fit_transform(x, y)
    model = model_type(x, y, num_classes=3, input_reducer=reducer)
    assert model.reduced_input_dim == 2
    assert model.posterior(x[:3]).mean.shape[-2] == 3
