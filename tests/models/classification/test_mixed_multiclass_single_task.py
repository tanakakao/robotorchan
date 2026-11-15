"""Mixed multiclass GP classifier smoke and input-contract tests."""

import pytest
import torch

from robotorchan.models.classification import (
    MixedMulticlassSingleTaskGPClassifier,
    get_classification_model_entry,
)


def test_mixed_multiclass_fit_and_probability_contract() -> None:
    torch.manual_seed(13)
    X = torch.rand(18, 2, dtype=torch.double)
    X[:, 1] = torch.arange(18) % 3
    y = (torch.arange(18) % 3).long()
    model = MixedMulticlassSingleTaskGPClassifier(
        X, y, num_classes=3, cat_dims=[1], inducing_points=8
    )
    model.fit(epochs=2, lr=0.01)
    test_X = X[:4].unsqueeze(0)
    assert model.posterior(test_X).mean.shape == (1, 4, 3)
    probabilities = model.predict_proba(test_X, num_samples=16)
    assert probabilities.shape == (1, 4, 3)
    assert torch.isfinite(probabilities).all()
    assert torch.allclose(
        probabilities.sum(dim=-1), torch.ones_like(probabilities[..., 0])
    )
    assert model.cat_dims == (1,)


def test_mixed_multiclass_registry_and_invalid_cat_dims() -> None:
    entry = get_classification_model_entry("multiclass.mixed")
    assert entry.model_class is MixedMulticlassSingleTaskGPClassifier
    X = torch.rand(6, 2)
    y = (torch.arange(6) % 3).long()
    with pytest.raises(ValueError):
        MixedMulticlassSingleTaskGPClassifier(
            X, y, num_classes=3, cat_dims=[2]
        )
