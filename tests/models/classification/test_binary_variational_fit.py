"""Fit contract for binary variational GP classifiers."""

import pytest
import torch

from robotorchan.models.classification import (
    BinarySingleTaskGPClassifier,
    MixedBinarySingleTaskGPClassifier,
    MultiTaskBinaryGPClassifier,
)


@pytest.mark.parametrize(
    "model_type",
    [
        BinarySingleTaskGPClassifier,
        MixedBinarySingleTaskGPClassifier,
        MultiTaskBinaryGPClassifier,
    ],
)
def test_binary_variational_fit_preserves_probability_contract(model_type: type) -> None:
    torch.manual_seed(0)
    X = torch.rand(12, 2, dtype=torch.double)
    Y = (X[:, 0] > 0.5).to(dtype=torch.double)
    kwargs = {"inducing_points": 5}
    if model_type is MixedBinarySingleTaskGPClassifier:
        X[:, 1] = (X[:, 1] > 0.5).to(dtype=X.dtype)
        kwargs["cat_dims"] = [1]
    elif model_type is MultiTaskBinaryGPClassifier:
        X[:, 1] = (X[:, 1] > 0.5).to(dtype=X.dtype)
        kwargs["task_feature"] = 1
    model = model_type(X, Y, **kwargs)
    assert model.supports_fit
    assert model.fit(epochs=2, lr=0.01, batch_size=6) is None
    assert not model.training
    assert not model.likelihood.training
    probabilities = model.predict_proba(X[:3])
    assert probabilities.shape == (3, 2)
    assert torch.isfinite(probabilities).all()
    torch.testing.assert_close(
        probabilities.sum(dim=-1), torch.ones(3, dtype=X.dtype)
    )
