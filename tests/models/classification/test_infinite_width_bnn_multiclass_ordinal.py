"""Phase 8 parity: infinite-width ReLU NNGP multiclass and ordinal classifiers."""

import pytest
import torch
from gpytorch.kernels import ScaleKernel

from robotorchan.acquisition.active_learning.classification import (
    PredictiveEntropy,
    qPredictiveEntropy,
)
from robotorchan.models.classification import (
    InfiniteWidthBNNMulticlassGPClassifier,
    InfiniteWidthBNNOrdinalGPClassifier,
    MixedInfiniteWidthBNNMulticlassGPClassifier,
    MixedInfiniteWidthBNNOrdinalGPClassifier,
)
from robotorchan.models.expressive.infinite_width_bnn import InfiniteWidthReLUKernel


@pytest.mark.parametrize(
    "model_type",
    [
        InfiniteWidthBNNMulticlassGPClassifier,
        InfiniteWidthBNNOrdinalGPClassifier,
        MixedInfiniteWidthBNNMulticlassGPClassifier,
        MixedInfiniteWidthBNNOrdinalGPClassifier,
    ],
)
def test_infinite_width_bnn_classification_fit_predict_and_propose(model_type):
    torch.manual_seed(63)
    x = torch.rand(12, 3, dtype=torch.double)
    mixed = model_type in (
        MixedInfiniteWidthBNNMulticlassGPClassifier,
        MixedInfiniteWidthBNNOrdinalGPClassifier,
    )
    if mixed:
        x[:, 1] = (torch.arange(12) % 2).to(torch.double)
    y = torch.arange(12) % 3
    kwargs = {"cat_dims": [1]} if mixed else {}
    model = model_type(x, y, num_classes=3, depth=2, **kwargs)
    if mixed:
        assert model.cat_dims == (1,)
    else:
        kernel = (
            model.class_models[0].model.covar_module
            if isinstance(model, InfiniteWidthBNNMulticlassGPClassifier)
            else model.model.covar_module
        )
        assert isinstance(kernel, ScaleKernel)
        assert isinstance(kernel.base_kernel, InfiniteWidthReLUKernel)

    model.fit(epochs=2, lr=0.005)
    pool = x[:8]
    with torch.no_grad():
        probs = model.predict_proba(pool)
        posterior_kwargs = (
            {"num_samples": 16, "seed": 42}
            if isinstance(
                model,
                (
                    InfiniteWidthBNNMulticlassGPClassifier,
                    MixedInfiniteWidthBNNMulticlassGPClassifier,
                ),
            )
            else {}
        )
        posterior = model.probability_posterior(pool, **posterior_kwargs)
        acquisition = PredictiveEntropy(model)(pool.unsqueeze(-2))
        batch_scores = qPredictiveEntropy(model)(pool.reshape(4, 2, 3))
    assert probs.shape == (8, 3)
    assert posterior.mean.shape == (8, 3)
    assert posterior.variance.shape == (8, 3)
    assert acquisition.shape == (8,)
    assert batch_scores.shape == (4,)
    assert torch.isfinite(probs).all()
    assert torch.isfinite(posterior.variance).all()
    torch.testing.assert_close(
        probs.sum(dim=-1),
        torch.ones(8, dtype=torch.double),
        atol=1e-5,
        rtol=1e-5,
    )
    assert pool[acquisition.argmax()].shape == (3,)


@pytest.mark.parametrize(
    "model_type",
    [MixedInfiniteWidthBNNMulticlassGPClassifier, MixedInfiniteWidthBNNOrdinalGPClassifier],
)
def test_mixed_infinite_width_requires_continuous_and_categorical_features(model_type):
    x = torch.rand(6, 2, dtype=torch.double)
    y = torch.arange(6) % 3
    with pytest.raises(ValueError, match="continuous feature"):
        model_type(x, y, num_classes=3, cat_dims=[0, 1])
    with pytest.raises(ValueError, match="categorical features"):
        model_type(x, y, num_classes=3, cat_dims=[])


@pytest.mark.parametrize(
    "model_type",
    [InfiniteWidthBNNMulticlassGPClassifier, InfiniteWidthBNNOrdinalGPClassifier],
)
def test_infinite_width_rejects_invalid_depth(model_type):
    x = torch.rand(6, 2, dtype=torch.double)
    y = torch.arange(6) % 3
    with pytest.raises(ValueError, match="depth"):
        model_type(x, y, num_classes=3, depth=0)
