"""Phase 12: candidate-input uncertainty for multiclass and ordinal GP models."""

import pytest
import torch

from robotorchan.acquisition.active_learning.classification import PredictiveEntropy
from robotorchan.models.classification import (
    ContinuousUncertainInputMulticlassGPClassifier,
    ContinuousUncertainInputOrdinalGPClassifier,
    UncertainCategoricalMulticlassGPClassifier,
    UncertainCategoricalOrdinalGPClassifier,
)


@pytest.mark.parametrize(
    "model_type",
    [
        ContinuousUncertainInputMulticlassGPClassifier,
        ContinuousUncertainInputOrdinalGPClassifier,
    ],
)
def test_continuous_uncertain_gp_fit_probabilities_and_candidate(model_type):
    torch.manual_seed(47)
    x = torch.rand(12, 2, dtype=torch.double)
    y = torch.arange(12) % 3
    model = model_type(x, y, num_classes=3)
    model.fit(epochs=2, lr=0.005)
    pool = torch.rand(6, 2, dtype=torch.double)
    generator = torch.Generator().manual_seed(13)
    with torch.no_grad():
        samples = model.sample_uncertain_inputs(
            pool,
            input_std=torch.full_like(pool, 0.04),
            num_samples=5,
            generator=generator,
        )
        uncertain = model.predict_proba_with_input_uncertainty(
            pool,
            input_std=torch.full_like(pool, 0.04),
            num_samples=5,
            generator=torch.Generator().manual_seed(13),
        )
        stable = model.predict_proba(pool)
        scores = PredictiveEntropy(model)(pool.unsqueeze(-2))
    assert samples.shape == (5, 6, 2)
    assert uncertain.shape == stable.shape == (6, 3)
    assert scores.shape == (6,)
    torch.testing.assert_close(
        uncertain.sum(-1), torch.ones(6, dtype=torch.double), atol=1e-5, rtol=1e-5
    )
    assert torch.isfinite(uncertain).all()
    assert pool[scores.argmax()].shape == (2,)
    assert model.uncertain_classification_metadata.preserves_latent_classification_posterior


@pytest.mark.parametrize(
    "model_type",
    [
        ContinuousUncertainInputMulticlassGPClassifier,
        ContinuousUncertainInputOrdinalGPClassifier,
    ],
)
def test_continuous_input_covariance_and_invalid_uncertainty(model_type):
    x = torch.rand(9, 2, dtype=torch.double)
    y = torch.arange(9) % 3
    model = model_type(x, y, num_classes=3)
    pool = x[:4]
    diag = torch.diag_embed(torch.full_like(pool, 0.01))
    with torch.no_grad():
        assert model.sample_uncertain_inputs(pool, input_covar=diag, num_samples=3).shape == (
            3, 4, 2
        )
    with pytest.raises(ValueError, match="Exactly one"):
        model.sample_uncertain_inputs(pool)
    with pytest.raises(ValueError, match="nonnegative"):
        model.sample_uncertain_inputs(pool, input_std=-torch.ones_like(pool))
    with pytest.raises(ValueError, match="positive semidefinite"):
        model.sample_uncertain_inputs(
            pool,
            input_covar=torch.diag_embed(
                torch.tensor([-0.1, 0.1], dtype=pool.dtype).expand_as(pool)
            ),
        )


@pytest.mark.parametrize(
    "model_type",
    [UncertainCategoricalMulticlassGPClassifier, UncertainCategoricalOrdinalGPClassifier],
)
def test_uncertain_category_fit_exact_weighted_predictions_and_candidate(model_type):
    torch.manual_seed(51)
    x = torch.rand(12, 2, dtype=torch.double)
    x[:, 1] = (torch.arange(12) % 2).to(torch.double)
    y = torch.arange(12) % 3
    model = model_type(x, y, num_classes=3, cat_dims=[1], uncertain_cat_dim=1)
    model.fit(epochs=2, lr=0.005)
    pool = x[:5].clone()
    categories = torch.tensor([0., 1.], dtype=torch.double)
    weights = torch.tensor([[0.3, 0.7]], dtype=torch.double).expand(5, 2)
    with torch.no_grad():
        torch.manual_seed(17)
        mixed = model.predict_proba_with_category_uncertainty(
            pool,
            category_values=categories,
            category_probabilities=weights,
        )
        expanded = pool.unsqueeze(-2).expand(5, 2, 2).clone()
        expanded[..., 1] = categories
        torch.manual_seed(17)
        direct = model.predict_proba(expanded)
        scores = PredictiveEntropy(model)(pool.unsqueeze(-2))
    assert mixed.shape == (5, 3)
    torch.testing.assert_close(
        mixed, (direct * weights.unsqueeze(-1)).sum(dim=-2), atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        mixed.sum(-1), torch.ones(5, dtype=torch.double), atol=1e-5, rtol=1e-5
    )
    assert torch.isfinite(mixed).all()
    assert scores.shape == (5,)
    assert pool[scores.argmax()].shape == (2,)


@pytest.mark.parametrize(
    "model_type",
    [UncertainCategoricalMulticlassGPClassifier, UncertainCategoricalOrdinalGPClassifier],
)
def test_uncertain_category_rejects_invalid_support(model_type):
    x = torch.rand(9, 2, dtype=torch.double)
    x[:, 1] = torch.arange(9) % 2
    y = torch.arange(9) % 3
    with pytest.raises(ValueError, match="categorical dimension"):
        model_type(x, y, num_classes=3, cat_dims=[1], uncertain_cat_dim=0)
    with pytest.raises(ValueError, match="out of range"):
        model_type(x, y, num_classes=3, cat_dims=[1], uncertain_cat_dim=5)
    model = model_type(x, y, num_classes=3, cat_dims=[1], uncertain_cat_dim=1)
    with pytest.raises(ValueError, match="sum to one"):
        model.predict_proba_with_category_uncertainty(
            x[:3],
            category_values=torch.tensor([0., 1.], dtype=torch.double),
            category_probabilities=torch.ones(3, 2, dtype=torch.double),
        )
