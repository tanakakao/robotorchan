"""Tests for deterministic non-GP binary classifier adapters."""

import pytest
import torch

from robotorchan.models.classification import (
    CLASSIFICATION_MODEL_REGISTRY,
    ExtraTreesBinaryClassifier,
    GradientBoostingBinaryClassifier,
    HistGradientBoostingBinaryClassifier,
    RandomForestBinaryClassifier,
)


def _training_data() -> tuple[torch.Tensor, torch.Tensor]:
    X = torch.linspace(-2.0, 2.0, 20, dtype=torch.double).unsqueeze(-1)
    y = (X[:, 0] > 0).long()
    return X, y


@pytest.mark.parametrize(
    "model_class",
    [
        RandomForestBinaryClassifier,
        ExtraTreesBinaryClassifier,
        GradientBoostingBinaryClassifier,
        HistGradientBoostingBinaryClassifier,
    ],
)
def test_non_gp_classifier_probability_contract(model_class) -> None:
    X, y = _training_data()
    model = model_class(X, y, random_state=0)
    model.fit()

    probabilities = model.predict_proba(X[:4])

    assert probabilities.shape == (4, 2)
    torch.testing.assert_close(
        probabilities.sum(dim=-1),
        torch.ones(4, dtype=torch.double),
    )
    assert torch.all((probabilities >= 0) & (probabilities <= 1))
    assert model.predictive_entropy(X[:4]).shape == (4,)
    assert model.predictive_variance(X[:4]).shape == (4, 2)


def test_single_non_gp_classifier_rejects_epistemic_sampling() -> None:
    X, y = _training_data()
    model = RandomForestBinaryClassifier(X, y, n_estimators=8, random_state=0)
    model.fit()

    with pytest.raises(NotImplementedError, match="posterior probability samples"):
        model.sample_class_probabilities(X[:2], torch.Size([4]))
    with pytest.raises(NotImplementedError, match="epistemic probability variance"):
        model.probability_variance(X[:2])


def test_non_gp_classifier_requires_fit_before_prediction() -> None:
    X, y = _training_data()
    model = RandomForestBinaryClassifier(X, y, n_estimators=8, random_state=0)

    with pytest.raises(RuntimeError, match=r"fit\(\)"):
        model.predict_proba(X[:2])


def test_non_gp_registry_does_not_claim_posterior_sampling() -> None:
    ids = (
        "binary.non_gp.random_forest",
        "binary.non_gp.extra_trees",
        "binary.non_gp.gradient_boosting",
        "binary.non_gp.hist_gradient_boosting",
    )
    for model_id in ids:
        entry = CLASSIFICATION_MODEL_REGISTRY[model_id]
        assert entry.family == "non_gp"
        assert entry.capabilities.non_gp
        assert not entry.capabilities.supports_posterior_samples
