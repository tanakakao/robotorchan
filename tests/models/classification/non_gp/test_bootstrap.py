"""Tests for bootstrap non-GP binary classification ensembles."""

import pytest
import torch

from robotorchan.models.capabilities import PosteriorSamplingType
from robotorchan.models.classification.binary.non_gp import (
    BootstrapGradientBoostingBinaryClassifier,
)
from robotorchan.models.classification.registry import get_classification_model_entry


def _training_data() -> tuple[torch.Tensor, torch.Tensor]:
    X = torch.linspace(0.0, 1.0, 30, dtype=torch.double).unsqueeze(-1)
    y = (X[:, 0] > 0.5).long()
    return X, y


def test_bootstrap_gradient_boosting_probability_contract() -> None:
    X, y = _training_data()
    model = BootstrapGradientBoostingBinaryClassifier(
        X,
        y,
        n_members=6,
        random_state=3,
        n_estimators=8,
    )
    model.fit()
    query = X[10:15]
    probabilities = model.predict_proba(query)
    samples = model.sample_class_probabilities(query, torch.Size([9]))

    assert probabilities.shape == (5, 2)
    assert samples.shape == (9, 5, 2)
    torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(5, dtype=X.dtype))
    torch.testing.assert_close(samples.sum(dim=-1), torch.ones(9, 5, dtype=X.dtype))


def test_bootstrap_uncertainty_matches_member_distribution() -> None:
    X, y = _training_data()
    model = BootstrapGradientBoostingBinaryClassifier(
        X,
        y,
        n_members=6,
        random_state=4,
        n_estimators=5,
    )
    model.fit()
    query = X[12:18]
    members = model._member_probabilities(query)

    torch.testing.assert_close(model.predict_proba(query), members.mean(dim=0))
    torch.testing.assert_close(
        model.probability_variance(query),
        members.var(dim=0, unbiased=False),
    )
    mutual_information = model.mutual_information(query)
    assert mutual_information.shape == (6,)
    assert torch.all(mutual_information >= -1e-12)


def test_bootstrap_requires_at_least_two_members() -> None:
    X, y = _training_data()
    with pytest.raises(ValueError, match="at least 2"):
        BootstrapGradientBoostingBinaryClassifier(X, y, n_members=1)


def test_bootstrap_registry_capabilities_are_truthful() -> None:
    entry = get_classification_model_entry("binary.non_gp.bootstrap_gradient_boosting")
    capabilities = entry.capabilities

    assert capabilities.non_gp
    assert capabilities.ensemble_posterior
    assert capabilities.supports_posterior_samples
    assert capabilities.posterior_sampling_type is PosteriorSamplingType.ENSEMBLE
