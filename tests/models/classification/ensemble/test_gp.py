"""Tests for homogeneous GP binary classification ensembles."""

import pytest
import torch

from robotorchan.acquisition.compatibility import (
    CompatibilityStatus,
    check_capabilities_acquisition_compatibility,
)
from robotorchan.models.capabilities import PosteriorSamplingType
from robotorchan.models.classification.binary.ensemble.gp import (
    GPBinaryClassificationEnsemble,
)
from robotorchan.models.classification.binary.standard.single_task import (
    BinarySingleTaskGPClassifier,
)
from robotorchan.models.classification.registry import get_classification_model_entry


def _members() -> tuple[BinarySingleTaskGPClassifier, BinarySingleTaskGPClassifier]:
    X = torch.linspace(0.0, 1.0, 8, dtype=torch.double).unsqueeze(-1)
    y = (X[:, 0] > 0.5).long()
    return (
        BinarySingleTaskGPClassifier(X, y, inducing_points=4),
        BinarySingleTaskGPClassifier(X, y, inducing_points=4),
    )


def test_gp_ensemble_probability_posterior_matches_members() -> None:
    members = _members()
    ensemble = GPBinaryClassificationEnsemble(*members)
    X = members[0].raw_train_X[:3]
    expected = torch.stack([member.predict_proba(X) for member in members])

    posterior = ensemble.probability_posterior(X)

    torch.testing.assert_close(posterior.probabilities, expected)
    torch.testing.assert_close(ensemble.predict_proba(X), expected.mean(dim=0))
    torch.testing.assert_close(
        ensemble.probability_variance(X),
        expected.var(dim=0, unbiased=False),
    )


def test_gp_ensemble_keeps_member_latent_posteriors_separate() -> None:
    members = _members()
    ensemble = GPBinaryClassificationEnsemble(*members)
    X = members[0].raw_train_X[:2]

    posteriors = ensemble.member_latent_posteriors(X)

    assert len(posteriors) == 2
    with pytest.raises(NotImplementedError, match="multiple independent latent posteriors"):
        ensemble.latent_posterior(X)


def test_gp_ensemble_probability_samples_and_information() -> None:
    ensemble = GPBinaryClassificationEnsemble(*_members())
    X = ensemble.raw_train_X[:2]
    samples = ensemble.sample_class_probabilities(X, torch.Size([5]))

    assert samples.shape == (5, 2, 2)
    torch.testing.assert_close(samples.sum(dim=-1), torch.ones(5, 2, dtype=X.dtype))
    assert torch.all(ensemble.mutual_information(X) >= 0.0)


def test_gp_ensemble_requires_matching_training_data() -> None:
    first, _ = _members()
    X = first.raw_train_X.clone()
    y = first.raw_train_Y.clone()
    X[0, 0] += 0.1
    different = BinarySingleTaskGPClassifier(X, y, inducing_points=4)

    with pytest.raises(ValueError, match="same training inputs"):
        GPBinaryClassificationEnsemble(first, different)


def test_gp_ensemble_registry_capabilities() -> None:
    capabilities = get_classification_model_entry("binary.ensemble.gp").capabilities

    assert not capabilities.non_gp
    assert capabilities.ensemble_posterior
    assert capabilities.supports_posterior_samples
    assert capabilities.posterior_sampling_type is PosteriorSamplingType.ENSEMBLE


def test_gp_ensemble_supports_probability_space_active_learning() -> None:
    capabilities = get_classification_model_entry("binary.ensemble.gp").capabilities

    for acquisition_name in (
        "BALD",
        "MarginUncertainty",
        "PredictiveEntropy",
        "ProbabilityVariance",
    ):
        result = check_capabilities_acquisition_compatibility(
            capabilities,
            acquisition_name,
        )
        assert result.status is CompatibilityStatus.COMPATIBLE

    latent = check_capabilities_acquisition_compatibility(capabilities, "LatentStraddle")
    assert latent.status is CompatibilityStatus.INCOMPATIBLE
