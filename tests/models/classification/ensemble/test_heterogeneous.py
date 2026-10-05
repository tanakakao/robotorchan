"""Tests for heterogeneous binary classification ensembles."""

import pytest
import torch

from robotorchan.acquisition.compatibility import (
    CompatibilityStatus,
    check_capabilities_acquisition_compatibility,
)
from robotorchan.models.capabilities import InferenceType, PosteriorSamplingType
from robotorchan.models.classification.binary.ensemble.heterogeneous import (
    HeterogeneousBinaryClassificationEnsemble,
)
from robotorchan.models.classification.binary.non_gp.sklearn import (
    RandomForestBinaryClassifier,
)
from robotorchan.models.classification.binary.standard.single_task import (
    BinarySingleTaskGPClassifier,
)
from robotorchan.models.classification.registry import get_classification_model_entry


def _members() -> tuple[BinarySingleTaskGPClassifier, RandomForestBinaryClassifier]:
    X = torch.linspace(0.0, 1.0, 12, dtype=torch.double).unsqueeze(-1)
    y = (X[:, 0] > 0.5).long()
    gp = BinarySingleTaskGPClassifier(X, y, inducing_points=4)
    forest = RandomForestBinaryClassifier(X, y, n_estimators=8, random_state=3)
    forest.fit()
    return gp, forest


def test_heterogeneous_ensemble_combines_gp_and_non_gp_probabilities() -> None:
    gp, forest = _members()
    ensemble = HeterogeneousBinaryClassificationEnsemble(gp, forest)
    X = gp.raw_train_X[:4]
    expected = torch.stack([gp.predict_proba(X), forest.predict_proba(X)])

    posterior = ensemble.probability_posterior(X)

    torch.testing.assert_close(posterior.probabilities, expected)
    torch.testing.assert_close(ensemble.predict_proba(X), expected.mean(dim=0))
    torch.testing.assert_close(
        ensemble.probability_variance(X),
        expected.var(dim=0, unbiased=False),
    )


def test_heterogeneous_ensemble_rejects_shared_latent_semantics() -> None:
    gp, forest = _members()
    ensemble = HeterogeneousBinaryClassificationEnsemble(gp, forest)

    with pytest.raises(NotImplementedError, match="do not share one latent posterior"):
        ensemble.latent_posterior(gp.raw_train_X[:2])
    with pytest.raises(NotImplementedError, match="do not share a latent sample space"):
        ensemble.sample_latent(gp.raw_train_X[:2])


def test_heterogeneous_ensemble_supports_probability_space_active_learning() -> None:
    capabilities = get_classification_model_entry("binary.ensemble.heterogeneous").capabilities

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


def test_heterogeneous_ensemble_registry_is_backend_neutral() -> None:
    capabilities = get_classification_model_entry("binary.ensemble.heterogeneous").capabilities

    assert capabilities.inference is InferenceType.NOT_APPLICABLE
    assert not capabilities.non_gp
    assert capabilities.ensemble_posterior
    assert capabilities.posterior_sampling_type is PosteriorSamplingType.ENSEMBLE


def test_heterogeneous_ensemble_applies_member_weights() -> None:
    gp, forest = _members()
    weights = torch.tensor([0.8, 0.2], dtype=torch.double)
    ensemble = HeterogeneousBinaryClassificationEnsemble(gp, forest, weights=weights)
    X = gp.raw_train_X[:4]
    member_probabilities = torch.stack([gp.predict_proba(X), forest.predict_proba(X)])
    expected = (member_probabilities * weights[:, None, None]).sum(dim=0)

    torch.testing.assert_close(ensemble.member_weights, weights)
    torch.testing.assert_close(ensemble.predict_proba(X), expected)
    torch.testing.assert_close(
        ensemble.probability_posterior(X).weights,
        weights,
    )


def test_heterogeneous_ensemble_weights_follow_module_state() -> None:
    gp, forest = _members()
    ensemble = HeterogeneousBinaryClassificationEnsemble(
        gp,
        forest,
        weights=torch.tensor([2.0, 1.0]),
    )

    assert "member_weights" in ensemble.state_dict()
    torch.testing.assert_close(
        ensemble.member_weights,
        torch.tensor([2.0 / 3.0, 1.0 / 3.0]),
    )


def test_heterogeneous_ensemble_preserves_q_one_candidate_axis() -> None:
    gp, forest = _members()
    ensemble = HeterogeneousBinaryClassificationEnsemble(gp, forest)
    X = gp.raw_train_X[:2].unsqueeze(-2)

    posterior = ensemble.probability_posterior(X)

    assert posterior.probabilities.shape == torch.Size([2, 2, 1, 2])
    assert ensemble.predict_proba(X).shape == torch.Size([2, 1, 2])
