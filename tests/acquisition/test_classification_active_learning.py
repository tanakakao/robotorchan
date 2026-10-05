"""Tests for classification active-learning acquisitions."""

import torch

from robotorchan.acquisition import (
    BALD,
    LatentStraddle,
    MarginUncertainty,
    PredictiveEntropy,
    ProbabilityVariance,
)
from robotorchan.acquisition.capabilities import AcquisitionTarget
from robotorchan.acquisition.compatibility import (
    CompatibilityStatus,
    check_capabilities_acquisition_compatibility,
)
from robotorchan.acquisition.registry import get_acquisition_registry_entry
from robotorchan.models.classification import (
    BinarySingleTaskGPClassifier,
    get_classification_model_entry,
)


def _model() -> BinarySingleTaskGPClassifier:
    train_X = torch.tensor([[0.0], [0.3], [0.7], [1.0]], dtype=torch.double)
    train_Y = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.double)
    return BinarySingleTaskGPClassifier(train_X, train_Y, inducing_points=4)


def test_classification_acquisitions_return_one_score_per_candidate() -> None:
    model = _model()
    X = torch.tensor([[[0.5]], [[0.8]]], dtype=torch.double)
    acquisitions = (
        PredictiveEntropy(model),
        MarginUncertainty(model),
        ProbabilityVariance(model, num_samples=8),
        BALD(model, num_samples=8),
        LatentStraddle(model),
    )
    for acquisition in acquisitions:
        value = acquisition(X)
        assert value.shape == torch.Size([2])
        assert torch.isfinite(value).all()


def test_entropy_margin_and_bald_are_non_negative() -> None:
    model = _model()
    X = torch.tensor([[[0.5]]], dtype=torch.double)
    assert PredictiveEntropy(model)(X).item() >= 0.0
    assert MarginUncertainty(model)(X).item() >= 0.0
    assert BALD(model, num_samples=16)(X).item() >= 0.0


def test_classification_registry_targets_are_explicit() -> None:
    assert (
        get_acquisition_registry_entry("PredictiveEntropy").capabilities.target
        is AcquisitionTarget.LABEL_UNCERTAINTY
    )
    assert (
        get_acquisition_registry_entry("ProbabilityVariance").capabilities.target
        is AcquisitionTarget.CLASS_PROBABILITY
    )
    assert (
        get_acquisition_registry_entry("LatentStraddle").capabilities.target
        is AcquisitionTarget.LATENT
    )


def test_binary_classifier_accepts_classification_acquisitions() -> None:
    capabilities = get_classification_model_entry("binary.standard").capabilities
    for name in (
        "PredictiveEntropy",
        "MarginUncertainty",
        "ProbabilityVariance",
        "BALD",
        "LatentStraddle",
    ):
        result = check_capabilities_acquisition_compatibility(capabilities, name)
        assert result.status is CompatibilityStatus.COMPATIBLE


def test_classification_acquisition_rejects_regression_model() -> None:
    from robotorchan.models.registry import MODEL_REGISTRY

    capabilities = MODEL_REGISTRY["SingleTaskGP"].capabilities
    result = check_capabilities_acquisition_compatibility(capabilities, "BALD")
    assert result.status is CompatibilityStatus.INCOMPATIBLE
    assert "acquisition does not support regression observations" in result.reasons


def test_classification_acquisitions_require_q_one() -> None:
    model = _model()
    X = torch.tensor([[[0.3], [0.7]]], dtype=torch.double)
    try:
        PredictiveEntropy(model)(X)
    except ValueError as error:
        assert "q=1" in str(error)
    else:
        raise AssertionError("PredictiveEntropy must reject q > 1")


def test_active_learning_matches_model_uncertainty_contract() -> None:
    model = _model()
    X = torch.tensor([[[0.5]], [[0.8]]], dtype=torch.double)

    probability_variance = ProbabilityVariance(model, num_samples=16)(X)
    torch.manual_seed(123)
    expected_variance = model.probability_variance(X, num_samples=16)
    expected_variance = expected_variance.mean(dim=-1).squeeze(-1)

    torch.manual_seed(123)
    probability_variance = ProbabilityVariance(model, num_samples=16)(X)
    torch.testing.assert_close(probability_variance, expected_variance)

    torch.manual_seed(456)
    expected_bald = model.mutual_information(X, num_samples=16).squeeze(-1)
    torch.manual_seed(456)
    actual_bald = BALD(model, num_samples=16)(X)
    torch.testing.assert_close(actual_bald, expected_bald)
