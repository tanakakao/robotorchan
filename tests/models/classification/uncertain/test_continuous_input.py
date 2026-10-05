"""Tests for continuous uncertain-input binary classification."""

import torch

from robotorchan.models.capabilities import RobustnessType
from robotorchan.models.classification.binary.registry import BINARY_CLASSIFICATION_MODEL_SPECS
from robotorchan.models.classification.binary.uncertain import (
    ClassificationInputUncertaintyType,
    ClassificationUncertaintyIntegration,
    ClassificationUncertaintyTarget,
    ContinuousUncertainInputBinarySingleTaskGPClassifier,
)


def _model() -> ContinuousUncertainInputBinarySingleTaskGPClassifier:
    train_x = torch.tensor(
        [[0.0], [0.25], [0.75], [1.0]],
        dtype=torch.double,
    )
    train_y = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.double)
    return ContinuousUncertainInputBinarySingleTaskGPClassifier(train_x, train_y)


def test_uncertain_input_metadata() -> None:
    model = _model()

    assert model.classification_input_uncertainty == frozenset(
        {ClassificationInputUncertaintyType.CONTINUOUS}
    )
    assert (
        model.classification_uncertainty_target
        is ClassificationUncertaintyTarget.CANDIDATE_INPUTS
    )
    assert (
        model.classification_uncertainty_integration
        is ClassificationUncertaintyIntegration.MONTE_CARLO
    )
    assert model.preserves_latent_classification_posterior is True


def test_zero_input_std_matches_standard_probability() -> None:
    model = _model()
    x = torch.tensor([[0.2], [0.8]], dtype=torch.double)
    standard = model.predict_proba(x)

    uncertain = model.predict_proba_with_input_uncertainty(
        x,
        input_std=torch.zeros_like(x),
        num_samples=16,
        generator=torch.Generator().manual_seed(7),
    )

    torch.testing.assert_close(uncertain, standard, atol=1e-6, rtol=1e-6)


def test_full_input_covariance_preserves_probability_shape_and_simplex() -> None:
    model = _model()
    x = torch.tensor([[0.2], [0.8]], dtype=torch.double)
    covariance = torch.full((2, 1, 1), 0.01, dtype=torch.double)

    probabilities = model.predict_proba_with_input_uncertainty(
        x,
        input_covar=covariance,
        num_samples=8,
        generator=torch.Generator().manual_seed(11),
    )

    assert probabilities.shape == (2, 2)
    torch.testing.assert_close(
        probabilities.sum(dim=-1),
        torch.ones(2, dtype=torch.double),
    )


def test_uncertain_probability_keeps_candidate_autograd() -> None:
    model = _model()
    x = torch.tensor([[0.4]], dtype=torch.double, requires_grad=True)

    probabilities = model.predict_proba_with_input_uncertainty(
        x,
        input_std=torch.full_like(x, 0.05),
        num_samples=4,
        generator=torch.Generator().manual_seed(13),
    )
    probabilities[..., 1].sum().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_registry_advertises_uncertain_input_capability() -> None:
    specs = {
        model_id: (family, capabilities)
        for model_id, _, family, capabilities in BINARY_CLASSIFICATION_MODEL_SPECS
    }
    family, capabilities = specs["binary.uncertain.continuous_input"]

    assert family == "uncertain"
    assert capabilities.robustness == frozenset({RobustnessType.UNCERTAIN_INPUT})
