"""Final cross-layer runtime contracts for classification workflows."""

import torch
from botorch.optim import optimize_acqf

from robotorchan.acquisition import BALD, PredictiveEntropy
from robotorchan.acquisition.classification_constraints import (
    ClassificationProbabilityOfFeasibility,
)
from robotorchan.acquisition.compatibility import (
    CompatibilityStatus,
    check_capabilities_acquisition_compatibility,
)
from robotorchan.models.classification import (
    BinarySingleTaskGPClassifier,
    CalibratedBinaryClassifier,
    TemperatureScalingCalibrator,
    get_classification_model_entry,
)


def _model() -> BinarySingleTaskGPClassifier:
    train_X = torch.tensor([[0.0], [0.25], [0.75], [1.0]], dtype=torch.double)
    train_Y = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    return BinarySingleTaskGPClassifier(train_X, train_Y, inducing_points=4)


def _bounds() -> torch.Tensor:
    return torch.tensor([[0.0], [1.0]], dtype=torch.double)


def test_probability_active_learning_optimizes_with_botorch_optimizer() -> None:
    model = _model()
    acquisition = PredictiveEntropy(model)

    candidate, value = optimize_acqf(
        acquisition,
        bounds=_bounds(),
        q=1,
        num_restarts=2,
        raw_samples=16,
        options={"maxiter": 20},
    )

    assert candidate.shape == torch.Size([1, 1])
    assert value.numel() == 1
    assert torch.isfinite(candidate).all()
    assert torch.isfinite(value).all()


def test_sampling_active_learning_optimizes_with_botorch_optimizer() -> None:
    model = _model()
    acquisition = BALD(model, num_samples=8)

    candidate, value = optimize_acqf(
        acquisition,
        bounds=_bounds(),
        q=1,
        num_restarts=2,
        raw_samples=16,
        options={"maxiter": 20},
    )

    assert candidate.shape == torch.Size([1, 1])
    assert value.numel() == 1
    assert torch.isfinite(candidate).all()
    assert torch.isfinite(value).all()


def test_calibrated_probability_of_feasibility_optimizes_end_to_end() -> None:
    model = _model()
    calibrated = CalibratedBinaryClassifier(
        model,
        TemperatureScalingCalibrator(temperature=1.5).double(),
    )
    acquisition = ClassificationProbabilityOfFeasibility(calibrated)

    candidate, value = optimize_acqf(
        acquisition,
        bounds=_bounds(),
        q=1,
        num_restarts=2,
        raw_samples=16,
        options={"maxiter": 20},
    )

    assert candidate.shape == torch.Size([1, 1])
    assert value.numel() == 1
    assert torch.isfinite(candidate).all()
    assert torch.isfinite(value).all()


def test_registry_rejects_sampling_acquisition_without_probability_samples() -> None:
    capabilities = get_classification_model_entry(
        "binary.non_gp.gradient_boosting"
    ).capabilities
    result = check_capabilities_acquisition_compatibility(capabilities, "BALD")

    assert result.status is CompatibilityStatus.INCOMPATIBLE
    assert "acquisition requires class-probability sampling support" in result.reasons


def test_registry_and_runtime_agree_for_standard_probability_sampling() -> None:
    capabilities = get_classification_model_entry("binary.standard").capabilities
    result = check_capabilities_acquisition_compatibility(capabilities, "BALD")
    model = _model()
    samples = model.sample_class_probabilities(
        torch.tensor([[0.5]], dtype=torch.double),
        torch.Size([4]),
    )

    assert result.status is CompatibilityStatus.COMPATIBLE
    assert capabilities.supports_probability_samples
    assert samples.shape == torch.Size([4, 1, 2])
    torch.testing.assert_close(
        samples.sum(dim=-1),
        torch.ones(4, 1, dtype=torch.double),
    )
