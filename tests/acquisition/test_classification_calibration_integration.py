"""Cross-layer tests for calibrated classification in BO and active learning."""

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from torch import Tensor, nn

from robotorchan.acquisition import (
    BALD,
    LatentStraddle,
    MarginUncertainty,
    PredictiveEntropy,
    ProbabilityVariance,
)
from robotorchan.acquisition.classification_constraints import (
    ClassificationProbabilityOfFeasibility,
    FeasibilityWeightedAcquisition,
)
from robotorchan.models.classification import (
    BinarySingleTaskGPClassifier,
    CalibratedBinaryClassifier,
    TemperatureScalingCalibrator,
)


class _ConstantObjectiveAcquisition(AcquisitionFunction):
    def __init__(self, model: nn.Module, value: float = 2.0) -> None:
        super().__init__(model=model)
        self.value = value

    def forward(self, X: Tensor) -> Tensor:
        return X.new_full(X.shape[:-2], self.value)


def _models() -> tuple[
    BinarySingleTaskGPClassifier,
    CalibratedBinaryClassifier,
]:
    train_X = torch.tensor([[0.0], [0.25], [0.75], [1.0]], dtype=torch.double)
    train_Y = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    base = BinarySingleTaskGPClassifier(train_X, train_Y, inducing_points=4)
    calibrated = CalibratedBinaryClassifier(
        base,
        TemperatureScalingCalibrator(temperature=2.0).double(),
    )
    return base, calibrated


def test_calibrated_probability_of_feasibility_uses_calibrated_probability() -> None:
    base, calibrated = _models()
    X = torch.tensor([[[0.2]], [[0.8]]], dtype=torch.double)
    pof = ClassificationProbabilityOfFeasibility(calibrated)

    torch.testing.assert_close(pof(X), calibrated.predict_proba(X)[..., 1])
    assert not torch.allclose(pof(X), base.predict_proba(X)[..., 1])


def test_feasibility_weighted_acquisition_uses_calibrated_pof() -> None:
    _, calibrated = _models()
    X = torch.tensor([[[0.2]], [[0.8]]], dtype=torch.double)
    objective = _ConstantObjectiveAcquisition(calibrated)
    pof = ClassificationProbabilityOfFeasibility(calibrated)
    acquisition = FeasibilityWeightedAcquisition(objective, pof)

    expected = objective(X) * pof(X).squeeze(-1)
    torch.testing.assert_close(acquisition(X), expected)


def test_probability_active_learning_uses_calibrated_contract() -> None:
    _, calibrated = _models()
    X = torch.tensor([[[0.35]], [[0.65]]], dtype=torch.double)

    entropy = PredictiveEntropy(calibrated)(X)
    margin = MarginUncertainty(calibrated)(X)
    variance = ProbabilityVariance(calibrated, num_samples=16)(X)
    bald = BALD(calibrated, num_samples=16)(X)

    assert torch.isfinite(entropy).all()
    assert torch.isfinite(margin).all()
    assert torch.isfinite(variance).all()
    assert torch.isfinite(bald).all()


def test_latent_straddle_is_invariant_to_post_hoc_probability_calibration() -> None:
    base, calibrated = _models()
    X = torch.tensor([[[0.4]], [[0.6]]], dtype=torch.double)

    torch.testing.assert_close(LatentStraddle(base)(X), LatentStraddle(calibrated)(X))


def test_calibrated_pof_remains_differentiable_for_candidate_optimization() -> None:
    _, calibrated = _models()
    X = torch.tensor([[[0.45]]], dtype=torch.double, requires_grad=True)
    value = ClassificationProbabilityOfFeasibility(calibrated)(X).sum()

    value.backward()

    assert X.grad is not None
    assert torch.isfinite(X.grad).all()
