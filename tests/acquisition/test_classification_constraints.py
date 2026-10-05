"""Tests for classifier-backed feasibility acquisitions."""

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from torch import nn

from robotorchan.acquisition.classification_constraints import (
    ClassificationProbabilityOfFeasibility,
    FeasibilityWeightedAcquisition,
    RobustClassificationProbabilityOfFeasibility,
)


class _Classifier(nn.Module):
    num_classes = 2

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        positive = torch.sigmoid(X[..., 0])
        return torch.stack((1.0 - positive, positive), dim=-1)


class _ObjectiveAcquisition(AcquisitionFunction):
    def __init__(self) -> None:
        super().__init__(model=None)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return X[..., 0].sum(dim=-1)


def test_classifier_probability_of_feasibility_selects_feasible_class() -> None:
    X = torch.tensor([[-1.0], [1.0]], dtype=torch.double)
    pof = ClassificationProbabilityOfFeasibility(_Classifier(), feasible_class=1)

    torch.testing.assert_close(pof(X), torch.sigmoid(X[..., 0]))


def test_robust_probability_of_feasibility_uses_worst_scenario() -> None:
    def scenarios(X: torch.Tensor) -> torch.Tensor:
        offsets = torch.tensor([-0.5, 0.5], dtype=X.dtype, device=X.device)
        return X.unsqueeze(-2) + offsets.view(1, 2, 1)

    X = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    pof = RobustClassificationProbabilityOfFeasibility(
        _Classifier(),
        scenario_generator=scenarios,
        risk_type="worst_case",
    )

    expected = torch.sigmoid(X[..., 0] - 0.5)
    torch.testing.assert_close(pof(X), expected)


def test_feasibility_weighted_acquisition_multiplies_joint_q_probability() -> None:
    X = torch.tensor([[[0.2], [0.8]]], dtype=torch.double)
    objective = _ObjectiveAcquisition()
    pof = ClassificationProbabilityOfFeasibility(_Classifier())
    acquisition = FeasibilityWeightedAcquisition(objective, pof)

    expected = objective(X) * torch.sigmoid(X[..., 0]).prod(dim=-1)
    torch.testing.assert_close(acquisition(X), expected)


def test_feasibility_weighted_acquisition_supports_minimum_q_policy() -> None:
    X = torch.tensor([[[0.2], [0.8]]], dtype=torch.double)
    objective = _ObjectiveAcquisition()
    pof = ClassificationProbabilityOfFeasibility(_Classifier())
    acquisition = FeasibilityWeightedAcquisition(
        objective,
        pof,
        q_reduction="minimum",
    )

    expected = objective(X) * torch.sigmoid(X[..., 0]).min(dim=-1).values
    torch.testing.assert_close(acquisition(X), expected)


def test_classifier_feasibility_path_preserves_candidate_autograd() -> None:
    X = torch.tensor([[[0.2], [0.8]]], dtype=torch.double, requires_grad=True)
    acquisition = FeasibilityWeightedAcquisition(
        _ObjectiveAcquisition(),
        ClassificationProbabilityOfFeasibility(_Classifier()),
    )

    acquisition(X).sum().backward()

    assert X.grad is not None
    assert torch.isfinite(X.grad).all()
