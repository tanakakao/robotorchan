"""Tests for classification reliability and OOD diagnostics."""

import torch
from torch import Tensor, nn

from robotorchan.models.classification import (
    ClassificationReliabilityEvaluator,
    max_probability_reliability,
)


class _ReliabilityModel(nn.Module):
    num_classes = 2

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "raw_train_X",
            torch.tensor([[-1.0], [-0.5], [0.5], [1.0]], dtype=torch.double),
        )

    def predict_proba(self, X: Tensor, **kwargs: object) -> Tensor:
        del kwargs
        positive = torch.sigmoid(X[..., 0])
        return torch.stack((1.0 - positive, positive), dim=-1)

    def mutual_information(
        self,
        X: Tensor,
        *,
        num_samples: int = 128,
        **kwargs: object,
    ) -> Tensor:
        del num_samples, kwargs
        return X[..., 0].abs() * 0.1


def test_max_probability_reliability_is_multiclass_ready() -> None:
    probabilities = torch.tensor([[0.1, 0.2, 0.7], [0.4, 0.3, 0.3]])
    score = max_probability_reliability(probabilities)
    torch.testing.assert_close(score, torch.tensor([0.7, 0.4]))


def test_predictive_entropy_uses_probability_contract() -> None:
    evaluator = ClassificationReliabilityEvaluator(_ReliabilityModel())
    X = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    score = evaluator.predictive_entropy(X)

    assert score.shape == torch.Size([2])
    assert score[0] > score[1]


def test_epistemic_disagreement_delegates_to_model_contract() -> None:
    evaluator = ClassificationReliabilityEvaluator(_ReliabilityModel())
    X = torch.tensor([[0.0], [2.0]], dtype=torch.double)

    torch.testing.assert_close(
        evaluator.epistemic_disagreement(X),
        torch.tensor([0.0, 0.2], dtype=torch.double),
    )


def test_input_ood_score_increases_far_from_reference_data() -> None:
    evaluator = ClassificationReliabilityEvaluator(_ReliabilityModel())
    near = evaluator.input_ood_score(torch.tensor([[0.0]], dtype=torch.double))
    far = evaluator.input_ood_score(torch.tensor([[5.0]], dtype=torch.double))

    assert far.item() > near.item()


def test_input_ood_score_preserves_candidate_gradients() -> None:
    evaluator = ClassificationReliabilityEvaluator(_ReliabilityModel())
    X = torch.tensor([[2.0]], dtype=torch.double, requires_grad=True)
    score = evaluator.input_ood_score(X).sum()

    score.backward()

    assert X.grad is not None
    assert torch.isfinite(X.grad).all()
