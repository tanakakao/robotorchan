"""Tests for split-conformal classification prediction sets."""

import torch
from torch import Tensor, nn

from robotorchan.models.classification import (
    SplitConformalClassifier,
    classification_nonconformity_scores,
    conformal_quantile,
)


class _ProbabilityModel(nn.Module):
    def predict_proba(self, X: Tensor, **kwargs: object) -> Tensor:
        del kwargs
        logits = torch.cat((X, -X, torch.zeros_like(X)), dim=-1)
        return logits.softmax(dim=-1)


def test_nonconformity_scores_use_observed_class_probability() -> None:
    probabilities = torch.tensor(
        [[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]],
        dtype=torch.double,
    )
    targets = torch.tensor([0, 2], dtype=torch.long)

    torch.testing.assert_close(
        classification_nonconformity_scores(probabilities, targets),
        torch.tensor([0.3, 0.4], dtype=torch.double),
    )


def test_conformal_quantile_uses_finite_sample_correction() -> None:
    scores = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.double)
    quantile = conformal_quantile(scores, alpha=0.2)

    torch.testing.assert_close(quantile, torch.tensor(0.4, dtype=torch.double))


def test_split_conformal_returns_multiclass_prediction_sets() -> None:
    wrapper = SplitConformalClassifier(_ProbabilityModel(), alpha=0.2)
    calibration_X = torch.tensor([[2.0], [-2.0], [1.5], [-1.5]], dtype=torch.double)
    calibration_Y = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    wrapper.calibrate(calibration_X, calibration_Y)

    X = torch.tensor([[2.0], [0.0]], dtype=torch.double)
    prediction_set = wrapper.prediction_set(X)

    assert prediction_set.dtype == torch.bool
    assert prediction_set.shape == torch.Size([2, 3])
    torch.testing.assert_close(
        wrapper.prediction_set_size(X),
        prediction_set.sum(dim=-1),
    )


def test_conformal_wrapper_does_not_modify_predictive_probabilities() -> None:
    model = _ProbabilityModel()
    wrapper = SplitConformalClassifier(model)
    X = torch.tensor([[0.5]], dtype=torch.double)

    torch.testing.assert_close(wrapper.predict_proba(X), model.predict_proba(X))


def test_prediction_set_requires_held_out_calibration() -> None:
    wrapper = SplitConformalClassifier(_ProbabilityModel())

    try:
        wrapper.prediction_set(torch.tensor([[0.0]], dtype=torch.double))
    except RuntimeError as error:
        assert "calibrate" in str(error)
    else:
        raise AssertionError("prediction_set must require conformal calibration.")
