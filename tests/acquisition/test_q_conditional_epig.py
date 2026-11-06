"""Incremental qEPIG conditions on previously selected pending candidates."""

from unittest.mock import patch

import torch
from botorch.models import SingleTaskGP

from robotorchan.acquisition.active_learning.epig import (
    qConditionalExpectedPredictiveInformationGain,
    qJointExpectedPredictiveInformationGain,
)


def test_conditional_epig_subtracts_pending_information() -> None:
    train_X = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    model = SingleTaskGP(train_X, train_X.square())
    pending = torch.tensor([[0.2]], dtype=torch.double)
    target = torch.tensor([[0.8]], dtype=torch.double)
    X = torch.tensor([[[0.4], [0.6]]], dtype=torch.double)
    acquisition = qConditionalExpectedPredictiveInformationGain(
        model, target, pending, observation_noise=1e-3
    )
    with patch.object(
        qJointExpectedPredictiveInformationGain,
        "forward",
        side_effect=lambda points: (
            points.shape[-2] * torch.ones(points.shape[:-2], dtype=points.dtype)
        ),
    ) as forward:
        result = acquisition(X)
    assert forward.call_count == 2
    assert forward.call_args_list[0].args[0].shape == (1, 3, 1)
    assert forward.call_args_list[1].args[0].shape == (1, 1, 1)
    torch.testing.assert_close(result, torch.tensor([2.0], dtype=torch.double))


def test_conditional_epig_pending_updates_and_clear() -> None:
    train_X = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    model = SingleTaskGP(train_X, train_X.square())
    model.eval()
    target = torch.tensor([[0.8]], dtype=torch.double)
    acquisition = qConditionalExpectedPredictiveInformationGain(
        model, target, torch.tensor([[0.2]], dtype=torch.double), observation_noise=1e-3
    )
    X = torch.tensor([[[0.4]]], dtype=torch.double)
    acquisition.set_X_pending(torch.tensor([[0.6]], dtype=torch.double))
    torch.testing.assert_close(acquisition.X_pending, torch.tensor([[0.6]], dtype=torch.double))
    assert torch.isfinite(acquisition(X)).all()
    acquisition.set_X_pending(None)
    assert acquisition.X_pending is None
    expected = qJointExpectedPredictiveInformationGain(model, target, observation_noise=1e-3)(X)
    torch.testing.assert_close(acquisition(X), expected)


def test_conditional_epig_starts_without_pending_candidates() -> None:
    train_X = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    model = SingleTaskGP(train_X, train_X.square())
    model.eval()
    target = torch.tensor([[0.8]], dtype=torch.double)
    X = torch.tensor([[[0.4]]], dtype=torch.double)
    acquisition = qConditionalExpectedPredictiveInformationGain(model, target, observation_noise=1e-3)
    assert acquisition.X_pending is None
    baseline = qJointExpectedPredictiveInformationGain(model, target, observation_noise=1e-3)
    torch.testing.assert_close(acquisition(X), baseline(X))
    acquisition.set_X_pending(torch.tensor([[0.2]], dtype=torch.double))
    assert torch.isfinite(acquisition(X)).all()
