"""Masked predictive information gain contracts."""

import torch
from botorch.models import ModelListGP, SingleTaskGP

from robotorchan.acquisition import (
    create_acquisition,
    qJointExpectedPredictiveInformationGain,
)


def test_decoupled_epig_matches_independent_joint_components():
    X = torch.linspace(0, 1, 6, dtype=torch.double).unsqueeze(-1)
    model = ModelListGP(SingleTaskGP(X, X.sin()), SingleTaskGP(X, X.cos()))
    model.eval()
    noise = X.new_tensor([0.01, 0.02])
    weights = X.new_tensor([0.25, 0.75])
    mask = torch.tensor([[True, False], [True, True]])
    acq = create_acquisition(
        "qDecoupledExpectedPredictiveInformationGain",
        model=model,
        target_X=X,
        X_evaluation_mask=mask,
        observation_noise=noise,
        output_weights=weights,
    )
    candidate = X.new_tensor([[[0.3], [0.7]], [[0.4], [0.8]]]).requires_grad_()
    value = acq(candidate)
    expected = sum(
        weights[j]
        * qJointExpectedPredictiveInformationGain(child, X, observation_noise=noise[j])(
            candidate[..., mask[:, j], :]
        )
        for j, child in enumerate(model.models)
    )
    torch.testing.assert_close(value, expected)
    assert value.shape == (2,)
    assert (value >= 0).all()
    (gradient,) = torch.autograd.grad(value.sum(), candidate)
    assert torch.isfinite(gradient).all()


def test_pending_epig_conditional_information():
    X = torch.linspace(0, 1, 6, dtype=torch.double).unsqueeze(-1)
    model = ModelListGP(SingleTaskGP(X, X.sin()), SingleTaskGP(X, X.cos()))
    model.eval()
    mask = torch.tensor([[True, False]])
    acq = create_acquisition(
        "qDecoupledExpectedPredictiveInformationGain",
        model=model,
        target_X=X,
        X_evaluation_mask=mask,
        observation_noise=X.new_tensor([0.1, 0.1]),
    )
    pending = X.new_tensor([[0.3]])
    candidate = X.new_tensor([[[0.4]], [[0.8]]]).requires_grad_()
    acq.set_X_pending(pending, X_pending_evaluation_mask=mask)
    expanded = pending.expand(2, 1, 1)
    component = acq.components[0]
    expected = 0.5 * (component(torch.cat((expanded, candidate), dim=-2)) - component(expanded))
    actual = acq(candidate)
    torch.testing.assert_close(actual, expected)
    assert (actual >= -1e-8).all()
    actual.sum().backward()
    assert torch.isfinite(candidate.grad).all()
    acq.set_X_pending(None)
    torch.testing.assert_close(acq(candidate), 0.5 * component(candidate))
