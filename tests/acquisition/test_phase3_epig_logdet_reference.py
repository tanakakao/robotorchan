"""Independent Gaussian log-determinant reference for joint and conditional EPIG."""

from types import SimpleNamespace

import torch
from botorch.models.model import Model

from robotorchan.acquisition.active_learning.epig import (
    qConditionalExpectedPredictiveInformationGain,
    qJointExpectedPredictiveInformationGain,
)


class FixedKernelGaussianModel(Model):
    @property
    def num_outputs(self) -> int:
        return 1

    def posterior(self, X: torch.Tensor, **kwargs: object) -> object:
        # A positive-definite covariance independent of the acquisition implementation.
        distances = X.unsqueeze(-2) - X.unsqueeze(-3)
        covariance = torch.exp(-distances.square().sum(dim=-1) / 0.5)
        covariance = covariance + 0.05 * torch.eye(X.shape[-2], dtype=X.dtype, device=X.device)
        return SimpleNamespace(mvn=SimpleNamespace(covariance_matrix=covariance))


def _logdet_information(
    candidate: torch.Tensor, target: torch.Tensor, noise: float
) -> torch.Tensor:
    # I(Y_candidates; Y_target) = (log|Sigma_C| + log|Sigma_T| - log|Sigma_CT|)/2.
    combined = torch.cat((candidate, target), dim=-2)
    distances = combined.unsqueeze(-2) - combined.unsqueeze(-3)
    covariance = torch.exp(-distances.square().sum(dim=-1) / 0.5)
    covariance = covariance + (0.05 + noise) * torch.eye(combined.shape[-2], dtype=combined.dtype)
    q = candidate.shape[-2]
    return 0.5 * (
        torch.linalg.slogdet(covariance[..., :q, :q]).logabsdet
        + torch.linalg.slogdet(covariance[..., q:, q:]).logabsdet
        - torch.linalg.slogdet(covariance).logabsdet
    )


def test_joint_epig_matches_independent_gaussian_logdet_reference() -> None:
    model = FixedKernelGaussianModel()
    target = torch.tensor([[0.25]], dtype=torch.double)
    candidates = torch.tensor([[[0.0], [0.7]]], dtype=torch.double)
    noise = 0.1
    actual = qJointExpectedPredictiveInformationGain(model, target, observation_noise=noise)(
        candidates
    )
    expected = _logdet_information(candidates, target.expand(1, 1, 1), noise)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)


def test_conditional_epig_matches_conditional_mutual_information_logdet() -> None:
    model = FixedKernelGaussianModel()
    target = torch.tensor([[0.25]], dtype=torch.double)
    pending = torch.tensor([[0.1]], dtype=torch.double)
    candidates = torch.tensor([[[0.7], [1.0]]], dtype=torch.double)
    noise = 0.1
    actual = qConditionalExpectedPredictiveInformationGain(
        model, target, pending, observation_noise=noise
    )(candidates)
    expanded_pending = pending.unsqueeze(0)
    expected = _logdet_information(
        torch.cat((expanded_pending, candidates), dim=-2), target.unsqueeze(0), noise
    ) - _logdet_information(expanded_pending, target.unsqueeze(0), noise)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
    assert actual.item() >= -1e-10
