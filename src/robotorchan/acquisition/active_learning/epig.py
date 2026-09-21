"""Expected Predictive Information Gain for Gaussian regression."""

from __future__ import annotations

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.models.model import Model
from torch import Tensor


class ExpectedPredictiveInformationGain(AcquisitionFunction):
    """EPIG between one candidate and a target input distribution.

    This implementation uses the mutual information of jointly Gaussian noisy
    observations. The target distribution is represented by a finite set of
    target points and optional non-negative integration weights.
    """

    def __init__(
        self,
        model: Model,
        target_X: Tensor,
        *,
        observation_noise: float | Tensor = 0.0,
        target_weights: Tensor | None = None,
    ) -> None:
        super().__init__(model=model)
        if target_X.ndim < 2:
            raise ValueError("target_X must have shape (..., n_target, d).")
        self.register_buffer("target_X", target_X)
        self.register_buffer("observation_noise", torch.as_tensor(observation_noise))
        if target_weights is not None:
            if target_weights.ndim != 1 or target_weights.shape[0] != target_X.shape[-2]:
                raise ValueError("target_weights must have shape (n_target,).")
            if (target_weights < 0).any() or target_weights.sum() <= 0:
                raise ValueError("target_weights must be non-negative with positive sum.")
        self.register_buffer("target_weights", target_weights)

    def forward(self, X: Tensor) -> Tensor:
        """Evaluate Gaussian-regression EPIG for q=1 candidates."""
        if X.shape[-2] != 1:
            raise ValueError("ExpectedPredictiveInformationGain supports q=1.")
        if self.model.num_outputs != 1:
            raise ValueError("ExpectedPredictiveInformationGain requires a single-output model.")

        target_X = self.target_X.to(dtype=X.dtype, device=X.device)
        n_target = target_X.shape[-2]
        candidate = X.squeeze(-2)
        expanded_candidate = candidate.unsqueeze(-2).expand(*candidate.shape[:-1], n_target, -1)
        expanded_target = target_X.expand(*candidate.shape[:-1], n_target, target_X.shape[-1])
        pair_X = torch.stack((expanded_candidate, expanded_target), dim=-2)
        posterior = self.model.posterior(pair_X)
        covariance = posterior.mvn.covariance_matrix

        noise = self.observation_noise.to(dtype=X.dtype, device=X.device).clamp_min(0.0)
        var_candidate = covariance[..., 0, 0] + noise
        var_target = covariance[..., 1, 1] + noise
        covariance_ct = covariance[..., 0, 1]
        rho_sq = covariance_ct.square() / (var_candidate * var_target).clamp_min(
            torch.finfo(X.dtype).tiny
        )
        rho_sq = rho_sq.clamp(min=0.0, max=1.0 - torch.finfo(X.dtype).eps)
        information = -0.5 * torch.log1p(-rho_sq)

        if self.target_weights is None:
            return information.mean(dim=-1)
        weights = self.target_weights.to(dtype=X.dtype, device=X.device)
        weights = weights / weights.sum()
        return (information * weights).sum(dim=-1)
