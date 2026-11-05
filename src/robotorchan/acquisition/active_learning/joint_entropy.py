"""Joint posterior entropy for Gaussian regression active learning."""

from __future__ import annotations

import math

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.models.model import Model
from botorch.posteriors.ensemble import EnsemblePosterior
from torch import Tensor


class qJointPosteriorEntropy(AcquisitionFunction):
    """Maximize Gaussian joint entropy of a batch, accounting for correlation.

    Unlike a sum of marginal variances, this acquisition uses the full
    covariance of the q candidate outputs. It is not predictive information
    gain about an external target distribution.
    """

    def __init__(self, model: Model, *, jitter: float = 1e-8) -> None:
        if not math.isfinite(jitter) or jitter < 0:
            raise ValueError("jitter must be finite and non-negative.")
        super().__init__(model=model)
        self.jitter = jitter

    def forward(self, X: Tensor) -> Tensor:
        """Return joint Gaussian differential entropy for each q-batch."""
        if X.ndim < 2 or X.shape[-2] < 1:
            raise ValueError("X must have shape (..., q, d) with q >= 1.")
        if self.model.num_outputs != 1:
            raise ValueError("qJointPosteriorEntropy requires a single-output model.")
        if getattr(self.model, "_is_ensemble", False):
            raise ValueError("qJointPosteriorEntropy does not support ensemble posteriors.")
        posterior = self.model.posterior(X)
        if isinstance(posterior, EnsemblePosterior):
            raise ValueError("qJointPosteriorEntropy does not support ensemble posteriors.")
        if not hasattr(posterior, "mvn"):
            raise ValueError("qJointPosteriorEntropy requires a joint Gaussian posterior.")
        covariance = posterior.mvn.covariance_matrix
        q = X.shape[-2]
        if covariance.shape[-2:] != (q, q):
            raise ValueError("posterior covariance must have shape (..., q, q).")
        identity = torch.eye(q, dtype=covariance.dtype, device=covariance.device)
        stabilized = covariance + self.jitter * identity
        sign, logdet = torch.linalg.slogdet(stabilized)
        if not torch.all(sign > 0):
            raise ValueError("posterior covariance must be positive definite.")
        return 0.5 * (q * math.log(2.0 * math.pi * math.e) + logdet)
