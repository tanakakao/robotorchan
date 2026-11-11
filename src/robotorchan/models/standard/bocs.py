"""Quadratic Bayesian linear surrogate for binary combinatorial optimization.

This implements the quadratic feature model used by BOCS with an isotropic
Gaussian coefficient prior and known Gaussian observation noise. Sparse
horseshoe priors and integer/categorical encodings are outside Phase 5.
"""

from __future__ import annotations

import torch
from botorch.models.model import Model
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from torch import Tensor


def bocs_quadratic_features(X: Tensor) -> Tensor:
    """Return intercept, first-order, and unique pairwise binary interactions."""
    if X.ndim < 1 or X.shape[-1] < 1:
        raise ValueError("X must have a positive final dimension.")
    d = X.shape[-1]
    i, j = torch.triu_indices(d, d, offset=1, device=X.device)
    return torch.cat(
        (torch.ones_like(X[..., :1]), X, X[..., i] * X[..., j]), dim=-1
    )


class BOCSSurrogate(Model):
    """Conjugate Bayesian quadratic regression for binary inputs.

    Uses a zero-mean isotropic Gaussian prior over all coefficients, with
    fixed positive prior precision and observation variance. The posterior
    is exact for this prior and supports standard BoTorch acquisition APIs.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        prior_precision: float = 1.0,
        noise_variance: float = 1e-2,
    ) -> None:
        super().__init__()
        if train_X.ndim != 2 or train_X.shape[1] < 1:
            raise ValueError("train_X must have shape [n, d] with d >= 1.")
        if train_Y.shape != (train_X.shape[0], 1):
            raise ValueError("train_Y must have shape [n, 1].")
        if not train_X.is_floating_point() or train_X.dtype != train_Y.dtype:
            raise ValueError("train_X and train_Y must have the same floating dtype.")
        if train_X.device != train_Y.device:
            raise ValueError("train_X and train_Y must share a device.")
        if not bool(torch.all((train_X == 0) | (train_X == 1))):
            raise ValueError("BOCS currently requires binary (0/1) training inputs.")
        if not bool(torch.isfinite(train_Y).all()):
            raise ValueError("train_Y must be finite.")
        if prior_precision <= 0 or noise_variance <= 0:
            raise ValueError("Prior precision and noise variance must be positive.")
        self.register_buffer("train_X", train_X.detach().clone())
        self.register_buffer("train_Y", train_Y.detach().clone())
        self.prior_precision = prior_precision
        self.noise_variance = noise_variance
        self._fitted = False
        self.fit()

    @property
    def num_outputs(self) -> int:
        return 1

    @property
    def batch_shape(self) -> torch.Size:
        return torch.Size()

    def fit(self) -> BOCSSurrogate:
        """Compute the exact Gaussian coefficient posterior in closed form."""
        features = bocs_quadratic_features(self.train_X)
        p = features.shape[-1]
        precision = (
            self.prior_precision * torch.eye(p, dtype=features.dtype, device=features.device)
            + features.transpose(-1, -2) @ features / self.noise_variance
        )
        chol = torch.linalg.cholesky(precision)
        rhs = features.transpose(-1, -2) @ self.train_Y / self.noise_variance
        mean = torch.cholesky_solve(rhs, chol)
        self.register_buffer("_precision_chol", chol)
        self.register_buffer("_weight_mean", mean)
        self._fitted = True
        return self

    def posterior(
        self,
        X: Tensor,
        output_indices: list[int] | None = None,
        observation_noise: bool | Tensor = False,
        posterior_transform=None,
    ) -> GPyTorchPosterior:
        """Return a BoTorch posterior with shape batch x q x 1."""
        if output_indices is not None and output_indices != [0]:
            raise ValueError("BOCS has only output index 0.")
        if X.shape[-1] != self.train_X.shape[-1] or X.ndim < 2:
            raise ValueError("X must have shape [..., q, d].")
        if X.dtype != self.train_X.dtype or X.device != self.train_X.device:
            raise ValueError("X must match training dtype and device.")
        if not bool(torch.all((X == 0) | (X == 1))):
            raise ValueError("BOCS currently requires binary (0/1) query inputs.")
        features = bocs_quadratic_features(X)
        mean = (features @ self._weight_mean).squeeze(-1)
        solved = torch.cholesky_solve(features.transpose(-1, -2), self._precision_chol)
        covariance = features @ solved
        if isinstance(observation_noise, Tensor):
            raise NotImplementedError("Tensor-valued observation noise is unsupported.")
        if observation_noise:
            covariance = covariance + self.noise_variance * torch.eye(
                X.shape[-2], dtype=X.dtype, device=X.device
            )
        posterior = GPyTorchPosterior(MultivariateNormal(mean, covariance))
        if posterior_transform is not None:
            return posterior_transform(posterior)
        return posterior
