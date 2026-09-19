"""Exact GP with analytically integrated diagonal Gaussian input uncertainty."""

from __future__ import annotations

import torch
from gpytorch.kernels import Kernel
from torch import Tensor

from robotorchan.models.single_task import SingleTaskGP


class GaussianUncertainInputKernel(Kernel):
    """RBF kernel marginalized over independent Gaussian input locations."""

    has_lengthscale = True

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        **params,
    ) -> Tensor:
        """Evaluate the expected RBF covariance on augmented mean/std inputs."""
        d = x1.shape[-1] // 2
        if x1.shape[-1] != 2 * d or x2.shape[-1] != 2 * d:
            raise ValueError("Inputs must concatenate d means and d standard deviations.")
        mean1, std1 = x1[..., :d], x1[..., d:]
        mean2, std2 = x2[..., :d], x2[..., d:]
        lengthscale2 = self.lengthscale.square()
        variance = std1.square().unsqueeze(-2) + std2.square().unsqueeze(-3)
        denominator = lengthscale2 + variance
        delta2 = (mean1.unsqueeze(-2) - mean2.unsqueeze(-3)).square()
        factor = torch.sqrt(lengthscale2 / denominator).prod(dim=-1)
        covariance = factor * torch.exp(-0.5 * (delta2 / denominator).sum(dim=-1))
        if diag:
            return covariance.diagonal(dim1=-2, dim2=-1)
        return covariance


class UncertainInputSingleTaskGP(SingleTaskGP):
    """Single-task GP that integrates diagonal Gaussian training-input uncertainty."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        train_X_std: Tensor,
    ) -> None:
        if train_X_std.shape != train_X.shape:
            raise ValueError("train_X_std must have the same shape as train_X.")
        if torch.any(train_X_std < 0):
            raise ValueError("train_X_std must be nonnegative.")
        self._input_dim = train_X.shape[-1]
        augmented_X = torch.cat([train_X, train_X_std], dim=-1)
        kernel = GaussianUncertainInputKernel()
        kernel.ard_num_dims = None
        super().__init__(augmented_X, train_Y, covar_module=kernel)
        self._store_raw_tensor("train_X", train_X.detach().clone())
        self._store_raw_tensor("train_X_std", train_X_std.detach().clone())

    @property
    def raw_train_X_std(self) -> Tensor:
        """Caller-supplied standard deviations of training input locations."""
        value = self._get_raw_tensor("train_X_std")
        if value is None:
            raise RuntimeError("raw_train_X_std was unexpectedly stored as None.")
        return value

    def posterior(
        self,
        X: Tensor,
        output_indices: list[int] | None = None,
        observation_noise: bool | Tensor = False,
        posterior_transform=None,
    ):
        """Return posterior at deterministic candidate inputs.

        Candidate locations are treated as exact here. Candidate-time input
        uncertainty remains the responsibility of the scenario/risk layer.
        """
        zeros = torch.zeros_like(X)
        augmented_X = torch.cat([X, zeros], dim=-1)
        return super().posterior(
            augmented_X,
            output_indices=output_indices,
            observation_noise=observation_noise,
            posterior_transform=posterior_transform,
        )
