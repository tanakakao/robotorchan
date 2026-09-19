"""Exact GP with analytically integrated Gaussian training-input uncertainty."""

from __future__ import annotations

import torch
from gpytorch.kernels import Kernel
from torch import Tensor

from robotorchan.models.single_task import SingleTaskGP


class GaussianUncertainInputKernel(Kernel):
    """RBF kernel marginalized over Gaussian input locations."""

    has_lengthscale = True

    def __init__(self, input_dim: int) -> None:
        super().__init__(ard_num_dims=None)
        self.input_dim = input_dim
        self.raw_lengthscale = torch.nn.Parameter(torch.zeros(1, 1, input_dim))
        self.register_constraint("raw_lengthscale", self.raw_lengthscale_constraint)

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        **params,
    ) -> Tensor:
        """Evaluate expected RBF covariance on augmented mean/covariance inputs."""
        d = self.input_dim
        width = d + d * d
        if x1.shape[-1] != width or x2.shape[-1] != width:
            raise ValueError("Inputs must concatenate means and flattened covariance matrices.")

        mean1 = x1[..., :d]
        mean2 = x2[..., :d]
        covar1 = x1[..., d:].reshape(*x1.shape[:-1], d, d)
        covar2 = x2[..., d:].reshape(*x2.shape[:-1], d, d)
        pair_covar = covar1.unsqueeze(-3) + covar2.unsqueeze(-4)

        lengthscale2 = self.lengthscale.square().reshape(
            *((1,) * (pair_covar.ndim - 2)), d
        )
        metric = torch.diag_embed(lengthscale2)
        system = metric + pair_covar
        chol = torch.linalg.cholesky(system)

        delta = mean1.unsqueeze(-2) - mean2.unsqueeze(-3)
        solved = torch.cholesky_solve(delta.unsqueeze(-1), chol).squeeze(-1)
        exponent = -0.5 * (delta * solved).sum(dim=-1)

        logdet_metric = lengthscale2.log().sum(dim=-1)
        logdet_system = 2 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(dim=-1)
        factor = torch.exp(0.5 * (logdet_metric - logdet_system))
        covariance = factor * torch.exp(exponent)
        if diag:
            return covariance.diagonal(dim1=-2, dim2=-1)
        return covariance


class UncertainInputSingleTaskGP(SingleTaskGP):
    """Single-task GP integrating Gaussian uncertainty in observed input locations."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        train_X_std: Tensor | None = None,
        train_X_covar: Tensor | None = None,
    ) -> None:
        if (train_X_std is None) == (train_X_covar is None):
            raise ValueError("Exactly one of train_X_std and train_X_covar must be supplied.")

        d = train_X.shape[-1]
        if train_X_std is not None:
            if train_X_std.shape != train_X.shape:
                raise ValueError("train_X_std must have the same shape as train_X.")
            if not torch.isfinite(train_X_std).all() or torch.any(train_X_std < 0):
                raise ValueError("train_X_std must be finite and nonnegative.")
            input_covar = torch.diag_embed(train_X_std.square())
        else:
            expected_shape = (*train_X.shape[:-1], d, d)
            if train_X_covar is None or train_X_covar.shape != expected_shape:
                raise ValueError("train_X_covar must have shape train_X.shape[:-1] + (d, d).")
            if not torch.isfinite(train_X_covar).all():
                raise ValueError("train_X_covar must be finite.")
            if not torch.allclose(train_X_covar, train_X_covar.transpose(-1, -2)):
                raise ValueError("train_X_covar must be symmetric.")
            eigenvalues = torch.linalg.eigvalsh(train_X_covar)
            if torch.any(eigenvalues < -1e-10):
                raise ValueError("train_X_covar must be positive semidefinite.")
            input_covar = train_X_covar

        self._input_dim = d
        augmented_X = torch.cat([train_X, input_covar.flatten(-2, -1)], dim=-1)
        kernel = GaussianUncertainInputKernel(input_dim=d)
        super().__init__(augmented_X, train_Y, covar_module=kernel)
        self._store_raw_tensor("train_X", train_X.detach().clone())
        self._store_raw_tensor(
            "train_X_std", None if train_X_std is None else train_X_std.detach().clone()
        )
        self._store_raw_tensor(
            "train_X_covar", None if train_X_covar is None else train_X_covar.detach().clone()
        )

    @property
    def raw_train_X_std(self) -> Tensor | None:
        """Caller-supplied diagonal input standard deviations, if used."""
        return self._get_raw_tensor("train_X_std")

    @property
    def raw_train_X_covar(self) -> Tensor | None:
        """Caller-supplied full input covariance matrices, if used."""
        return self._get_raw_tensor("train_X_covar")

    def posterior(
        self,
        X: Tensor,
        output_indices: list[int] | None = None,
        observation_noise: bool | Tensor = False,
        posterior_transform=None,
    ):
        """Return posterior at deterministic candidate inputs."""
        d = self._input_dim
        zeros = X.new_zeros(*X.shape[:-1], d * d)
        augmented_X = torch.cat([X, zeros], dim=-1)
        return super().posterior(
            augmented_X,
            output_indices=output_indices,
            observation_noise=observation_noise,
            posterior_transform=posterior_transform,
        )
