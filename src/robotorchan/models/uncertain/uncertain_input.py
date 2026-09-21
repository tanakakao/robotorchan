"""Exact GP with analytically integrated Gaussian training-input uncertainty."""

from __future__ import annotations

import torch
from botorch.models.kernels.categorical import CategoricalKernel
from gpytorch.kernels import Kernel
from torch import Tensor

from robotorchan.models.base import continuous_feature_dims, normalize_feature_dims
from robotorchan.models.standard.single_task import SingleTaskGP


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

        lengthscale2 = self.lengthscale.square().reshape(*((1,) * (pair_covar.ndim - 2)), d)
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


def _validate_input_uncertainty(
    train_X: Tensor,
    *,
    train_X_std: Tensor | None,
    train_X_covar: Tensor | None,
) -> Tensor:
    """Validate Gaussian input uncertainty and return covariance matrices."""
    if (train_X_std is None) == (train_X_covar is None):
        raise ValueError("Exactly one of train_X_std and train_X_covar must be supplied.")
    d = train_X.shape[-1]
    if train_X_std is not None:
        if train_X_std.shape != train_X.shape:
            raise ValueError("train_X_std must have the same shape as train_X.")
        if not torch.isfinite(train_X_std).all() or torch.any(train_X_std < 0):
            raise ValueError("train_X_std must be finite and nonnegative.")
        return torch.diag_embed(train_X_std.square())
    expected_shape = (*train_X.shape[:-1], d, d)
    if train_X_covar is None or train_X_covar.shape != expected_shape:
        raise ValueError("train_X_covar must have shape train_X.shape[:-1] + (d, d).")
    if not torch.isfinite(train_X_covar).all():
        raise ValueError("train_X_covar must be finite.")
    if not torch.allclose(train_X_covar, train_X_covar.transpose(-1, -2)):
        raise ValueError("train_X_covar must be symmetric.")
    if torch.any(torch.linalg.eigvalsh(train_X_covar) < -1e-10):
        raise ValueError("train_X_covar must be positive semidefinite.")
    return train_X_covar


class MixedGaussianUncertainInputKernel(Kernel):
    """Expected continuous RBF covariance times deterministic categorical covariance."""

    def __init__(self, continuous_dim: int, categorical_dim: int) -> None:
        super().__init__()
        self.continuous_dim = continuous_dim
        self.categorical_dim = categorical_dim
        self.continuous_kernel = GaussianUncertainInputKernel(continuous_dim)
        self.categorical_kernel = CategoricalKernel(ard_num_dims=categorical_dim)

    def forward(self, x1: Tensor, x2: Tensor, diag: bool = False, **params) -> Tensor:
        """Evaluate mixed expected covariance on the private augmented representation."""
        d = self.continuous_dim
        continuous_width = d + d * d
        expected_width = continuous_width + self.categorical_dim
        if x1.shape[-1] != expected_width or x2.shape[-1] != expected_width:
            raise ValueError("Unexpected augmented mixed uncertain-input width.")
        cont1, cat1 = x1[..., :continuous_width], x1[..., continuous_width:]
        cont2, cat2 = x2[..., :continuous_width], x2[..., continuous_width:]
        return self.continuous_kernel(cont1, cont2, diag=diag, **params) * self.categorical_kernel(
            cat1, cat2, diag=diag, **params
        )


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
        input_covar = _validate_input_uncertainty(
            train_X,
            train_X_std=train_X_std,
            train_X_covar=train_X_covar,
        )
        d = train_X.shape[-1]
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


class MixedUncertainInputSingleTaskGP(SingleTaskGP):
    """Exact GP with uncertain continuous training inputs and deterministic categories."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        cat_dims: list[int] | tuple[int, ...],
        train_X_std: Tensor | None = None,
        train_X_covar: Tensor | None = None,
    ) -> None:
        input_dim = train_X.shape[-1]
        categorical = normalize_feature_dims(cat_dims, input_dim, name="cat_dims")
        continuous = continuous_feature_dims(input_dim, cat_dims=categorical)
        if not continuous:
            raise ValueError("Mixed uncertain-input GP requires at least one continuous feature.")
        continuous_X = train_X[..., list(continuous)]
        input_covar = _validate_input_uncertainty(
            continuous_X,
            train_X_std=train_X_std,
            train_X_covar=train_X_covar,
        )
        category_X = train_X[..., list(categorical)]
        augmented_X = torch.cat([continuous_X, input_covar.flatten(-2, -1), category_X], dim=-1)
        kernel = MixedGaussianUncertainInputKernel(len(continuous), len(categorical))
        self._raw_input_dim = input_dim
        self._continuous_dims = continuous
        self.cat_dims = list(categorical)
        super().__init__(augmented_X, train_Y, covar_module=kernel)
        self._store_raw_tensor("train_X", train_X)
        self._store_raw_tensor("train_X_std", train_X_std)
        self._store_raw_tensor("train_X_covar", train_X_covar)

    @property
    def raw_train_X_std(self) -> Tensor | None:
        """Caller-supplied continuous-coordinate standard deviations, if used."""
        return self._get_raw_tensor("train_X_std")

    @property
    def raw_train_X_covar(self) -> Tensor | None:
        """Caller-supplied continuous-coordinate covariance matrices, if used."""
        return self._get_raw_tensor("train_X_covar")

    def _augment_deterministic(self, X: Tensor) -> Tensor:
        if X.shape[-1] != self._raw_input_dim:
            raise ValueError(f"Expected raw inputs with {self._raw_input_dim} features.")
        continuous = X[..., list(self._continuous_dims)]
        d = len(self._continuous_dims)
        zeros = X.new_zeros(*X.shape[:-1], d * d)
        categorical = X[..., self.cat_dims]
        return torch.cat([continuous, zeros, categorical], dim=-1)

    def posterior(
        self,
        X: Tensor,
        output_indices: list[int] | None = None,
        observation_noise: bool | Tensor = False,
        posterior_transform=None,
    ):
        """Return posterior at deterministic raw mixed-space candidates."""
        return super().posterior(
            self._augment_deterministic(X),
            output_indices=output_indices,
            observation_noise=observation_noise,
            posterior_transform=posterior_transform,
        )
