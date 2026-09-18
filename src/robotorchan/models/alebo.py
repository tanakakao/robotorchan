"""Mahalanobis GP components for ALEBO."""

from __future__ import annotations

import torch
from botorch.fit import fit_gpytorch_mll
from gpytorch.kernels import Kernel, ScaleKernel
from gpytorch.module import Module
from torch import Tensor

from robotorchan.models.single_task import SingleTaskGP


class MahalanobisRBFKernel(Kernel):
    """RBF kernel with a learned full positive-definite distance metric."""

    has_lengthscale = False

    def __init__(self, ard_num_dims: int, **kwargs: object) -> None:
        super().__init__(**kwargs)
        if ard_num_dims < 1:
            raise ValueError("ard_num_dims must be positive.")
        self.ard_num_dims = ard_num_dims
        self.register_parameter(
            name="raw_tril",
            parameter=torch.nn.Parameter(torch.eye(ard_num_dims)),
        )

    @property
    def metric_factor(self) -> Tensor:
        """Lower-triangular factor whose Gram matrix is the distance metric."""
        lower = torch.tril(self.raw_tril, diagonal=-1)
        diagonal = torch.diagonal(self.raw_tril, dim1=-2, dim2=-1)
        positive_diagonal = torch.nn.functional.softplus(diagonal) + 1e-6
        return lower + torch.diag_embed(positive_diagonal)

    @property
    def metric(self) -> Tensor:
        """Symmetric positive-definite Mahalanobis metric."""
        factor = self.metric_factor
        return factor @ factor.transpose(-2, -1)

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        **params: object,
    ) -> Tensor:
        """Evaluate exp(-0.5 * Mahalanobis squared distance)."""
        del params
        transformed_x1 = x1 @ self.metric_factor
        transformed_x2 = x2 @ self.metric_factor
        if diag:
            squared_distance = (transformed_x1 - transformed_x2).square().sum(dim=-1)
        else:
            difference = transformed_x1.unsqueeze(-2) - transformed_x2.unsqueeze(-3)
            squared_distance = difference.square().sum(dim=-1)
        return torch.exp(-0.5 * squared_distance)


class ALEBOGP(SingleTaskGP):
    """Single-task GP using ALEBO's full Mahalanobis RBF geometry."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        *,
        covar_module: Module | None = None,
    ) -> None:
        if train_X.ndim < 2:
            raise ValueError("train_X must have at least two dimensions.")
        if covar_module is None:
            covar_module = ScaleKernel(MahalanobisRBFKernel(ard_num_dims=train_X.shape[-1]))
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            covar_module=covar_module,
        )

    @property
    def mahalanobis_kernel(self) -> MahalanobisRBFKernel:
        """Return the ALEBO Mahalanobis base kernel."""
        if not isinstance(self.covar_module, ScaleKernel) or not isinstance(
            self.covar_module.base_kernel, MahalanobisRBFKernel
        ):
            raise TypeError("ALEBOGP requires a ScaleKernel wrapping MahalanobisRBFKernel.")
        return self.covar_module.base_kernel

    @property
    def metric(self) -> Tensor:
        """Return the current learned Mahalanobis metric."""
        return self.mahalanobis_kernel.metric

    def fit(self, **fit_kwargs: object) -> ALEBOGP:
        """Fit ALEBO GP hyperparameters by maximizing the exact marginal likelihood."""
        fit_gpytorch_mll(self.make_mll(), **fit_kwargs)
        return self
