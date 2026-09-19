"""Exact GP with an input-dependent Gibbs covariance kernel."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from gpytorch.kernels import Kernel, ScaleKernel
from torch import Tensor, nn

from robotorchan.models.single_task import SingleTaskGP


class GibbsKernel(Kernel):
    """Gibbs covariance with affine input-dependent positive lengthscales."""

    has_lengthscale = False

    def __init__(self, input_dim: int, *, lengthscale_floor: float = 1e-3) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be positive.")
        if lengthscale_floor <= 0:
            raise ValueError("lengthscale_floor must be positive.")

        self.input_dim = int(input_dim)
        self.lengthscale_floor = float(lengthscale_floor)
        self.lengthscale_intercept = nn.Parameter(torch.zeros(input_dim))
        self.lengthscale_slope = nn.Parameter(torch.zeros(input_dim, input_dim))

    def local_lengthscale(self, X: Tensor) -> Tensor:
        """Return positive local lengthscales for each input location."""
        if X.shape[-1] != self.input_dim:
            raise ValueError("X has an incompatible feature dimension.")
        linear = self.lengthscale_intercept + X @ self.lengthscale_slope.transpose(-1, -2)
        return F.softplus(linear) + self.lengthscale_floor

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        **params,
    ) -> Tensor:
        """Evaluate the positive-semidefinite Gibbs covariance."""
        ell1 = self.local_lengthscale(x1)
        ell2 = self.local_lengthscale(x2)

        if diag:
            if x1.shape != x2.shape:
                raise ValueError("diag=True requires x1 and x2 to have the same shape.")
            ell1_sq = ell1.square()
            ell2_sq = ell2.square()
            denominator = ell1_sq + ell2_sq
            prefactor = torch.sqrt(2 * ell1 * ell2 / denominator).prod(dim=-1)
            exponent = -((x1 - x2).square() / denominator).sum(dim=-1)
            return prefactor * torch.exp(exponent)

        ell1_sq = ell1.square().unsqueeze(-2)
        ell2_sq = ell2.square().unsqueeze(-3)
        denominator = ell1_sq + ell2_sq
        prefactor = torch.sqrt(
            2 * ell1.unsqueeze(-2) * ell2.unsqueeze(-3) / denominator
        ).prod(dim=-1)
        delta = x1.unsqueeze(-2) - x2.unsqueeze(-3)
        exponent = -(delta.square() / denominator).sum(dim=-1)
        return prefactor * torch.exp(exponent)


class NonstationarySingleTaskGP(SingleTaskGP):
    """Single-task exact GP with input-dependent latent-process smoothness."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        *,
        lengthscale_floor: float = 1e-3,
    ) -> None:
        input_dim = train_X.shape[-1]
        gibbs_kernel = GibbsKernel(input_dim, lengthscale_floor=lengthscale_floor)
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            covar_module=ScaleKernel(gibbs_kernel),
        )

    @property
    def gibbs_kernel(self) -> GibbsKernel:
        """Underlying nonstationary covariance kernel."""
        kernel = self.covar_module.base_kernel
        if not isinstance(kernel, GibbsKernel):
            raise RuntimeError("Expected GibbsKernel as the base covariance module.")
        return kernel

    def local_lengthscale(self, X: Tensor) -> Tensor:
        """Return learned input-dependent lengthscales."""
        return self.gibbs_kernel.local_lengthscale(X)
