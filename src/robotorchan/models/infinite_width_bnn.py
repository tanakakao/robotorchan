"""Infinite-width neural-network kernel Gaussian process models."""

from __future__ import annotations

import math

import torch
from gpytorch.kernels import Kernel, ScaleKernel
from torch import Tensor

from robotorchan.models.single_task import SingleTaskGP


class InfiniteWidthReLUKernel(Kernel):
    """NNGP kernel induced by an infinite-width fully connected ReLU network.

    The recursion starts from a linear covariance with configurable weight and
    bias variances, then applies the analytic ReLU covariance map for each
    hidden layer.
    """

    has_lengthscale = True

    def __init__(
        self,
        *,
        depth: int = 2,
        weight_variance: float = 1.0,
        bias_variance: float = 0.1,
        eps: float = 1e-7,
        ard_num_dims: int | None = None,
        **kwargs,
    ) -> None:
        if depth <= 0:
            raise ValueError("depth must be positive.")
        if weight_variance <= 0:
            raise ValueError("weight_variance must be positive.")
        if bias_variance < 0:
            raise ValueError("bias_variance must be non-negative.")
        if eps <= 0:
            raise ValueError("eps must be positive.")
        super().__init__(ard_num_dims=ard_num_dims, **kwargs)
        self.depth = int(depth)
        self.weight_variance = float(weight_variance)
        self.bias_variance = float(bias_variance)
        self.eps = float(eps)

    def _scaled(self, X: Tensor) -> Tensor:
        return X / self.lengthscale

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        **params,
    ) -> Tensor:
        del params
        x1 = self._scaled(x1)
        x2 = self._scaled(x2)
        input_dim = x1.shape[-1]

        covariance = (
            self.bias_variance + self.weight_variance * (x1 @ x2.transpose(-1, -2)) / input_dim
        )
        variance1 = self.bias_variance + self.weight_variance * x1.square().mean(dim=-1)
        variance2 = self.bias_variance + self.weight_variance * x2.square().mean(dim=-1)

        for _ in range(self.depth):
            denominator = torch.sqrt(variance1.unsqueeze(-1) * variance2.unsqueeze(-2)).clamp_min(
                self.eps
            )
            cosine = (covariance / denominator).clamp(-1.0, 1.0)
            theta = torch.acos(cosine)
            relu_covariance = (
                denominator * (torch.sin(theta) + (math.pi - theta) * cosine) / (2.0 * math.pi)
            )
            covariance = self.bias_variance + self.weight_variance * relu_covariance
            variance1 = self.bias_variance + self.weight_variance * variance1 / 2.0
            variance2 = self.bias_variance + self.weight_variance * variance2 / 2.0

        if diag:
            return covariance.diagonal(dim1=-2, dim2=-1)
        return covariance


class InfiniteWidthBNNGP(SingleTaskGP):
    """Exact GP using the infinite-width ReLU neural-network kernel."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        *,
        depth: int = 2,
        weight_variance: float = 1.0,
        bias_variance: float = 0.1,
        ard: bool = True,
        eps: float = 1e-7,
    ) -> None:
        ard_num_dims = train_X.shape[-1] if ard else None
        base_kernel = InfiniteWidthReLUKernel(
            depth=depth,
            weight_variance=weight_variance,
            bias_variance=bias_variance,
            eps=eps,
            ard_num_dims=ard_num_dims,
        )
        covar_module = ScaleKernel(base_kernel)
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            covar_module=covar_module,
        )
        self.depth = int(depth)
        self.weight_variance = float(weight_variance)
        self.bias_variance = float(bias_variance)
