"""Infinite-width neural-network kernel Gaussian process models."""

from __future__ import annotations

import math

import torch
from gpytorch.kernels import Kernel, ScaleKernel
from torch import Tensor

from robotorchan.models.base import make_mixed_covar_module, normalize_feature_dims
from robotorchan.models.standard.multitask import MultiTaskGP
from robotorchan.models.standard.single_task import SingleTaskGP


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


class InfiniteWidthBNNMultiTaskGP(MultiTaskGP):
    """Long-format multi-task GP with an infinite-width ReLU data kernel.

    The task feature is handled by BoTorch's task covariance and is excluded
    from the NNGP data kernel. The public input remains the original
    long-format representation.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        task_feature: int,
        train_Yvar: Tensor | None = None,
        *,
        depth: int = 2,
        weight_variance: float = 1.0,
        bias_variance: float = 0.1,
        ard: bool = True,
        eps: float = 1e-7,
        rank: int | None = None,
    ) -> None:
        input_dim = train_X.shape[-1]
        resolved_task_feature = normalize_feature_dims(
            [task_feature], input_dim, name="task_feature"
        )[0]
        data_dims = [i for i in range(input_dim) if i != resolved_task_feature]
        ard_num_dims = len(data_dims) if ard else None
        base_kernel = InfiniteWidthReLUKernel(
            depth=depth,
            weight_variance=weight_variance,
            bias_variance=bias_variance,
            eps=eps,
            ard_num_dims=ard_num_dims,
            active_dims=data_dims,
        )
        covar_module = ScaleKernel(base_kernel)
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            task_feature=task_feature,
            train_Yvar=train_Yvar,
            covar_module=covar_module,
            rank=rank,
        )
        self.depth = int(depth)
        self.weight_variance = float(weight_variance)
        self.bias_variance = float(bias_variance)


class MixedInfiniteWidthBNNGP(SingleTaskGP):
    """Mixed-input exact GP using an infinite-width ReLU continuous kernel."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: list[int],
        train_Yvar: Tensor | None = None,
        *,
        depth: int = 2,
        weight_variance: float = 1.0,
        bias_variance: float = 0.1,
        ard: bool = True,
        eps: float = 1e-7,
    ) -> None:
        input_dim = train_X.shape[-1]
        normalized_cat_dims = normalize_feature_dims(cat_dims, input_dim, name="cat_dims")

        def continuous_kernel_factory(batch_shape, num_dims, active_dims):
            return ScaleKernel(
                InfiniteWidthReLUKernel(
                    depth=depth,
                    weight_variance=weight_variance,
                    bias_variance=bias_variance,
                    eps=eps,
                    ard_num_dims=num_dims if ard else None,
                    active_dims=active_dims,
                    batch_shape=batch_shape,
                ),
                batch_shape=batch_shape,
            )

        covar_module = make_mixed_covar_module(
            input_dim=input_dim,
            cat_dims=normalized_cat_dims,
            batch_shape=train_X.shape[:-2],
            cont_kernel_factory=continuous_kernel_factory,
        )
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            covar_module=covar_module,
        )
        self.cat_dims = tuple(normalized_cat_dims)
        self.depth = int(depth)
        self.weight_variance = float(weight_variance)
        self.bias_variance = float(bias_variance)


class MixedInfiniteWidthBNNMultiTaskGP(MultiTaskGP):
    """Mixed long-format multi-task GP with an infinite-width ReLU data kernel."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        task_feature: int,
        cat_dims: list[int],
        train_Yvar: Tensor | None = None,
        *,
        depth: int = 2,
        weight_variance: float = 1.0,
        bias_variance: float = 0.1,
        ard: bool = True,
        eps: float = 1e-7,
        rank: int | None = None,
    ) -> None:
        input_dim = train_X.shape[-1]
        task_dim = normalize_feature_dims([task_feature], input_dim, name="task_feature")[0]
        cats = normalize_feature_dims(
            cat_dims,
            input_dim,
            name="cat_dims",
            excluded_dims=[task_dim],
        )

        def continuous_kernel_factory(batch_shape, num_dims, active_dims):
            return ScaleKernel(
                InfiniteWidthReLUKernel(
                    depth=depth,
                    weight_variance=weight_variance,
                    bias_variance=bias_variance,
                    eps=eps,
                    ard_num_dims=num_dims if ard else None,
                    active_dims=active_dims,
                    batch_shape=batch_shape,
                ),
                batch_shape=batch_shape,
            )

        covar_module = make_mixed_covar_module(
            input_dim=input_dim,
            cat_dims=cats,
            excluded_dims=[task_dim],
            batch_shape=train_X.shape[:-2],
            cont_kernel_factory=continuous_kernel_factory,
        )
        covar_module.active_dims = torch.arange(input_dim, device=train_X.device)
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            task_feature=task_feature,
            train_Yvar=train_Yvar,
            covar_module=covar_module,
            rank=rank,
        )
        self.cat_dims = tuple(cats)
        self.depth = int(depth)
        self.weight_variance = float(weight_variance)
        self.bias_variance = float(bias_variance)
