"""Infinite-width ReLU neural-network kernel binary classifiers."""

from __future__ import annotations

from botorch.models.transforms.input import InputTransform
from botorch.models.utils.inducing_point_allocators import InducingPointAllocator
from gpytorch.kernels import ScaleKernel
from gpytorch.means import Mean
from gpytorch.variational import VariationalStrategy, _VariationalDistribution, _VariationalStrategy
from torch import Tensor

from robotorchan.models.base import make_mixed_covar_module, normalize_feature_dims
from robotorchan.models.classification.binary.standard.single_task import (
    BinarySingleTaskGPClassifier,
)
from robotorchan.models.expressive.infinite_width_bnn import InfiniteWidthReLUKernel


class InfiniteWidthBNNBinaryGPClassifier(BinarySingleTaskGPClassifier):
    """Binary variational GP with the existing infinite-width ReLU NNGP kernel."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        depth: int = 2,
        weight_variance: float = 1.0,
        bias_variance: float = 0.1,
        ard: bool = True,
        eps: float = 1e-7,
        learn_inducing_points: bool = True,
        mean_module: Mean | None = None,
        variational_distribution: _VariationalDistribution | None = None,
        variational_strategy: type[_VariationalStrategy] = VariationalStrategy,
        inducing_points: Tensor | int | None = None,
        inducing_point_allocator: InducingPointAllocator | None = None,
        input_transform: InputTransform | None = None,
    ) -> None:
        kernel = InfiniteWidthReLUKernel(
            depth=depth,
            weight_variance=weight_variance,
            bias_variance=bias_variance,
            eps=eps,
            ard_num_dims=train_X.shape[-1] if ard else None,
        )
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            covar_module=ScaleKernel(kernel),
            learn_inducing_points=learn_inducing_points,
            mean_module=mean_module,
            variational_distribution=variational_distribution,
            variational_strategy=variational_strategy,
            inducing_points=inducing_points,
            inducing_point_allocator=inducing_point_allocator,
            input_transform=input_transform,
        )
        self.depth = depth
        self.weight_variance = weight_variance
        self.bias_variance = bias_variance
        self.ard = ard
        self.eps = eps


class MixedInfiniteWidthBNNBinaryGPClassifier(BinarySingleTaskGPClassifier):
    """Mixed binary variational GP with NNGP continuous and categorical kernels."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        cat_dims: list[int],
        depth: int = 2,
        weight_variance: float = 1.0,
        bias_variance: float = 0.1,
        ard: bool = True,
        eps: float = 1e-7,
        learn_inducing_points: bool = True,
        mean_module: Mean | None = None,
        variational_distribution: _VariationalDistribution | None = None,
        variational_strategy: type[_VariationalStrategy] = VariationalStrategy,
        inducing_points: Tensor | int | None = None,
        inducing_point_allocator: InducingPointAllocator | None = None,
        input_transform: InputTransform | None = None,
    ) -> None:
        cats = normalize_feature_dims(cat_dims, train_X.shape[-1], name="cat_dims")
        if len(cats) == train_X.shape[-1]:
            raise ValueError("At least one continuous feature is required.")

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

        covariance = make_mixed_covar_module(
            input_dim=train_X.shape[-1],
            cat_dims=cats,
            batch_shape=train_X.shape[:-2],
            cont_kernel_factory=continuous_kernel_factory,
        )
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            covar_module=covariance,
            learn_inducing_points=learn_inducing_points,
            mean_module=mean_module,
            variational_distribution=variational_distribution,
            variational_strategy=variational_strategy,
            inducing_points=inducing_points,
            inducing_point_allocator=inducing_point_allocator,
            input_transform=input_transform,
        )
        self.cat_dims = tuple(cats)
        self.depth = depth
        self.weight_variance = weight_variance
        self.bias_variance = bias_variance
        self.ard = ard
        self.eps = eps
