"""Multiclass and ordinal GP classification with infinite-width ReLU NNGP kernels.

The kernel is the same analytic infinite-width ReLU covariance used by the
binary classifiers; likelihood and class-probability APIs are inherited from
the existing multiclass and ordinal variational GP implementations.
"""

from __future__ import annotations

from gpytorch.kernels import ScaleKernel
from torch import Tensor

from robotorchan.models.base import make_mixed_covar_module, normalize_feature_dims
from robotorchan.models.classification.multiclass.standard.single_task import (
    MulticlassSingleTaskGPClassifier,
)
from robotorchan.models.classification.ordinal.standard.single_task import (
    OrdinalSingleTaskGPClassifier,
)
from robotorchan.models.expressive.infinite_width_bnn import InfiniteWidthReLUKernel


def _continuous_kernel(
    train_X: Tensor,
    *,
    depth: int,
    weight_variance: float,
    bias_variance: float,
    ard: bool,
    eps: float,
) -> ScaleKernel:
    return ScaleKernel(
        InfiniteWidthReLUKernel(
            depth=depth,
            weight_variance=weight_variance,
            bias_variance=bias_variance,
            eps=eps,
            ard_num_dims=train_X.shape[-1] if ard else None,
        )
    )


def _mixed_kernel(
    train_X: Tensor,
    *,
    cat_dims: list[int],
    depth: int,
    weight_variance: float,
    bias_variance: float,
    ard: bool,
    eps: float,
):
    cats = normalize_feature_dims(cat_dims, train_X.shape[-1], name="cat_dims")
    if not cats:
        raise ValueError("Mixed infinite-width BNN requires categorical features.")
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

    return make_mixed_covar_module(
        input_dim=train_X.shape[-1],
        cat_dims=cats,
        batch_shape=train_X.shape[:-2],
        cont_kernel_factory=continuous_kernel_factory,
    ), tuple(cats)


class InfiniteWidthBNNMulticlassGPClassifier(MulticlassSingleTaskGPClassifier):
    """Multiclass variational GP using the infinite-width ReLU NNGP kernel."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_classes: int,
        depth: int = 2,
        weight_variance: float = 1.0,
        bias_variance: float = 0.1,
        ard: bool = True,
        eps: float = 1e-7,
    ) -> None:
        kernel = _continuous_kernel(
            train_X,
            depth=depth,
            weight_variance=weight_variance,
            bias_variance=bias_variance,
            ard=ard,
            eps=eps,
        )
        super().__init__(train_X, train_Y, num_classes=num_classes, covar_module=kernel)
        self.depth = depth
        self.weight_variance = weight_variance
        self.bias_variance = bias_variance
        self.ard = ard
        self.eps = eps


class InfiniteWidthBNNOrdinalGPClassifier(OrdinalSingleTaskGPClassifier):
    """Ordinal-probit variational GP using the infinite-width ReLU NNGP kernel."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_classes: int,
        depth: int = 2,
        weight_variance: float = 1.0,
        bias_variance: float = 0.1,
        ard: bool = True,
        eps: float = 1e-7,
    ) -> None:
        kernel = _continuous_kernel(
            train_X,
            depth=depth,
            weight_variance=weight_variance,
            bias_variance=bias_variance,
            ard=ard,
            eps=eps,
        )
        super().__init__(train_X, train_Y, num_classes=num_classes, covar_module=kernel)
        self.depth = depth
        self.weight_variance = weight_variance
        self.bias_variance = bias_variance
        self.ard = ard
        self.eps = eps


class MixedInfiniteWidthBNNMulticlassGPClassifier(MulticlassSingleTaskGPClassifier):
    """Mixed-input multiclass GP with native categorical and NNGP covariance."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_classes: int,
        cat_dims: list[int],
        depth: int = 2,
        weight_variance: float = 1.0,
        bias_variance: float = 0.1,
        ard: bool = True,
        eps: float = 1e-7,
    ) -> None:
        covariance, cats = _mixed_kernel(
            train_X,
            cat_dims=cat_dims,
            depth=depth,
            weight_variance=weight_variance,
            bias_variance=bias_variance,
            ard=ard,
            eps=eps,
        )
        super().__init__(train_X, train_Y, num_classes=num_classes, covar_module=covariance)
        self.cat_dims = cats
        self.depth = depth
        self.weight_variance = weight_variance
        self.bias_variance = bias_variance
        self.ard = ard
        self.eps = eps


class MixedInfiniteWidthBNNOrdinalGPClassifier(OrdinalSingleTaskGPClassifier):
    """Mixed-input ordinal GP with native categorical and NNGP covariance."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_classes: int,
        cat_dims: list[int],
        depth: int = 2,
        weight_variance: float = 1.0,
        bias_variance: float = 0.1,
        ard: bool = True,
        eps: float = 1e-7,
    ) -> None:
        covariance, cats = _mixed_kernel(
            train_X,
            cat_dims=cat_dims,
            depth=depth,
            weight_variance=weight_variance,
            bias_variance=bias_variance,
            ard=ard,
            eps=eps,
        )
        super().__init__(train_X, train_Y, num_classes=num_classes, covar_module=covariance)
        self.cat_dims = cats
        self.depth = depth
        self.weight_variance = weight_variance
        self.bias_variance = bias_variance
        self.ard = ard
        self.eps = eps
