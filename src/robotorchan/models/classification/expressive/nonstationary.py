"""Input-dependent Gibbs-kernel multiclass and ordinal GP classifiers."""

from __future__ import annotations

import torch
from gpytorch.kernels import ScaleKernel
from torch import Tensor

from robotorchan.models.base import (
    continuous_feature_dims,
    make_mixed_covar_module,
    normalize_feature_dims,
)
from robotorchan.models.classification.multiclass.standard.single_task import (
    MulticlassSingleTaskGPClassifier,
)
from robotorchan.models.classification.ordinal.standard.single_task import (
    OrdinalSingleTaskGPClassifier,
)
from robotorchan.models.robust.nonstationary import GibbsKernel


def _gibbs_covariance(train_X: Tensor, lengthscale_floor: float) -> ScaleKernel:
    if train_X.ndim != 2 or train_X.shape[0] < 2:
        raise ValueError("train_X must have shape [n, d] with n >= 2.")
    return ScaleKernel(GibbsKernel(train_X.shape[-1], lengthscale_floor=lengthscale_floor)).to(
        train_X
    )


def _mixed_gibbs_covariance(train_X: Tensor, cat_dims: list[int], lengthscale_floor: float):
    normalized = normalize_feature_dims(cat_dims, train_X.shape[-1], name="cat_dims")
    continuous = continuous_feature_dims(train_X.shape[-1], cat_dims=normalized)
    if not continuous:
        raise ValueError("Mixed nonstationary GP requires a continuous dimension.")

    def continuous_factory(batch_shape, ard_num_dims, active_dims):
        del batch_shape, ard_num_dims
        return ScaleKernel(
            GibbsKernel(len(active_dims), lengthscale_floor=lengthscale_floor),
            active_dims=active_dims,
        )

    kernel = make_mixed_covar_module(
        input_dim=train_X.shape[-1],
        cat_dims=normalized,
        batch_shape=train_X.shape[:-2],
        cont_kernel_factory=continuous_factory,
    ).to(train_X)
    return kernel, tuple(normalized), tuple(continuous)


class NonstationaryMulticlassGPClassifier(MulticlassSingleTaskGPClassifier):
    """Independent class GPs with learned input-dependent Gibbs lengthscales."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_classes: int,
        lengthscale_floor: float = 1e-3,
        inducing_points: Tensor | int | None = None,
        learn_inducing_points: bool = True,
    ) -> None:
        super().__init__(
            train_X,
            train_Y,
            num_classes=num_classes,
            covar_module=_gibbs_covariance(train_X, lengthscale_floor),
            inducing_points=inducing_points,
            learn_inducing_points=learn_inducing_points,
        )

    def local_lengthscale(self, X: Tensor) -> Tensor:
        """Return local lengthscales for each class GP, stacked on axis zero."""
        return torch.stack(
            [gp.model.covar_module.base_kernel.local_lengthscale(X) for gp in self.class_models],
            dim=0,
        )


class NonstationaryOrdinalGPClassifier(OrdinalSingleTaskGPClassifier):
    """Ordered-probit GP with learned input-dependent Gibbs lengthscales."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_classes: int,
        lengthscale_floor: float = 1e-3,
        inducing_points: Tensor | int | None = None,
        learn_inducing_points: bool = True,
    ) -> None:
        super().__init__(
            train_X,
            train_Y,
            num_classes=num_classes,
            covar_module=_gibbs_covariance(train_X, lengthscale_floor),
            inducing_points=inducing_points,
            learn_inducing_points=learn_inducing_points,
        )

    def local_lengthscale(self, X: Tensor) -> Tensor:
        """Return the ordinal latent GP's local lengthscales."""
        return self.model.covar_module.base_kernel.local_lengthscale(X)


class MixedNonstationaryMulticlassGPClassifier(MulticlassSingleTaskGPClassifier):
    """Mixed-input multiclass GP with Gibbs covariance on continuous features."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_classes: int,
        cat_dims: list[int],
        lengthscale_floor: float = 1e-3,
        inducing_points: Tensor | int | None = None,
        learn_inducing_points: bool = True,
    ) -> None:
        kernel, categories, continuous = _mixed_gibbs_covariance(
            train_X, cat_dims, lengthscale_floor
        )
        super().__init__(
            train_X,
            train_Y,
            num_classes=num_classes,
            covar_module=kernel,
            inducing_points=inducing_points,
            learn_inducing_points=learn_inducing_points,
        )
        self.cat_dims = categories
        self.continuous_dims = continuous


class MixedNonstationaryOrdinalGPClassifier(OrdinalSingleTaskGPClassifier):
    """Mixed-input ordinal GP with Gibbs covariance on continuous features."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_classes: int,
        cat_dims: list[int],
        lengthscale_floor: float = 1e-3,
        inducing_points: Tensor | int | None = None,
        learn_inducing_points: bool = True,
    ) -> None:
        kernel, categories, continuous = _mixed_gibbs_covariance(
            train_X, cat_dims, lengthscale_floor
        )
        super().__init__(
            train_X,
            train_Y,
            num_classes=num_classes,
            covar_module=kernel,
            inducing_points=inducing_points,
            learn_inducing_points=learn_inducing_points,
        )
        self.cat_dims = categories
        self.continuous_dims = continuous
