"""Spectral-mixture GP classifiers for multiclass and ordinal outcomes."""

from __future__ import annotations

from gpytorch.kernels import SpectralMixtureKernel
from torch import Tensor

from robotorchan.models.base import make_mixed_covar_module, normalize_feature_dims
from robotorchan.models.classification.multiclass.standard.single_task import (
    MulticlassSingleTaskGPClassifier,
)
from robotorchan.models.classification.ordinal.standard.single_task import (
    OrdinalSingleTaskGPClassifier,
)


def _spectral_kernel(train_X: Tensor, num_mixtures: int) -> SpectralMixtureKernel:
    if train_X.ndim != 2 or train_X.shape[0] < 2:
        raise ValueError("train_X must have shape [n, d] with n >= 2.")
    if num_mixtures < 1:
        raise ValueError("num_mixtures must be positive.")
    kernel = SpectralMixtureKernel(
        num_mixtures=num_mixtures,
        ard_num_dims=train_X.shape[-1],
    ).to(train_X)
    kernel.initialize_from_data(train_X, train_X.new_zeros(train_X.shape[0]))
    return kernel


class SpectralMixtureMulticlassGPClassifier(MulticlassSingleTaskGPClassifier):
    """Multiclass variational GP with independent spectral-mixture class kernels."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_classes: int,
        num_mixtures: int = 4,
        inducing_points: Tensor | int | None = None,
        learn_inducing_points: bool = True,
    ) -> None:
        super().__init__(
            train_X,
            train_Y,
            num_classes=num_classes,
            covar_module=_spectral_kernel(train_X, num_mixtures),
            inducing_points=inducing_points,
            learn_inducing_points=learn_inducing_points,
        )


class SpectralMixtureOrdinalGPClassifier(OrdinalSingleTaskGPClassifier):
    """Ordinal-probit variational GP with a spectral-mixture latent kernel."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_classes: int,
        num_mixtures: int = 4,
        inducing_points: Tensor | int | None = None,
        learn_inducing_points: bool = True,
    ) -> None:
        super().__init__(
            train_X,
            train_Y,
            num_classes=num_classes,
            covar_module=_spectral_kernel(train_X, num_mixtures),
            inducing_points=inducing_points,
            learn_inducing_points=learn_inducing_points,
        )


def _mixed_spectral_kernel(
    train_X: Tensor, cat_dims: list[int], num_mixtures: int
):
    if not isinstance(num_mixtures, int) or isinstance(num_mixtures, bool) or num_mixtures < 1:
        raise ValueError("num_mixtures must be a positive integer.")
    categories = normalize_feature_dims(cat_dims, train_X.shape[-1], name="cat_dims")
    continuous = [dim for dim in range(train_X.shape[-1]) if dim not in categories]
    if not continuous:
        raise ValueError("Mixed spectral mixture requires a continuous feature.")

    def spectral_factory(batch_shape, ard_num_dims, active_dims):
        kernel = SpectralMixtureKernel(
            num_mixtures=num_mixtures,
            ard_num_dims=ard_num_dims,
            active_dims=active_dims,
            batch_shape=batch_shape,
        )
        kernel.initialize_from_data(
            train_X[..., active_dims],
            train_X.new_zeros(train_X.shape[0]),
        )
        return kernel

    covariance = make_mixed_covar_module(
        input_dim=train_X.shape[-1],
        cat_dims=categories,
        batch_shape=train_X.shape[:-2],
        cont_kernel_factory=spectral_factory,
    )
    return covariance.to(train_X), tuple(categories)


class MixedSpectralMixtureMulticlassGPClassifier(MulticlassSingleTaskGPClassifier):
    """Multiclass GP with spectral continuous and categorical covariance."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_classes: int,
        cat_dims: list[int],
        num_mixtures: int = 4,
        inducing_points: Tensor | int | None = None,
        learn_inducing_points: bool = True,
    ) -> None:
        kernel, categories = _mixed_spectral_kernel(train_X, cat_dims, num_mixtures)
        super().__init__(
            train_X,
            train_Y,
            num_classes=num_classes,
            covar_module=kernel,
            inducing_points=inducing_points,
            learn_inducing_points=learn_inducing_points,
        )
        self.cat_dims = categories
        self.num_mixtures = num_mixtures


class MixedSpectralMixtureOrdinalGPClassifier(OrdinalSingleTaskGPClassifier):
    """Ordinal GP with spectral continuous and categorical covariance."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_classes: int,
        cat_dims: list[int],
        num_mixtures: int = 4,
        inducing_points: Tensor | int | None = None,
        learn_inducing_points: bool = True,
    ) -> None:
        kernel, categories = _mixed_spectral_kernel(train_X, cat_dims, num_mixtures)
        super().__init__(
            train_X,
            train_Y,
            num_classes=num_classes,
            covar_module=kernel,
            inducing_points=inducing_points,
            learn_inducing_points=learn_inducing_points,
        )
        self.cat_dims = categories
        self.num_mixtures = num_mixtures
