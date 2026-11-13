"""Mixed-input spectral-mixture binary variational GP classifier."""

from __future__ import annotations

from botorch.models.transforms.input import InputTransform
from botorch.models.utils.inducing_point_allocators import InducingPointAllocator
from gpytorch.kernels import SpectralMixtureKernel
from gpytorch.means import Mean
from gpytorch.variational import VariationalStrategy, _VariationalDistribution, _VariationalStrategy
from torch import Tensor

from robotorchan.models.base import make_mixed_covar_module, normalize_feature_dims
from robotorchan.models.classification.binary.standard.single_task import (
    BinarySingleTaskGPClassifier,
)


class MixedSpectralMixtureBinaryGPClassifier(BinarySingleTaskGPClassifier):
    """Binary classifier combining spectral continuous and categorical covariance."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        cat_dims: list[int],
        num_mixtures: int = 4,
        learn_inducing_points: bool = True,
        mean_module: Mean | None = None,
        variational_distribution: _VariationalDistribution | None = None,
        variational_strategy: type[_VariationalStrategy] = VariationalStrategy,
        inducing_points: Tensor | int | None = None,
        inducing_point_allocator: InducingPointAllocator | None = None,
        input_transform: InputTransform | None = None,
    ) -> None:
        if not isinstance(num_mixtures, int) or isinstance(num_mixtures, bool) or num_mixtures < 1:
            raise ValueError("num_mixtures must be a positive integer.")
        categorical = normalize_feature_dims(cat_dims, train_X.shape[-1], name="cat_dims")
        continuous = [dim for dim in range(train_X.shape[-1]) if dim not in categorical]
        if not continuous:
            raise ValueError("Mixed spectral mixture requires a continuous feature.")

        def spectral_kernel(batch_shape, ard_num_dims, active_dims):
            kernel = SpectralMixtureKernel(
                num_mixtures=num_mixtures,
                ard_num_dims=ard_num_dims,
                active_dims=active_dims,
                batch_shape=batch_shape,
            )
            kernel.initialize_from_data(
                train_X[..., active_dims],
                train_Y.to(dtype=train_X.dtype),
            )
            return kernel

        covariance = make_mixed_covar_module(
            input_dim=train_X.shape[-1],
            cat_dims=categorical,
            batch_shape=train_X.shape[:-2],
            cont_kernel_factory=spectral_kernel,
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
        self.cat_dims = categorical
        self.num_mixtures = num_mixtures
