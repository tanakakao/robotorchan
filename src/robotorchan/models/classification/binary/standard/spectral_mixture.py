"""Spectral-mixture binary variational Gaussian process classifier."""

from __future__ import annotations

from botorch.models.transforms.input import InputTransform
from botorch.models.utils.inducing_point_allocators import InducingPointAllocator
from gpytorch.kernels import ScaleKernel, SpectralMixtureKernel
from gpytorch.means import Mean
from gpytorch.variational import VariationalStrategy, _VariationalDistribution, _VariationalStrategy
from torch import Tensor

from robotorchan.models.classification.binary.standard.single_task import (
    BinarySingleTaskGPClassifier,
)


class SpectralMixtureBinaryGPClassifier(BinarySingleTaskGPClassifier):
    """Binary classifier using GPyTorch's spectral mixture covariance."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
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
        kernel = SpectralMixtureKernel(
            num_mixtures=num_mixtures,
            ard_num_dims=train_X.shape[-1],
        )
        kernel.initialize_from_data(train_X, train_Y.to(dtype=train_X.dtype))
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            learn_inducing_points=learn_inducing_points,
            covar_module=ScaleKernel(kernel),
            mean_module=mean_module,
            variational_distribution=variational_distribution,
            variational_strategy=variational_strategy,
            inducing_points=inducing_points,
            inducing_point_allocator=inducing_point_allocator,
            input_transform=input_transform,
        )
        self.num_mixtures = num_mixtures
