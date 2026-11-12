"""Continuous-input multi-fidelity binary variational GP classifier."""

from __future__ import annotations

from collections.abc import Sequence

from botorch.models.kernels.downsampling import DownsamplingKernel
from botorch.models.kernels.exponential_decay import ExponentialDecayKernel
from botorch.models.transforms.input import InputTransform
from botorch.models.utils.inducing_point_allocators import InducingPointAllocator
from gpytorch.kernels import Kernel, MaternKernel, ProductKernel, ScaleKernel
from gpytorch.means import Mean
from gpytorch.variational import VariationalStrategy, _VariationalDistribution, _VariationalStrategy
from torch import Tensor

from robotorchan.models.base import (
    ContinuousKernelFactory,
    make_mixed_covar_module,
    normalize_feature_dims,
)
from robotorchan.models.classification.binary.standard.single_task import (
    BinarySingleTaskGPClassifier,
)


class MultiFidelityBinaryGPClassifier(BinarySingleTaskGPClassifier):
    """Bernoulli variational GP with separate design and fidelity covariance.

    Fidelity inputs use BoTorch's native exponential-decay and downsampling
    kernels. The product structure follows the non-linear-truncated
    multi-fidelity model; the linear-truncated prior is not supported.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        iteration_fidelity: int | None = None,
        data_fidelities: Sequence[int] | None = None,
        linear_truncated: bool = False,
        nu: float = 2.5,
        learn_inducing_points: bool = True,
        mean_module: Mean | None = None,
        covar_module: Kernel | None = None,
        variational_distribution: _VariationalDistribution | None = None,
        variational_strategy: type[_VariationalStrategy] = VariationalStrategy,
        inducing_points: Tensor | int | None = None,
        inducing_point_allocator: InducingPointAllocator | None = None,
        input_transform: InputTransform | None = None,
    ) -> None:
        if linear_truncated:
            raise ValueError(
                "linear_truncated=True is not supported by the binary "
                "multi-fidelity product kernel."
            )
        input_dim = train_X.shape[-1]
        iteration_dims = (
            []
            if iteration_fidelity is None
            else normalize_feature_dims([iteration_fidelity], input_dim, name="iteration_fidelity")
        )
        data_dims = (
            []
            if data_fidelities is None
            else normalize_feature_dims(data_fidelities, input_dim, name="data_fidelities")
        )
        fidelity_dims = list(iteration_dims) + list(data_dims)
        if not fidelity_dims:
            raise ValueError("At least one fidelity feature must be specified.")
        if len(set(fidelity_dims)) != len(fidelity_dims):
            raise ValueError("Fidelity features must be distinct.")
        design_dims = [i for i in range(input_dim) if i not in fidelity_dims]
        if not design_dims:
            raise ValueError("At least one non-fidelity design feature is required.")

        design_kernel = covar_module
        if design_kernel is None:
            design_kernel = MaternKernel(
                nu=nu, ard_num_dims=len(design_dims), active_dims=design_dims
            )
        kernels = [design_kernel]
        if iteration_dims:
            kernels.append(ExponentialDecayKernel(active_dims=iteration_dims))
        kernels.extend(DownsamplingKernel(active_dims=[dim]) for dim in data_dims)
        combined_covar_module = ScaleKernel(ProductKernel(*kernels))
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            learn_inducing_points=learn_inducing_points,
            covar_module=combined_covar_module,
            mean_module=mean_module,
            variational_distribution=variational_distribution,
            variational_strategy=variational_strategy,
            inducing_points=inducing_points,
            inducing_point_allocator=inducing_point_allocator,
            input_transform=input_transform,
        )
        self.iteration_fidelity = iteration_dims[0] if iteration_dims else None
        self.data_fidelities = tuple(data_dims)
        self.fidelity_dims = tuple(fidelity_dims)
        self.design_dims = tuple(design_dims)


class MixedMultiFidelityBinaryGPClassifier(MultiFidelityBinaryGPClassifier):
    """Mixed categorical/continuous binary GP with native fidelity kernels.

    Categorical features belong to the design space and cannot overlap
    iteration or data fidelity features.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        cat_dims: list[int],
        iteration_fidelity: int | None = None,
        data_fidelities: Sequence[int] | None = None,
        linear_truncated: bool = False,
        nu: float = 2.5,
        learn_inducing_points: bool = True,
        cont_kernel_factory: ContinuousKernelFactory | None = None,
        mean_module: Mean | None = None,
        variational_distribution: _VariationalDistribution | None = None,
        variational_strategy: type[_VariationalStrategy] = VariationalStrategy,
        inducing_points: Tensor | int | None = None,
        inducing_point_allocator: InducingPointAllocator | None = None,
        input_transform: InputTransform | None = None,
    ) -> None:
        input_dim = train_X.shape[-1]
        fidelity_dims: list[int] = []
        if iteration_fidelity is not None:
            fidelity_dims.extend(
                normalize_feature_dims([iteration_fidelity], input_dim, name="iteration_fidelity")
            )
        if data_fidelities is not None:
            fidelity_dims.extend(
                normalize_feature_dims(data_fidelities, input_dim, name="data_fidelities")
            )
        if not fidelity_dims:
            raise ValueError("At least one fidelity feature must be specified.")
        if len(set(fidelity_dims)) != len(fidelity_dims):
            raise ValueError("Fidelity features must be distinct.")
        normalized_cat_dims = normalize_feature_dims(
            cat_dims, input_dim, name="cat_dims", excluded_dims=fidelity_dims
        )
        if not normalized_cat_dims:
            raise ValueError("Mixed multi-fidelity classification requires categorical features.")
        design_kernel = make_mixed_covar_module(
            input_dim=input_dim,
            cat_dims=normalized_cat_dims,
            excluded_dims=fidelity_dims,
            batch_shape=train_X.shape[:-2],
            cont_kernel_factory=cont_kernel_factory,
        )
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            iteration_fidelity=iteration_fidelity,
            data_fidelities=data_fidelities,
            linear_truncated=linear_truncated,
            nu=nu,
            learn_inducing_points=learn_inducing_points,
            mean_module=mean_module,
            covar_module=design_kernel,
            variational_distribution=variational_distribution,
            variational_strategy=variational_strategy,
            inducing_points=inducing_points,
            inducing_point_allocator=inducing_point_allocator,
            input_transform=input_transform,
        )
        self.cat_dims = tuple(normalized_cat_dims)
