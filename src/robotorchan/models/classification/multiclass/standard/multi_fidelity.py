"""Multi-fidelity variational GP classifiers for multiclass and ordinal labels.

Use the same BoTorch fidelity covariance families as binary classification.
Classification posteriors and learning reuse the existing public model paths.
"""

from __future__ import annotations

from collections.abc import Sequence

from botorch.models.kernels.downsampling import DownsamplingKernel
from botorch.models.kernels.exponential_decay import ExponentialDecayKernel
from gpytorch.kernels import Kernel, MaternKernel, ProductKernel, ScaleKernel
from torch import Tensor

from robotorchan.models.base import (
    ContinuousKernelFactory,
    make_mixed_covar_module,
    normalize_feature_dims,
)
from robotorchan.models.classification.multiclass.standard.single_task import (
    MulticlassSingleTaskGPClassifier,
)
from robotorchan.models.classification.ordinal.standard.single_task import (
    OrdinalSingleTaskGPClassifier,
)


def _fidelity_kernel(
    train_X: Tensor,
    *,
    iteration_fidelity: int | None,
    data_fidelities: Sequence[int] | None,
    linear_truncated: bool,
    nu: float,
    design_kernel: Kernel | None = None,
) -> tuple[Kernel, tuple[int, ...], tuple[int, ...]]:
    """Construct the binary-family nonlinear multi-fidelity product kernel."""
    if linear_truncated:
        raise ValueError("linear_truncated=True is not supported by this product kernel.")
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
    fidelity_dims = [*iteration_dims, *data_dims]
    if not fidelity_dims:
        raise ValueError("At least one fidelity feature must be specified.")
    if len(set(fidelity_dims)) != len(fidelity_dims):
        raise ValueError("Fidelity features must be distinct.")
    design_dims = [dim for dim in range(input_dim) if dim not in fidelity_dims]
    if not design_dims:
        raise ValueError("At least one non-fidelity design feature is required.")
    if design_kernel is None:
        design_kernel = MaternKernel(nu=nu, ard_num_dims=len(design_dims), active_dims=design_dims)
    parts: list[Kernel] = [design_kernel]
    if iteration_dims:
        parts.append(ExponentialDecayKernel(active_dims=iteration_dims))
    parts.extend(DownsamplingKernel(active_dims=[dim]) for dim in data_dims)
    return ScaleKernel(ProductKernel(*parts)), tuple(fidelity_dims), tuple(design_dims)


class MultiFidelityMulticlassGPClassifier(MulticlassSingleTaskGPClassifier):
    """Multiclass variational GP with design and fidelity product kernels."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_classes: int,
        iteration_fidelity: int | None = None,
        data_fidelities: Sequence[int] | None = None,
        linear_truncated: bool = False,
        nu: float = 2.5,
        covar_module: Kernel | None = None,
    ) -> None:
        kernel, fidelity_dims, design_dims = _fidelity_kernel(
            train_X,
            iteration_fidelity=iteration_fidelity,
            data_fidelities=data_fidelities,
            linear_truncated=linear_truncated,
            nu=nu,
            design_kernel=covar_module,
        )
        super().__init__(train_X, train_Y, num_classes=num_classes, covar_module=kernel)
        self.iteration_fidelity = (
            normalize_feature_dims(
                [iteration_fidelity], train_X.shape[-1], name="iteration_fidelity"
            )[0]
            if iteration_fidelity is not None
            else None
        )
        self.data_fidelities = tuple(
            normalize_feature_dims(data_fidelities, train_X.shape[-1], name="data_fidelities")
            if data_fidelities is not None
            else ()
        )
        self.fidelity_dims = fidelity_dims
        self.design_dims = design_dims


class MultiFidelityOrdinalGPClassifier(OrdinalSingleTaskGPClassifier):
    """Ordinal-probit GP with BoTorch design/fidelity product covariance."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_classes: int,
        iteration_fidelity: int | None = None,
        data_fidelities: Sequence[int] | None = None,
        linear_truncated: bool = False,
        nu: float = 2.5,
        covar_module: Kernel | None = None,
    ) -> None:
        kernel, fidelity_dims, design_dims = _fidelity_kernel(
            train_X,
            iteration_fidelity=iteration_fidelity,
            data_fidelities=data_fidelities,
            linear_truncated=linear_truncated,
            nu=nu,
            design_kernel=covar_module,
        )
        super().__init__(train_X, train_Y, num_classes=num_classes, covar_module=kernel)
        self.iteration_fidelity = (
            normalize_feature_dims(
                [iteration_fidelity], train_X.shape[-1], name="iteration_fidelity"
            )[0]
            if iteration_fidelity is not None
            else None
        )
        self.data_fidelities = tuple(
            normalize_feature_dims(data_fidelities, train_X.shape[-1], name="data_fidelities")
            if data_fidelities is not None
            else ()
        )
        self.fidelity_dims = fidelity_dims
        self.design_dims = design_dims


class MixedMultiFidelityMulticlassGPClassifier(MultiFidelityMulticlassGPClassifier):
    """Categorical design inputs bypass fidelity dimensions."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_classes: int,
        cat_dims: list[int],
        iteration_fidelity: int | None = None,
        data_fidelities: Sequence[int] | None = None,
        linear_truncated: bool = False,
        nu: float = 2.5,
        cont_kernel_factory: ContinuousKernelFactory | None = None,
    ) -> None:
        fidelity_dims = _get_fidelity_dims(train_X, iteration_fidelity, data_fidelities)
        normalized_cats = normalize_feature_dims(
            cat_dims, train_X.shape[-1], name="cat_dims", excluded_dims=fidelity_dims
        )
        if not normalized_cats:
            raise ValueError("Mixed multi-fidelity classification requires categorical features.")
        design_kernel = make_mixed_covar_module(
            input_dim=train_X.shape[-1],
            cat_dims=normalized_cats,
            excluded_dims=fidelity_dims,
            batch_shape=train_X.shape[:-2],
            cont_kernel_factory=cont_kernel_factory,
        )
        super().__init__(
            train_X,
            train_Y,
            num_classes=num_classes,
            iteration_fidelity=iteration_fidelity,
            data_fidelities=data_fidelities,
            linear_truncated=linear_truncated,
            nu=nu,
            covar_module=design_kernel,
        )
        self.cat_dims = tuple(normalized_cats)


class MixedMultiFidelityOrdinalGPClassifier(MultiFidelityOrdinalGPClassifier):
    """Ordinal-probit multi-fidelity GP with categorical design inputs."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_classes: int,
        cat_dims: list[int],
        iteration_fidelity: int | None = None,
        data_fidelities: Sequence[int] | None = None,
        linear_truncated: bool = False,
        nu: float = 2.5,
        cont_kernel_factory: ContinuousKernelFactory | None = None,
    ) -> None:
        fidelity_dims = _get_fidelity_dims(train_X, iteration_fidelity, data_fidelities)
        normalized_cats = normalize_feature_dims(
            cat_dims, train_X.shape[-1], name="cat_dims", excluded_dims=fidelity_dims
        )
        if not normalized_cats:
            raise ValueError("Mixed multi-fidelity classification requires categorical features.")
        design_kernel = make_mixed_covar_module(
            input_dim=train_X.shape[-1],
            cat_dims=normalized_cats,
            excluded_dims=fidelity_dims,
            batch_shape=train_X.shape[:-2],
            cont_kernel_factory=cont_kernel_factory,
        )
        super().__init__(
            train_X,
            train_Y,
            num_classes=num_classes,
            iteration_fidelity=iteration_fidelity,
            data_fidelities=data_fidelities,
            linear_truncated=linear_truncated,
            nu=nu,
            covar_module=design_kernel,
        )
        self.cat_dims = tuple(normalized_cats)


def _get_fidelity_dims(
    train_X: Tensor,
    iteration_fidelity: int | None,
    data_fidelities: Sequence[int] | None,
) -> list[int]:
    input_dim = train_X.shape[-1]
    dims = []
    if iteration_fidelity is not None:
        dims.extend(
            normalize_feature_dims([iteration_fidelity], input_dim, name="iteration_fidelity")
        )
    if data_fidelities is not None:
        dims.extend(normalize_feature_dims(data_fidelities, input_dim, name="data_fidelities"))
    if not dims:
        raise ValueError("At least one fidelity feature must be specified.")
    if len(set(dims)) != len(dims):
        raise ValueError("Fidelity features must be distinct.")
    return dims
