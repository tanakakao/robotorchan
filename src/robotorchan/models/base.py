"""Shared infrastructure for robotorchan model wrappers."""

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar

import torch
from botorch.models.kernels.categorical import CategoricalKernel
from botorch.models.transforms.input import InputTransform
from botorch.models.utils.gpytorch_modules import get_covar_module_with_dim_scaled_prior
from gpytorch.kernels import AdditiveKernel, Kernel, ProductKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood, MarginalLogLikelihood
from torch import Tensor


class UnsupportedModelOperationError(RuntimeError):
    """Raised when a common robotorchan operation is unsupported by a model."""


class RawDataMixin:
    """Provide provenance snapshots for caller-supplied raw tensors.

    Raw tensors are detached, cloned, and registered as buffers. They represent
    the tensors supplied to the wrapper constructor before BoTorch transforms or
    preprocessing. They are intentionally not rewritten by later model-update
    operations such as ``condition_on_observations`` or ``fantasize``.

    This keeps provenance separate from BoTorch's current training state. Use
    native BoTorch attributes such as ``train_inputs`` and ``train_targets``
    when the current conditioned / fantasy training state is required.
    """

    _RAW_BUFFER_PREFIX: ClassVar[str] = "_raw_"
    _raw_data_names: tuple[str, ...]

    def _store_raw_tensor(self, name: str, tensor: Tensor | None) -> None:
        """Store one raw tensor under a stable robotorchan name."""
        if not name or "." in name:
            raise ValueError("Raw-data names must be non-empty and cannot contain '.'.")

        buffer_name = f"{self._RAW_BUFFER_PREFIX}{name}"
        value = None if tensor is None else tensor.detach().clone()

        buffers = getattr(self, "_buffers", {})
        if buffer_name in buffers:
            setattr(self, buffer_name, value)
        else:
            self.register_buffer(buffer_name, value)

        names = list(getattr(self, "_raw_data_names", ()))
        if name not in names:
            names.append(name)
        self._raw_data_names = tuple(names)

    def _get_raw_tensor(self, name: str) -> Tensor | None:
        """Return a raw tensor previously stored with ``_store_raw_tensor``."""
        return getattr(self, f"{self._RAW_BUFFER_PREFIX}{name}")

    @property
    def raw_data_names(self) -> tuple[str, ...]:
        """Names of the constructor-level raw-data snapshots retained by the model."""
        return getattr(self, "_raw_data_names", ())

    @property
    def raw_data(self) -> dict[str, Tensor | None]:
        """Return all constructor-level raw-data snapshots retained by the wrapper."""
        return {name: self._get_raw_tensor(name) for name in self.raw_data_names}


class SupervisedTrainingDataMixin(RawDataMixin):
    """Common raw-data convention for supervised surrogate models."""

    _raw_train_X: Tensor
    _raw_train_Y: Tensor
    _raw_train_Yvar: Tensor | None

    def _store_supervised_training_data(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
    ) -> None:
        """Store detached copies of standard supervised training tensors."""
        self._store_raw_tensor("train_X", train_X)
        self._store_raw_tensor("train_Y", train_Y)
        self._store_raw_tensor("train_Yvar", train_Yvar)

    @property
    def raw_train_X(self) -> Tensor:
        """Constructor-supplied inputs before any model input transform."""
        value = self._get_raw_tensor("train_X")
        if value is None:  # pragma: no cover - guarded by constructor contract
            raise RuntimeError("raw_train_X was unexpectedly stored as None.")
        return value

    @property
    def raw_train_Y(self) -> Tensor:
        """Constructor-supplied outcomes before any model outcome transform."""
        value = self._get_raw_tensor("train_Y")
        if value is None:  # pragma: no cover - guarded by constructor contract
            raise RuntimeError("raw_train_Y was unexpectedly stored as None.")
        return value

    @property
    def raw_train_Yvar(self) -> Tensor | None:
        """Constructor-supplied observation variances, if any."""
        return self._get_raw_tensor("train_Yvar")


class ModelTrainingMixin:
    """Describe common training-objective capabilities of a wrapper."""

    supports_mll: ClassVar[bool] = False

    def make_mll(self) -> MarginalLogLikelihood:
        """Construct this model's marginal-likelihood training objective."""
        raise UnsupportedModelOperationError(f"{type(self).__name__} does not support make_mll().")


class ExactGPModelMixin(SupervisedTrainingDataMixin, ModelTrainingMixin):
    """Common conveniences for exact GPyTorch models."""

    supports_mll: ClassVar[bool] = True

    def make_mll(self) -> ExactMarginalLogLikelihood:
        """Construct the exact marginal log likelihood for this model."""
        return ExactMarginalLogLikelihood(self.likelihood, self)


ContinuousKernelFactory = Callable[[torch.Size, int, list[int]], Kernel]


def _normalize_dims(dims: list[int], input_dim: int, *, name: str) -> list[int]:
    """Normalize feature indices against an input dimension."""
    if input_dim <= 0:
        raise ValueError("input_dim must be positive.")

    normalized: list[int] = []
    for dim in dims:
        resolved = dim + input_dim if dim < 0 else dim
        if resolved < 0 or resolved >= input_dim:
            raise ValueError(f"{name} dimension {dim} is out of range for input_dim={input_dim}.")
        normalized.append(resolved)

    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not contain duplicate feature indices.")
    return sorted(normalized)


def _normalize_cat_dims(cat_dims: list[int], input_dim: int) -> list[int]:
    """Normalize categorical feature indices against an input dimension."""
    if not cat_dims:
        raise ValueError("cat_dims must contain at least one categorical feature index.")
    return _normalize_dims(cat_dims, input_dim, name="Categorical")


class _CategoricalOneHotInputTransform(InputTransform):
    """Encode categorical scalar columns while keeping a raw mixed public space.

    This private transform is used only for model families whose upstream
    covariance architecture cannot safely accept robotorchan's native mixed
    covariance. Category values are learned from ``train_X`` and stored as
    buffers so training, posterior, conditioning, serialization, and dtype /
    device moves share the same encoding.
    """

    def __init__(self, train_X: Tensor, cat_dims: list[int]) -> None:
        super().__init__()
        self.transform_on_train = True
        self.transform_on_eval = True
        self.transform_on_fantasize = True
        self.raw_input_dim = train_X.shape[-1]
        self.cat_dims = tuple(_normalize_cat_dims(cat_dims, self.raw_input_dim))
        self._category_buffer_names = tuple(
            f"_category_values_{index}" for index in range(len(self.cat_dims))
        )

        encoded_input_dim = self.raw_input_dim
        for name, dim in zip(self._category_buffer_names, self.cat_dims, strict=True):
            values = torch.unique(train_X[..., dim]).sort().values.detach().clone()
            if values.numel() == 0:  # pragma: no cover - impossible for valid training data
                raise ValueError(f"Categorical feature {dim} has no observed values.")
            self.register_buffer(name, values)
            encoded_input_dim += values.numel() - 1
        self.encoded_input_dim = encoded_input_dim

    @property
    def category_values(self) -> tuple[Tensor, ...]:
        """Observed category values in normalized ``cat_dims`` order."""
        return tuple(getattr(self, name) for name in self._category_buffer_names)

    def transform(self, X: Tensor) -> Tensor:
        """One-hot encode categorical columns in a raw mixed-space tensor."""
        if X.shape[-1] != self.raw_input_dim:
            raise ValueError(
                f"Expected inputs with {self.raw_input_dim} features, got {X.shape[-1]}."
            )

        value_by_dim = dict(zip(self.cat_dims, self.category_values, strict=True))
        parts: list[Tensor] = []
        for dim in range(self.raw_input_dim):
            if dim not in value_by_dim:
                parts.append(X[..., dim : dim + 1])
                continue

            values = value_by_dim[dim].to(device=X.device, dtype=X.dtype)
            shape = (1,) * (X.ndim - 1) + (values.numel(),)
            encoded = X[..., dim : dim + 1] == values.reshape(shape)
            if not torch.all(encoded.sum(dim=-1) == 1):
                raise ValueError(f"Input contains an unseen category in categorical feature {dim}.")
            parts.append(encoded.to(dtype=X.dtype))

        return torch.cat(parts, dim=-1)

    def equals(self, other: InputTransform) -> bool:
        """Return whether another transform has the same encoding contract."""
        return (
            isinstance(other, _CategoricalOneHotInputTransform)
            and self.raw_input_dim == other.raw_input_dim
            and self.cat_dims == other.cat_dims
            and super().equals(other)
        )


def _get_cont_dims(
    *,
    input_dim: int,
    cat_dims: list[int],
    excluded_dims: list[int] | None = None,
) -> list[int]:
    """Return continuous feature indices excluding categorical/structural columns."""
    normalized_cat_dims = _normalize_cat_dims(cat_dims=cat_dims, input_dim=input_dim)
    normalized_excluded_dims = _normalize_dims(excluded_dims or [], input_dim, name="Excluded")
    overlap = set(normalized_cat_dims).intersection(normalized_excluded_dims)
    if overlap:
        raise ValueError("cat_dims and excluded_dims must be disjoint.")

    unavailable = set(normalized_cat_dims) | set(normalized_excluded_dims)
    return [dim for dim in range(input_dim) if dim not in unavailable]


def _make_mixed_covar_module(
    *,
    input_dim: int,
    cat_dims: list[int],
    excluded_dims: list[int] | None = None,
    batch_shape: torch.Size | None = None,
    cont_kernel_factory: ContinuousKernelFactory | None = None,
) -> Kernel:
    """Build the default robotorchan covariance for a mixed input space."""
    resolved_batch_shape = torch.Size() if batch_shape is None else batch_shape
    normalized_cat_dims = _normalize_cat_dims(cat_dims=cat_dims, input_dim=input_dim)
    normalized_excluded_dims = _normalize_dims(excluded_dims or [], input_dim, name="Excluded")
    overlap = set(normalized_cat_dims).intersection(normalized_excluded_dims)
    if overlap:
        raise ValueError("cat_dims and excluded_dims must be disjoint.")

    cont_dims = _get_cont_dims(
        input_dim=input_dim,
        cat_dims=normalized_cat_dims,
        excluded_dims=normalized_excluded_dims,
    )

    def make_categorical_kernel(*, scaled: bool) -> Kernel:
        kernel: Kernel = CategoricalKernel(
            batch_shape=resolved_batch_shape,
            ard_num_dims=len(normalized_cat_dims),
            active_dims=normalized_cat_dims,
        )
        if scaled:
            kernel = ScaleKernel(kernel, batch_shape=resolved_batch_shape)
        return kernel

    if not cont_dims:
        return make_categorical_kernel(scaled=True)

    def make_continuous_kernel() -> Kernel:
        if cont_kernel_factory is not None:
            return cont_kernel_factory(resolved_batch_shape, len(cont_dims), cont_dims)
        return get_covar_module_with_dim_scaled_prior(
            ard_num_dims=len(cont_dims),
            batch_shape=resolved_batch_shape,
            active_dims=cont_dims,
        )

    continuous_main = make_continuous_kernel()
    continuous_interaction = make_continuous_kernel()
    categorical_main = make_categorical_kernel(scaled=True)
    categorical_interaction = make_categorical_kernel(scaled=False)

    return AdditiveKernel(
        continuous_main,
        categorical_main,
        ProductKernel(continuous_interaction, categorical_interaction),
    )
