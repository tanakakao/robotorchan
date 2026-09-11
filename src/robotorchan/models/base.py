"""Shared infrastructure for robotorchan model wrappers."""

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar

import torch
from botorch.models.kernels.categorical import CategoricalKernel
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
        """Store one raw tensor under a stable robotorchan name.

        Args:
            name: Public raw-data name without the ``raw_`` prefix, for example
                ``train_X`` or ``comparisons``.
            tensor: Tensor supplied by the caller, or ``None`` when the concept
                applies but no tensor was supplied.

        Raises:
            ValueError: If ``name`` is empty or cannot be used as a PyTorch
                buffer name.
        """
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
        """Return all constructor-level raw-data snapshots retained by the wrapper.

        The returned dictionary is a new mapping, while tensor values refer to
        the model-owned buffers.
        """
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
        """Construct this model's marginal-likelihood training objective.

        Raises:
            UnsupportedModelOperationError: If the model does not use an MLL
                style training objective.
        """
        raise UnsupportedModelOperationError(f"{type(self).__name__} does not support make_mll().")


class ExactGPModelMixin(SupervisedTrainingDataMixin, ModelTrainingMixin):
    """Common conveniences for exact GPyTorch models."""

    supports_mll: ClassVar[bool] = True

    def make_mll(self) -> ExactMarginalLogLikelihood:
        """Construct the exact marginal log likelihood for this model."""
        return ExactMarginalLogLikelihood(self.likelihood, self)


ContinuousKernelFactory = Callable[[torch.Size, int, list[int]], Kernel]


def _normalize_cat_dims(cat_dims: list[int], input_dim: int) -> list[int]:
    """Normalize categorical feature indices against an input dimension.

    Negative indices follow normal Python indexing semantics. The returned
    indices are unique, non-negative, and sorted so downstream kernel
    construction has a stable representation.

    Args:
        cat_dims: Categorical feature indices.
        input_dim: Total number of input features.

    Returns:
        Normalized categorical feature indices.

    Raises:
        ValueError: If ``input_dim`` is not positive, ``cat_dims`` is empty,
            contains duplicates, or contains an out-of-range index.
    """
    if input_dim <= 0:
        raise ValueError("input_dim must be positive.")
    if not cat_dims:
        raise ValueError("cat_dims must contain at least one categorical feature index.")

    normalized: list[int] = []
    for dim in cat_dims:
        resolved = dim + input_dim if dim < 0 else dim
        if resolved < 0 or resolved >= input_dim:
            raise ValueError(
                f"Categorical dimension {dim} is out of range for input_dim={input_dim}."
            )
        normalized.append(resolved)

    if len(set(normalized)) != len(normalized):
        raise ValueError("cat_dims must not contain duplicate feature indices.")

    return sorted(normalized)


def _get_cont_dims(*, input_dim: int, cat_dims: list[int]) -> list[int]:
    """Return continuous feature indices complementary to ``cat_dims``."""
    normalized_cat_dims = _normalize_cat_dims(cat_dims=cat_dims, input_dim=input_dim)
    categorical = set(normalized_cat_dims)
    return [dim for dim in range(input_dim) if dim not in categorical]


def _make_mixed_covar_module(
    *,
    input_dim: int,
    cat_dims: list[int],
    batch_shape: torch.Size = torch.Size(),
    cont_kernel_factory: ContinuousKernelFactory | None = None,
) -> Kernel:
    """Build the default robotorchan covariance for a mixed input space.

    For mixed continuous/categorical inputs, the covariance is the sum of a
    continuous component, a categorical component, and their interaction. For
    categorical-only inputs, a scaled categorical covariance is returned.

    This helper is intentionally private. Model wrappers expose ``cat_dims`` and
    delegate kernel construction here so mixed-space behavior stays consistent
    across single-task, multi-task, variational, and other model families.

    Args:
        input_dim: Total number of model input features handled by this kernel.
        cat_dims: Categorical feature indices. Negative indices are supported.
        batch_shape: Batch shape for kernel hyperparameters.
        cont_kernel_factory: Optional continuous-kernel factory with the same
            calling convention used by BoTorch ``MixedSingleTaskGP``.

    Returns:
        A GPyTorch covariance module for the mixed input space.
    """
    normalized_cat_dims = _normalize_cat_dims(cat_dims=cat_dims, input_dim=input_dim)
    cont_dims = _get_cont_dims(input_dim=input_dim, cat_dims=normalized_cat_dims)

    def make_categorical_kernel(*, scaled: bool) -> Kernel:
        kernel: Kernel = CategoricalKernel(
            batch_shape=batch_shape,
            ard_num_dims=len(normalized_cat_dims),
            active_dims=normalized_cat_dims,
        )
        if scaled:
            kernel = ScaleKernel(kernel, batch_shape=batch_shape)
        return kernel

    if not cont_dims:
        return make_categorical_kernel(scaled=True)

    factory = cont_kernel_factory or get_covar_module_with_dim_scaled_prior
    continuous_main = factory(batch_shape, len(cont_dims), cont_dims)
    continuous_interaction = factory(batch_shape, len(cont_dims), cont_dims)
    categorical_main = make_categorical_kernel(scaled=True)
    categorical_interaction = make_categorical_kernel(scaled=False)

    return AdditiveKernel(
        continuous_main,
        categorical_main,
        ProductKernel(continuous_interaction, categorical_interaction),
    )
