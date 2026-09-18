"""Shared infrastructure for robotorchan model wrappers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import ClassVar

import torch
from botorch.models.kernels.categorical import CategoricalKernel
from botorch.models.utils.gpytorch_modules import get_covar_module_with_dim_scaled_prior
from gpytorch.kernels import AdditiveKernel, Kernel, ProductKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood, MarginalLogLikelihood
from torch import Tensor


def normalize_feature_dims(
    dims: Sequence[int],
    input_dim: int,
    *,
    name: str = "dims",
    excluded_dims: Sequence[int] = (),
    require_nonempty: bool = True,
) -> tuple[int, ...]:
    """Normalize raw-space feature indices with structural-dimension validation.

    Negative indices follow normal Python indexing. Duplicate dimensions are
    rejected after normalization, including aliases such as -1 and d - 1.
    excluded_dims is for structural task, fidelity, context, or hierarchy
    columns that must not be ordinary categorical design variables.
    """
    if input_dim < 1:
        raise ValueError("input_dim must be positive.")

    def _normalize(dim: int) -> int:
        value = int(dim)
        if value < 0:
            value += input_dim
        if value < 0 or value >= input_dim:
            raise ValueError(
                f"{name} entries must index the raw input dimension {input_dim}; got {dim}."
            )
        return value

    normalized = tuple(_normalize(dim) for dim in dims)
    if require_nonempty and not normalized:
        raise ValueError(f"{name} must contain at least one dimension.")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not contain duplicate dimensions after normalization.")

    excluded = tuple(_normalize(dim) for dim in excluded_dims)
    overlap = sorted(set(normalized).intersection(excluded))
    if overlap:
        raise ValueError(f"{name} must not overlap structural dimensions {overlap}.")
    return normalized


def continuous_feature_dims(
    input_dim: int,
    *,
    cat_dims: Sequence[int],
    excluded_dims: Sequence[int] = (),
) -> tuple[int, ...]:
    """Return ordinary continuous design dimensions in raw-input coordinates."""
    batch_shape = torch.Size() if batch_shape is None else batch_shape
    categorical = normalize_feature_dims(cat_dims, input_dim, name="cat_dims")
    structural = normalize_feature_dims(
        excluded_dims,
        input_dim,
        name="excluded_dims",
        require_nonempty=False,
    )
    occupied = set(categorical).union(structural)
    return tuple(dim for dim in range(input_dim) if dim not in occupied)


ContinuousKernelFactory = Callable[[torch.Size, int, list[int]], Kernel]


def make_mixed_covar_module(
    *,
    input_dim: int,
    cat_dims: Sequence[int],
    excluded_dims: Sequence[int] = (),
    batch_shape: torch.Size | None = None,
    cont_kernel_factory: ContinuousKernelFactory | None = None,
) -> Kernel:
    """Build a native mixed covariance over ordinary design dimensions.

    Structural dimensions are excluded from both continuous and categorical
    covariance components. The default follows BoTorch's MixedSingleTaskGP
    structure: continuous + categorical + their interaction.
    """
    categorical = normalize_feature_dims(
        cat_dims,
        input_dim,
        name="cat_dims",
        excluded_dims=excluded_dims,
    )
    continuous = continuous_feature_dims(
        input_dim,
        cat_dims=categorical,
        excluded_dims=excluded_dims,
    )

    def categorical_kernel(*, scaled: bool) -> Kernel:
        kernel: Kernel = CategoricalKernel(
            batch_shape=batch_shape,
            ard_num_dims=len(categorical),
            active_dims=list(categorical),
        )
        return ScaleKernel(kernel, batch_shape=batch_shape) if scaled else kernel

    if not continuous:
        return categorical_kernel(scaled=True)

    def continuous_kernel() -> Kernel:
        if cont_kernel_factory is not None:
            return cont_kernel_factory(batch_shape, len(continuous), list(continuous))
        return get_covar_module_with_dim_scaled_prior(
            ard_num_dims=len(continuous),
            batch_shape=batch_shape,
            active_dims=list(continuous),
        )

    return AdditiveKernel(
        continuous_kernel(),
        categorical_kernel(scaled=True),
        ProductKernel(continuous_kernel(), categorical_kernel(scaled=False)),
    )


class UnsupportedModelOperationError(RuntimeError):
    """Raised when a common robotorchan operation is unsupported by a model."""


class RawDataMixin:
    """Provide provenance snapshots for caller-supplied raw tensors.

    Raw tensors are detached, cloned, and registered as non-persistent buffers.
    They therefore follow model device / dtype changes, but are intentionally
    excluded from ``state_dict`` so that BoTorch-internal model cloning and
    parameter loading are not affected by robotorchan-only provenance data.

    The snapshots represent tensors supplied to the wrapper constructor before
    BoTorch transforms or preprocessing. They are intentionally not rewritten by
    later model-update operations such as ``condition_on_observations`` or
    ``fantasize``.

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
            self.register_buffer(buffer_name, value, persistent=False)

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
