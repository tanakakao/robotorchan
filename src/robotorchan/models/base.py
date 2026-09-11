"""Shared infrastructure for robotorchan model wrappers."""

from __future__ import annotations

from typing import ClassVar

from gpytorch.mlls import ExactMarginalLogLikelihood, MarginalLogLikelihood
from torch import Tensor


class UnsupportedModelOperationError(RuntimeError):
    """Raised when a common robotorchan operation is unsupported by a model."""


class RawDataMixin:
    """Provide generic storage for caller-supplied raw tensors.

    Raw tensors are detached, cloned, and registered as buffers. This keeps
    them independent from caller-owned tensors while preserving normal PyTorch
    device / dtype moves and ``state_dict`` serialization behavior.
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
    def raw_data(self) -> dict[str, Tensor | None]:
        """Return all raw tensors registered by this wrapper.

        The returned dictionary is a new mapping, while tensor values refer to
        the model-owned buffers.
        """
        return {name: self._get_raw_tensor(name) for name in self._raw_data_names}


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
        """Training inputs before any model input transform is applied."""
        value = self._get_raw_tensor("train_X")
        if value is None:  # pragma: no cover - guarded by constructor contract
            raise RuntimeError("raw_train_X was unexpectedly stored as None.")
        return value

    @property
    def raw_train_Y(self) -> Tensor:
        """Training outcomes before any model outcome transform is applied."""
        value = self._get_raw_tensor("train_Y")
        if value is None:  # pragma: no cover - guarded by constructor contract
            raise RuntimeError("raw_train_Y was unexpectedly stored as None.")
        return value

    @property
    def raw_train_Yvar(self) -> Tensor | None:
        """Observation variances supplied by the caller, if any."""
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
        raise UnsupportedModelOperationError(
            f"{type(self).__name__} does not support make_mll()."
        )


class ExactGPModelMixin(SupervisedTrainingDataMixin, ModelTrainingMixin):
    """Common conveniences for exact GPyTorch models."""

    supports_mll: ClassVar[bool] = True

    def make_mll(self) -> ExactMarginalLogLikelihood:
        """Construct the exact marginal log likelihood for this model."""
        return ExactMarginalLogLikelihood(self.likelihood, self)


# Backward-compatible internal alias retained while the wrapper API is young.
RawTrainingDataMixin = SupervisedTrainingDataMixin
