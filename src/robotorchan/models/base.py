"""Shared model conveniences used by robotorchan wrappers."""

from __future__ import annotations

from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor


class RawTrainingDataMixin:
    """Store untransformed training tensors supplied at model construction.

    Raw tensors are registered as buffers so that they follow the model across
    devices and dtypes and participate in ``state_dict`` serialization.
    """

    _raw_train_X: Tensor
    _raw_train_Y: Tensor
    _raw_train_Yvar: Tensor | None

    def _store_raw_training_data(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
    ) -> None:
        """Store detached copies of the tensors supplied by the caller."""
        self.register_buffer("_raw_train_X", train_X.detach().clone())
        self.register_buffer("_raw_train_Y", train_Y.detach().clone())
        self.register_buffer(
            "_raw_train_Yvar",
            None if train_Yvar is None else train_Yvar.detach().clone(),
        )

    @property
    def raw_train_X(self) -> Tensor:
        """Training inputs before any model input transform is applied."""
        return self._raw_train_X

    @property
    def raw_train_Y(self) -> Tensor:
        """Training outcomes before any model outcome transform is applied."""
        return self._raw_train_Y

    @property
    def raw_train_Yvar(self) -> Tensor | None:
        """Observation variances supplied by the caller, if any."""
        return self._raw_train_Yvar


class ExactGPModelMixin(RawTrainingDataMixin):
    """Common conveniences for exact GPyTorch models."""

    def make_mll(self) -> ExactMarginalLogLikelihood:
        """Construct the exact marginal log likelihood for this model."""
        return ExactMarginalLogLikelihood(self.likelihood, self)
