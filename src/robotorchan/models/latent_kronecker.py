"""Latent Kronecker GP wrapper with robotorchan model conventions."""

from __future__ import annotations

from botorch.models.latent_kronecker_gp import (
    LatentKroneckerGP as BoTorchLatentKroneckerGP,
)
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.utils.types import DEFAULT, _DefaultType
from gpytorch.likelihoods import Likelihood
from gpytorch.means import Mean
from gpytorch.module import Module
from torch import Tensor

from robotorchan.models.base import ExactGPModelMixin


class LatentKroneckerGP(ExactGPModelMixin, BoTorchLatentKroneckerGP):
    """BoTorch ``LatentKroneckerGP`` with robotorchan conveniences.

    Missing-output handling, product-space covariance structure, iterative
    inference, pathwise posterior sampling, and transforms remain delegated to
    BoTorch. robotorchan retains caller-supplied ``train_X``, ``train_T``, and
    ``train_Y`` before upstream broadcasting, masking, or transformations.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_T: Tensor,
        train_Y: Tensor,
        likelihood: Likelihood | None = None,
        mean_module_X: Mean | None = None,
        mean_module_T: Mean | None = None,
        covar_module_X: Module | None = None,
        covar_module_T: Module | None = None,
        input_transform: InputTransform | None = None,
        outcome_transform: OutcomeTransform | _DefaultType | None = DEFAULT,
    ) -> None:
        """Initialize the model while retaining the raw product-space tensors."""
        raw_train_X = train_X.detach().clone()
        raw_train_T = train_T.detach().clone()
        raw_train_Y = train_Y.detach().clone()

        super().__init__(
            train_X=train_X,
            train_T=train_T,
            train_Y=train_Y,
            likelihood=likelihood,
            mean_module_X=mean_module_X,
            mean_module_T=mean_module_T,
            covar_module_X=covar_module_X,
            covar_module_T=covar_module_T,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
        )
        self._store_supervised_training_data(
            train_X=raw_train_X,
            train_Y=raw_train_Y,
            train_Yvar=None,
        )
        self._store_raw_tensor("train_T", raw_train_T)

    @property
    def raw_train_T(self) -> Tensor:
        """Caller-supplied task / time coordinates before upstream broadcasting."""
        value = self._get_raw_tensor("train_T")
        if value is None:  # pragma: no cover - guarded by constructor contract
            raise RuntimeError("raw_train_T was unexpectedly stored as None.")
        return value
