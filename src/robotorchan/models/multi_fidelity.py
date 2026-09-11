"""Multi-fidelity GP wrapper with robotorchan model conventions."""

from __future__ import annotations

from collections.abc import Sequence

from botorch.models import SingleTaskMultiFidelityGP as BoTorchSingleTaskMultiFidelityGP
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.utils.types import DEFAULT, _DefaultType
from gpytorch.likelihoods import Likelihood
from gpytorch.module import Module
from torch import Tensor

from robotorchan.models.base import ExactGPModelMixin


class SingleTaskMultiFidelityGP(ExactGPModelMixin, BoTorchSingleTaskMultiFidelityGP):
    """BoTorch ``SingleTaskMultiFidelityGP`` with robotorchan conveniences."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        iteration_fidelity: int | None = None,
        data_fidelities: Sequence[int] | None = None,
        linear_truncated: bool = True,
        nu: float = 2.5,
        covar_module: Module | None = None,
        likelihood: Likelihood | None = None,
        outcome_transform: OutcomeTransform | _DefaultType | None = DEFAULT,
        input_transform: InputTransform | None = None,
    ) -> None:
        """Initialize the multi-fidelity GP while retaining raw training tensors."""
        raw_train_X = train_X.detach().clone()
        raw_train_Y = train_Y.detach().clone()
        raw_train_Yvar = None if train_Yvar is None else train_Yvar.detach().clone()

        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            iteration_fidelity=iteration_fidelity,
            data_fidelities=data_fidelities,
            linear_truncated=linear_truncated,
            nu=nu,
            covar_module=covar_module,
            likelihood=likelihood,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
        )
        self._store_supervised_training_data(
            train_X=raw_train_X,
            train_Y=raw_train_Y,
            train_Yvar=raw_train_Yvar,
        )
