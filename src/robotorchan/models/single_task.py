"""Single-task GP wrapper with robotorchan model conventions."""

from __future__ import annotations

from botorch.models import SingleTaskGP as BoTorchSingleTaskGP
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.utils.types import DEFAULT, _DefaultType
from gpytorch.likelihoods import Likelihood
from gpytorch.means import Mean
from gpytorch.module import Module
from torch import Tensor

from robotorchan.models.base import ExactGPModelMixin


class SingleTaskGP(ExactGPModelMixin, BoTorchSingleTaskGP):
    """BoTorch ``SingleTaskGP`` with robotorchan convenience features.

    The predictive model and posterior behavior are inherited directly from
    BoTorch. robotorchan only adds the common wrapper surface: raw training
    data retention, training-capability metadata, and ``make_mll()``.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        likelihood: Likelihood | None = None,
        covar_module: Module | None = None,
        mean_module: Mean | None = None,
        outcome_transform: OutcomeTransform | _DefaultType | None = DEFAULT,
        input_transform: InputTransform | None = None,
    ) -> None:
        """Initialize the model while retaining the caller's raw tensors."""
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            likelihood=likelihood,
            covar_module=covar_module,
            mean_module=mean_module,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
        )
        self._store_supervised_training_data(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
        )
