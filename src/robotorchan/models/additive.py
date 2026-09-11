"""Additive GP wrappers with robotorchan model conventions."""

from __future__ import annotations

from botorch.models.additive_gp import OrthogonalAdditiveGP as BoTorchOrthogonalAdditiveGP
from botorch.models.kernels.orthogonal_additive_kernel import OrthogonalAdditiveKernel
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.utils.types import DEFAULT, _DefaultType
from gpytorch.likelihoods import Likelihood
from gpytorch.means import Mean
from torch import Tensor

from robotorchan.models.base import ExactGPModelMixin


class OrthogonalAdditiveGP(ExactGPModelMixin, BoTorchOrthogonalAdditiveGP):
    """BoTorch ``OrthogonalAdditiveGP`` with robotorchan raw-data retention."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        covar_module: OrthogonalAdditiveKernel | None = None,
        second_order: bool = False,
        likelihood: Likelihood | None = None,
        mean_module: Mean | None = None,
        outcome_transform: OutcomeTransform | _DefaultType | None = DEFAULT,
        input_transform: InputTransform | None = None,
    ) -> None:
        raw_train_X = train_X.detach().clone()
        raw_train_Y = train_Y.detach().clone()

        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            covar_module=covar_module,
            second_order=second_order,
            likelihood=likelihood,
            mean_module=mean_module,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
        )
        self._store_supervised_training_data(raw_train_X, raw_train_Y, None)
