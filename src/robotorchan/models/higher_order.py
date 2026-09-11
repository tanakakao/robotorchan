"""Higher-order GP wrapper with robotorchan model conventions."""

from __future__ import annotations

from botorch.models.higher_order_gp import HigherOrderGP as BoTorchHigherOrderGP
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.utils.types import DEFAULT, _DefaultType
from gpytorch.kernels import Kernel
from gpytorch.likelihoods import Likelihood
from torch import Tensor

from robotorchan.models.base import ExactGPModelMixin


class HigherOrderGP(ExactGPModelMixin, BoTorchHigherOrderGP):
    """BoTorch ``HigherOrderGP`` with robotorchan convenience features.

    Tensor-output covariance structure, latent parameters, transforms, posterior
    sampling, and specialized Kronecker solves remain delegated to BoTorch.
    robotorchan only retains caller-supplied raw training tensors and provides
    the common exact-GP ``make_mll()`` helper.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        likelihood: Likelihood | None = None,
        covar_modules: list[Kernel] | None = None,
        num_latent_dims: list[int] | None = None,
        learn_latent_pars: bool = True,
        latent_init: str = "default",
        outcome_transform: OutcomeTransform | _DefaultType | None = DEFAULT,
        input_transform: InputTransform | None = None,
    ) -> None:
        """Initialize the HOGP while retaining caller-supplied raw tensors."""
        raw_train_X = train_X.detach().clone()
        raw_train_Y = train_Y.detach().clone()

        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            likelihood=likelihood,
            covar_modules=covar_modules,
            num_latent_dims=num_latent_dims,
            learn_latent_pars=learn_latent_pars,
            latent_init=latent_init,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
        )
        self._store_supervised_training_data(
            train_X=raw_train_X,
            train_Y=raw_train_Y,
            train_Yvar=None,
        )
