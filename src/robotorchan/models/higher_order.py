"""Higher-order GP wrapper with robotorchan model conventions."""

from __future__ import annotations

from botorch.models.higher_order_gp import HigherOrderGP as BoTorchHigherOrderGP
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.utils.types import DEFAULT, _DefaultType
from gpytorch.kernels import Kernel
from gpytorch.likelihoods import Likelihood
from torch import Tensor

from robotorchan.models.base import (
    ContinuousKernelFactory,
    ExactGPModelMixin,
    _make_mixed_covar_module,
    _normalize_cat_dims,
)


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


class MixedHigherOrderGP(HigherOrderGP):
    """Higher-order GP with native mixed covariance on the design inputs.

    ``HigherOrderGP`` factorizes covariance between the design input ``X`` and
    each tensor-output axis. BoTorch stores the design-input kernel as
    ``covar_modules[0]`` and the remaining kernels describe output axes. This
    wrapper replaces only the design-input factor with robotorchan's native
    mixed covariance, leaving the tensor-output Kronecker structure unchanged.

    Custom ``covar_modules`` are not currently accepted because the Mixed
    wrapper owns the design-input covariance. Supporting custom output-axis
    kernels can be added later without changing the public ``cat_dims`` API.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: list[int],
        likelihood: Likelihood | None = None,
        covar_modules: list[Kernel] | None = None,
        cont_kernel_factory: ContinuousKernelFactory | None = None,
        num_latent_dims: list[int] | None = None,
        learn_latent_pars: bool = True,
        latent_init: str = "default",
        outcome_transform: OutcomeTransform | _DefaultType | None = DEFAULT,
        input_transform: InputTransform | None = None,
    ) -> None:
        if covar_modules is not None:
            raise ValueError(
                "MixedHigherOrderGP manages the design-input covariance internally; "
                "custom covar_modules are not currently supported."
            )

        input_dim = train_X.shape[-1]
        normalized_cat_dims = _normalize_cat_dims(cat_dims=cat_dims, input_dim=input_dim)

        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            likelihood=likelihood,
            covar_modules=None,
            num_latent_dims=num_latent_dims,
            learn_latent_pars=learn_latent_pars,
            latent_init=latent_init,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
        )

        mixed_input_covar = _make_mixed_covar_module(
            input_dim=input_dim,
            cat_dims=normalized_cat_dims,
            batch_shape=train_X.shape[:-2],
            cont_kernel_factory=cont_kernel_factory,
        ).to(train_X)
        self.covar_modules[0] = mixed_input_covar
        self.cat_dims = tuple(normalized_cat_dims)
