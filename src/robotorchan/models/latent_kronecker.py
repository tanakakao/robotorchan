"""Latent Kronecker GP wrappers with robotorchan model conventions."""

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

from robotorchan.models.base import (
    ContinuousKernelFactory,
    ExactGPModelMixin,
    _make_mixed_covar_module,
    _normalize_cat_dims,
)


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


class MixedLatentKroneckerGP(LatentKroneckerGP):
    """Latent Kronecker GP with native mixed covariance on the X factor.

    The Kronecker structure separates design covariance ``K_X`` from the
    task/time covariance ``K_T``. Since BoTorch exposes ``covar_module_X``
    directly, categorical design dimensions can use robotorchan's native mixed
    covariance without changing the latent Kronecker or iterative-inference
    semantics. ``train_T`` remains entirely under the upstream T covariance.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_T: Tensor,
        train_Y: Tensor,
        cat_dims: list[int],
        likelihood: Likelihood | None = None,
        mean_module_X: Mean | None = None,
        mean_module_T: Mean | None = None,
        cont_kernel_factory: ContinuousKernelFactory | None = None,
        covar_module_T: Module | None = None,
        input_transform: InputTransform | None = None,
        outcome_transform: OutcomeTransform | _DefaultType | None = DEFAULT,
    ) -> None:
        input_dim = train_X.shape[-1]
        normalized_cat_dims = _normalize_cat_dims(cat_dims=cat_dims, input_dim=input_dim)
        covar_module_X = _make_mixed_covar_module(
            input_dim=input_dim,
            cat_dims=normalized_cat_dims,
            batch_shape=train_X.shape[:-2],
            cont_kernel_factory=cont_kernel_factory,
        )
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
        self.cat_dims = tuple(normalized_cat_dims)
