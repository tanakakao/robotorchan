"""Robust GP wrappers with robotorchan model conventions."""

from __future__ import annotations

from botorch.models.robust_relevance_pursuit_model import (
    RobustRelevancePursuitSingleTaskGP as BoTorchRobustRelevancePursuitSingleTaskGP,
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


class RobustRelevancePursuitSingleTaskGP(
    ExactGPModelMixin,
    BoTorchRobustRelevancePursuitSingleTaskGP,
):
    """BoTorch robust relevance-pursuit GP with robotorchan conveniences."""

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
        convex_parameterization: bool = True,
        prior_mean_of_support: float | None = None,
        cache_model_trace: bool = False,
    ) -> None:
        raw_train_X = train_X.detach().clone()
        raw_train_Y = train_Y.detach().clone()
        raw_train_Yvar = None if train_Yvar is None else train_Yvar.detach().clone()

        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            likelihood=likelihood,
            covar_module=covar_module,
            mean_module=mean_module,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
            convex_parameterization=convex_parameterization,
            prior_mean_of_support=prior_mean_of_support,
            cache_model_trace=cache_model_trace,
        )
        self._store_supervised_training_data(raw_train_X, raw_train_Y, raw_train_Yvar)


class MixedRobustRelevancePursuitSingleTaskGP(RobustRelevancePursuitSingleTaskGP):
    """Robust relevance-pursuit GP with native mixed categorical covariance.

    Relevance pursuit modifies the likelihood / outlier-noise model rather than
    imposing a special data covariance. BoTorch exposes ``covar_module`` directly,
    so this Mixed wrapper can use robotorchan's native categorical covariance
    without changing the relevance-pursuit fitting semantics.

    ``cat_dims`` refers to categorical columns in the original mixed input space.
    Continuous and categorical effects use the shared robotorchan mixed kernel,
    including their interaction. The robust likelihood and specialized BoTorch
    ``fit_gpytorch_mll`` dispatch remain inherited from the upstream model.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: list[int],
        train_Yvar: Tensor | None = None,
        likelihood: Likelihood | None = None,
        mean_module: Mean | None = None,
        cont_kernel_factory: ContinuousKernelFactory | None = None,
        outcome_transform: OutcomeTransform | _DefaultType | None = DEFAULT,
        input_transform: InputTransform | None = None,
        convex_parameterization: bool = True,
        prior_mean_of_support: float | None = None,
        cache_model_trace: bool = False,
    ) -> None:
        input_dim = train_X.shape[-1]
        normalized_cat_dims = _normalize_cat_dims(cat_dims=cat_dims, input_dim=input_dim)
        _, aug_batch_shape = self.get_batch_dimensions(
            train_X=train_X,
            train_Y=train_Y,
        )
        covar_module = _make_mixed_covar_module(
            input_dim=input_dim,
            cat_dims=normalized_cat_dims,
            batch_shape=aug_batch_shape,
            cont_kernel_factory=cont_kernel_factory,
        )

        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            likelihood=likelihood,
            covar_module=covar_module,
            mean_module=mean_module,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
            convex_parameterization=convex_parameterization,
            prior_mean_of_support=prior_mean_of_support,
            cache_model_trace=cache_model_trace,
        )
        self.cat_dims = tuple(normalized_cat_dims)
