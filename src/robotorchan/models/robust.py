"""Robust GP wrappers with robotorchan model conventions."""

from __future__ import annotations

import torch
from botorch.models.robust_relevance_pursuit_model import (
    RobustRelevancePursuitSingleTaskGP as BoTorchRobustRelevancePursuitSingleTaskGP,
)
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.utils.types import DEFAULT, _DefaultType
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood, Likelihood
from gpytorch.means import Mean
from gpytorch.module import Module
from torch import Tensor

from robotorchan.models.base import (
    ContinuousKernelFactory,
    ExactGPModelMixin,
    make_mixed_covar_module,
    normalize_feature_dims,
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
    """Robust relevance-pursuit GP with native mixed categorical covariance."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
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
        normalized_cat_dims = normalize_feature_dims(cat_dims, input_dim, name="cat_dims")
        _, aug_batch_shape = self.get_batch_dimensions(train_X=train_X, train_Y=train_Y)
        covar_module = make_mixed_covar_module(
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
        self.cat_dims = normalized_cat_dims


class HeteroskedasticSingleTaskGP(ExactGPModelMixin, BoTorchRobustRelevancePursuitSingleTaskGP):
    """Iterative two-GP surrogate for input-dependent observation noise."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        noise_floor: float = 1e-6,
        outcome_transform: OutcomeTransform | _DefaultType | None = DEFAULT,
        input_transform: InputTransform | None = None,
    ) -> None:
        if noise_floor <= 0:
            raise ValueError("noise_floor must be positive.")
        self.noise_floor = noise_floor
        self._noise_model_fitted = False
        initial_noise = torch.full_like(train_Y, noise_floor)
        likelihood = FixedNoiseGaussianLikelihood(
            noise=initial_noise.squeeze(-1),
            learn_additional_noise=False,
        )
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=initial_noise,
            likelihood=likelihood,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
        )
        self._store_supervised_training_data(train_X, train_Y, initial_noise)
        self.noise_model = None

    def fit_heteroskedastic(self, *, iterations: int = 3) -> HeteroskedasticSingleTaskGP:
        """Alternately fit the mean GP and a GP for log residual variance."""
        from botorch.fit import fit_gpytorch_mll

        from robotorchan.models.single_task import SingleTaskGP

        if iterations < 1:
            raise ValueError("iterations must be at least 1.")
        train_X = self.raw_train_X
        train_Y = self.raw_train_Y
        for _ in range(iterations):
            fit_gpytorch_mll(self.make_mll())
            with torch.no_grad():
                residual = train_Y - self.posterior(train_X).mean
                log_noise = torch.log(residual.square().clamp_min(self.noise_floor))
            noise_model = SingleTaskGP(train_X, log_noise)
            fit_gpytorch_mll(noise_model.make_mll())
            with torch.no_grad():
                predicted_noise = noise_model.posterior(train_X).mean.exp()
                predicted_noise = predicted_noise.clamp_min(self.noise_floor)
            self.likelihood.noise_covar.noise = predicted_noise.squeeze(-1)
            self.noise_model = noise_model
            self._noise_model_fitted = True
        return self

    def noise_posterior(self, X: Tensor):
        """Return the latent posterior for log observation variance."""
        if self.noise_model is None or not self._noise_model_fitted:
            raise RuntimeError("fit_heteroskedastic must be called before noise_posterior.")
        self.noise_model.prediction_strategy = None
        return self.noise_model.posterior(X)

    def predicted_noise(self, X: Tensor) -> Tensor:
        """Return input-dependent observation variance on the original scale."""
        return self.noise_posterior(X).mean.exp().clamp_min(self.noise_floor)


class MixedHeteroskedasticSingleTaskGP(RobustRelevancePursuitSingleTaskGP):
    """Iterative heteroskedastic GP with native mixed categorical covariance."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        cat_dims: list[int],
        noise_floor: float = 1e-6,
        cont_kernel_factory: ContinuousKernelFactory | None = None,
        outcome_transform: OutcomeTransform | _DefaultType | None = DEFAULT,
        input_transform: InputTransform | None = None,
    ) -> None:
        if noise_floor <= 0:
            raise ValueError("noise_floor must be positive.")
        self.noise_floor = noise_floor
        self._noise_model_fitted = False
        input_dim = train_X.shape[-1]
        normalized_cat_dims = normalize_feature_dims(cat_dims, input_dim, name="cat_dims")
        _, aug_batch_shape = self.get_batch_dimensions(train_X=train_X, train_Y=train_Y)
        covar_module = make_mixed_covar_module(
            input_dim=input_dim,
            cat_dims=normalized_cat_dims,
            batch_shape=aug_batch_shape,
            cont_kernel_factory=cont_kernel_factory,
        )
        initial_noise = torch.full_like(train_Y, noise_floor)
        likelihood = FixedNoiseGaussianLikelihood(
            noise=initial_noise.squeeze(-1),
            learn_additional_noise=False,
        )
        RobustRelevancePursuitSingleTaskGP.__init__(
            self,
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=initial_noise,
            likelihood=likelihood,
            covar_module=covar_module,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
        )
        self.cat_dims = normalized_cat_dims
        self.noise_model = None

    def fit_heteroskedastic(self, *, iterations: int = 3) -> MixedHeteroskedasticSingleTaskGP:
        """Alternately fit mixed response and mixed log-noise GPs."""
        from botorch.fit import fit_gpytorch_mll

        from robotorchan.models.single_task import MixedSingleTaskGP

        if iterations < 1:
            raise ValueError("iterations must be at least 1.")
        train_X = self.raw_train_X
        train_Y = self.raw_train_Y
        for _ in range(iterations):
            fit_gpytorch_mll(self.make_mll())
            with torch.no_grad():
                residual = train_Y - self.posterior(train_X).mean
                log_noise = torch.log(residual.square().clamp_min(self.noise_floor))
            noise_model = MixedSingleTaskGP(train_X, log_noise, cat_dims=list(self.cat_dims))
            fit_gpytorch_mll(noise_model.make_mll())
            with torch.no_grad():
                predicted_noise = noise_model.posterior(train_X).mean.exp()
                predicted_noise = predicted_noise.clamp_min(self.noise_floor)
            self.likelihood.noise_covar.noise = predicted_noise.squeeze(-1)
            self.noise_model = noise_model
            self._noise_model_fitted = True
        return self

    def noise_posterior(self, X: Tensor):
        """Return the mixed latent posterior for log observation variance."""
        if self.noise_model is None or not self._noise_model_fitted:
            raise RuntimeError("fit_heteroskedastic must be called before noise_posterior.")
        self.noise_model.prediction_strategy = None
        return self.noise_model.posterior(X)

    def predicted_noise(self, X: Tensor) -> Tensor:
        """Return mixed input-dependent observation variance on the original scale."""
        return self.noise_posterior(X).mean.exp().clamp_min(self.noise_floor)
