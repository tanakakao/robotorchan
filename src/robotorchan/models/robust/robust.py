"""Robust GP wrappers with robotorchan model conventions."""

from __future__ import annotations

import torch
from botorch.models.robust_relevance_pursuit_model import RobustRelevancePursuitMixin
from botorch.models.robust_relevance_pursuit_model import (
    RobustRelevancePursuitSingleTaskGP as BoTorchRobustRelevancePursuitSingleTaskGP,
)
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.utils.types import DEFAULT, _DefaultType
from gpytorch.likelihoods import (
    FixedNoiseGaussianLikelihood,
    Likelihood,
)
from gpytorch.means import Mean
from gpytorch.module import Module
from gpytorch.priors import Prior
from torch import Tensor

from robotorchan.models.base import (
    ContinuousKernelFactory,
    ExactGPModelMixin,
    make_mixed_covar_module,
    normalize_feature_dims,
)
from robotorchan.models.standard.multi_fidelity import SingleTaskMultiFidelityGP
from robotorchan.models.standard.multitask import KroneckerMultiTaskGP, MultiTaskGP


class RobustRelevancePursuitMultiTaskGP(MultiTaskGP, RobustRelevancePursuitMixin):
    """Long-format multi-task GP with sparse outlier relevance pursuit."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        task_feature: int,
        train_Yvar: Tensor | None = None,
        mean_module: Module | None = None,
        covar_module: Module | None = None,
        likelihood: Likelihood | None = None,
        task_covar_prior: Prior | _DefaultType | None = DEFAULT,
        output_tasks: list[int] | None = None,
        rank: int | None = None,
        all_tasks: list[int] | None = None,
        outcome_transform: OutcomeTransform | _DefaultType | None = DEFAULT,
        input_transform: InputTransform | None = None,
        validate_task_values: bool = True,
        convex_parameterization: bool = True,
        prior_mean_of_support: float | None = None,
        cache_model_trace: bool = False,
    ) -> None:
        """Initialize the multi-task GP and wrap its base noise with sparse outlier noise."""
        self._original_X = train_X
        self._original_Y = train_Y
        self._robust_task_feature = task_feature
        self._robust_output_tasks = output_tasks
        self._robust_rank = rank
        self._robust_all_tasks = all_tasks
        self._robust_validate_task_values = validate_task_values

        MultiTaskGP.__init__(
            self,
            train_X=train_X,
            train_Y=train_Y,
            task_feature=task_feature,
            train_Yvar=train_Yvar,
            mean_module=mean_module,
            covar_module=covar_module,
            likelihood=likelihood,
            task_covar_prior=task_covar_prior,
            output_tasks=output_tasks,
            rank=rank,
            all_tasks=all_tasks,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
            validate_task_values=validate_task_values,
        )
        RobustRelevancePursuitMixin.__init__(
            self,
            base_likelihood=self.likelihood,
            dim=train_X.shape[-2],
            prior_mean_of_support=prior_mean_of_support,
            convex_parameterization=convex_parameterization,
            cache_model_trace=cache_model_trace,
        )

    def to_standard_model(self) -> MultiTaskGP:
        """Return the equivalent non-dispatching MultiTaskGP for numerical fitting."""
        is_training = self.training
        model = MultiTaskGP(
            train_X=self._original_X,
            train_Y=self._original_Y,
            task_feature=self._robust_task_feature,
            likelihood=self.likelihood,
            mean_module=self.mean_module,
            covar_module=self.covar_module,
            output_tasks=self._robust_output_tasks,
            rank=self._robust_rank,
            all_tasks=self._robust_all_tasks,
            outcome_transform=getattr(self, "outcome_transform", None),
            input_transform=getattr(self, "input_transform", None),
            validate_task_values=self._robust_validate_task_values,
        )
        if not is_training:
            model.eval()
        return model


class MixedRobustRelevancePursuitMultiTaskGP(RobustRelevancePursuitMultiTaskGP):
    """Robust relevance-pursuit MultiTaskGP for mixed data features."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        task_feature: int,
        cat_dims: list[int],
        train_Yvar: Tensor | None = None,
        mean_module: Module | None = None,
        likelihood: Likelihood | None = None,
        task_covar_prior: Prior | _DefaultType | None = DEFAULT,
        output_tasks: list[int] | None = None,
        rank: int | None = None,
        all_tasks: list[int] | None = None,
        cont_kernel_factory: ContinuousKernelFactory | None = None,
        outcome_transform: OutcomeTransform | _DefaultType | None = DEFAULT,
        input_transform: InputTransform | None = None,
        validate_task_values: bool = True,
        convex_parameterization: bool = True,
        prior_mean_of_support: float | None = None,
        cache_model_trace: bool = False,
    ) -> None:
        """Initialize mixed data covariance while preserving task-feature semantics."""
        input_dim = train_X.shape[-1]
        resolved_task_feature = normalize_feature_dims(
            [task_feature], input_dim, name="task_feature"
        )[0]
        normalized_cat_dims = normalize_feature_dims(
            cat_dims,
            input_dim,
            name="cat_dims",
            excluded_dims=[resolved_task_feature],
        )
        data_covar_module = make_mixed_covar_module(
            input_dim=input_dim,
            cat_dims=normalized_cat_dims,
            excluded_dims=[resolved_task_feature],
            batch_shape=train_X.shape[:-2],
            cont_kernel_factory=cont_kernel_factory,
        )
        data_covar_module.active_dims = torch.arange(input_dim, device=train_X.device)

        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            task_feature=task_feature,
            train_Yvar=train_Yvar,
            mean_module=mean_module,
            covar_module=data_covar_module,
            likelihood=likelihood,
            task_covar_prior=task_covar_prior,
            output_tasks=output_tasks,
            rank=rank,
            all_tasks=all_tasks,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
            validate_task_values=validate_task_values,
            convex_parameterization=convex_parameterization,
            prior_mean_of_support=prior_mean_of_support,
            cache_model_trace=cache_model_trace,
        )
        self.cat_dims = tuple(normalized_cat_dims)


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

        from robotorchan.models.standard.single_task import SingleTaskGP

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


class HeteroskedasticMultiFidelityGP(SingleTaskMultiFidelityGP):
    """Iterative multi-fidelity GP with fidelity-dependent observation noise."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        iteration_fidelity: int | None = None,
        data_fidelities: list[int] | None = None,
        noise_floor: float = 1e-6,
        outcome_transform: OutcomeTransform | _DefaultType | None = DEFAULT,
        input_transform: InputTransform | None = None,
    ) -> None:
        if noise_floor <= 0:
            raise ValueError("noise_floor must be positive.")
        self.noise_floor = noise_floor
        self._noise_model_fitted = False
        self._iteration_fidelity = iteration_fidelity
        self._data_fidelities = data_fidelities
        initial_noise = torch.full_like(train_Y, noise_floor)
        likelihood = FixedNoiseGaussianLikelihood(
            noise=initial_noise.squeeze(-1),
            learn_additional_noise=False,
        )
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=initial_noise,
            iteration_fidelity=iteration_fidelity,
            data_fidelities=data_fidelities,
            likelihood=likelihood,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
        )
        self.noise_model = None

    def fit_heteroskedastic(self, *, iterations: int = 3) -> HeteroskedasticMultiFidelityGP:
        """Alternately fit response and multi-fidelity log-noise GPs."""
        from botorch.fit import fit_gpytorch_mll

        if iterations < 1:
            raise ValueError("iterations must be at least 1.")
        train_X = self.raw_train_X
        train_Y = self.raw_train_Y
        for _ in range(iterations):
            fit_gpytorch_mll(self.make_mll())
            with torch.no_grad():
                residual = train_Y - self.posterior(train_X).mean
                log_noise = torch.log(residual.square().clamp_min(self.noise_floor))
            noise_model = SingleTaskMultiFidelityGP(
                train_X=train_X,
                train_Y=log_noise,
                iteration_fidelity=self._iteration_fidelity,
                data_fidelities=self._data_fidelities,
            )
            fit_gpytorch_mll(noise_model.make_mll())
            with torch.no_grad():
                predicted_noise = noise_model.posterior(train_X).mean.exp()
                predicted_noise = predicted_noise.clamp_min(self.noise_floor)
            self.likelihood.noise_covar.noise = predicted_noise.squeeze(-1)
            self.noise_model = noise_model
            self._noise_model_fitted = True
        return self

    def noise_posterior(self, X: Tensor):
        """Return the latent multi-fidelity posterior for log observation variance."""
        if self.noise_model is None or not self._noise_model_fitted:
            raise RuntimeError("fit_heteroskedastic must be called before noise_posterior.")
        self.noise_model.prediction_strategy = None
        return self.noise_model.posterior(X)

    def predicted_noise(self, X: Tensor) -> Tensor:
        """Return fidelity-dependent observation variance on the original scale."""
        return self.noise_posterior(X).mean.exp().clamp_min(self.noise_floor)


class HeteroskedasticKroneckerMultiTaskGP(KroneckerMultiTaskGP):
    """Prototype block-design GP with task-specific input-dependent noise.

    The latent response covariance remains Kronecker structured. A second
    block-design Kronecker GP models task-specific log observation variance.
    The prototype intentionally does not advertise public observation-posterior
    semantics until BoTorch exposes a compatible input-dependent multitask
    observation-noise hook.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        train_Yvar: Tensor | None = None,
        noise_floor: float = 1e-6,
        rank: int | None = None,
    ) -> None:
        if noise_floor <= 0:
            raise ValueError("noise_floor must be positive.")
        if train_X.ndim < 2 or train_Y.ndim < 2:
            raise ValueError("Expected train_X (..., n, d) and train_Y (..., n, m).")
        if train_X.shape[-2] != train_Y.shape[-2]:
            raise ValueError("train_X and train_Y must share the observation dimension.")
        if train_Yvar is None:
            initial_noise = torch.full_like(train_Y, noise_floor)
        else:
            if train_Yvar.shape != train_Y.shape:
                raise ValueError("train_Yvar must have exactly the same shape as train_Y.")
            if not torch.isfinite(train_Yvar).all() or torch.any(train_Yvar < 0):
                raise ValueError("train_Yvar must contain finite non-negative variances.")
            initial_noise = train_Yvar.clamp_min(noise_floor)
        super().__init__(train_X=train_X, train_Y=train_Y, rank=rank)
        self.noise_floor = float(noise_floor)
        self.raw_train_Yvar = initial_noise.detach().clone()
        self.noise_model: KroneckerMultiTaskGP | None = None
        self._noise_model_fitted = False

    def build_noise_model(self, log_noise: Tensor) -> KroneckerMultiTaskGP:
        """Build the task-specific block-design log-noise surrogate."""
        if log_noise.shape != self.raw_train_Y.shape:
            raise ValueError("log_noise must have exactly the same shape as train_Y.")
        return KroneckerMultiTaskGP(self.raw_train_X, log_noise)

    def set_noise_model(self, noise_model: KroneckerMultiTaskGP) -> None:
        """Attach a fitted block-design noise surrogate for prototype evaluation."""
        if noise_model.raw_train_Y.shape != self.raw_train_Y.shape:
            raise ValueError("noise model outputs must match the response task axis.")
        self.noise_model = noise_model
        self._noise_model_fitted = True

    def noise_posterior(self, X: Tensor):
        """Return the task-specific log-variance posterior."""
        if self.noise_model is None or not self._noise_model_fitted:
            raise RuntimeError("A fitted noise model is required before noise_posterior.")
        return self.noise_model.posterior(X)

    def predicted_noise(self, X: Tensor) -> Tensor:
        """Return task-specific observation variance with shape (..., q, m)."""
        return self.noise_posterior(X).mean.exp().clamp_min(self.noise_floor)

    def observation_covariance_diagonal(self, X: Tensor) -> Tensor:
        """Return the explicit task-specific diagonal observation-noise term."""
        noise = self.predicted_noise(X)
        return noise.reshape(*noise.shape[:-2], -1)


class HeteroskedasticMultiTaskGP(MultiTaskGP):
    """Iterative long-format multi-task GP for input-dependent observation noise."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        task_feature: int,
        *,
        noise_floor: float = 1e-6,
        task_covar_prior: Prior | _DefaultType | None = DEFAULT,
        output_tasks: list[int] | None = None,
        rank: int | None = None,
        all_tasks: list[int] | None = None,
        outcome_transform: OutcomeTransform | _DefaultType | None = DEFAULT,
        input_transform: InputTransform | None = None,
        validate_task_values: bool = True,
    ) -> None:
        if noise_floor <= 0:
            raise ValueError("noise_floor must be positive.")
        self.noise_floor = noise_floor
        self._noise_model_fitted = False
        self._heteroskedastic_task_feature = task_feature
        initial_noise = torch.full_like(train_Y, noise_floor)
        likelihood = FixedNoiseGaussianLikelihood(
            noise=initial_noise.squeeze(-1),
            learn_additional_noise=False,
        )
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            task_feature=task_feature,
            train_Yvar=initial_noise,
            likelihood=likelihood,
            task_covar_prior=task_covar_prior,
            output_tasks=output_tasks,
            rank=rank,
            all_tasks=all_tasks,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
            validate_task_values=validate_task_values,
        )
        self.noise_model = None

    def fit_heteroskedastic(self, *, iterations: int = 3) -> HeteroskedasticMultiTaskGP:
        """Alternately fit the response GP and a multi-task log-noise GP."""
        from botorch.fit import fit_gpytorch_mll

        if iterations < 1:
            raise ValueError("iterations must be at least 1.")
        train_X = self.raw_train_X
        train_Y = self.raw_train_Y
        for _ in range(iterations):
            fit_gpytorch_mll(self.make_mll())
            with torch.no_grad():
                latent = self(train_X).mean.unsqueeze(-1)
                residual = train_Y - latent
                log_noise = torch.log(residual.square().clamp_min(self.noise_floor))
            noise_model = MultiTaskGP(
                train_X,
                log_noise,
                task_feature=self._heteroskedastic_task_feature,
            )
            fit_gpytorch_mll(noise_model.make_mll())
            with torch.no_grad():
                predicted_noise = noise_model(train_X).mean.unsqueeze(-1).exp()
                predicted_noise = predicted_noise.clamp_min(self.noise_floor)
            self.likelihood.noise_covar.noise = predicted_noise.squeeze(-1)
            self.noise_model = noise_model
            self._noise_model_fitted = True
        return self

    def noise_posterior(self, X: Tensor):
        """Return the latent posterior for long-format log observation variance."""
        if self.noise_model is None or not self._noise_model_fitted:
            raise RuntimeError("fit_heteroskedastic must be called before noise_posterior.")
        self.noise_model.prediction_strategy = None
        return self.noise_model(X)

    def predicted_noise(self, X: Tensor) -> Tensor:
        """Return long-format observation variance on the original scale."""
        return self.noise_posterior(X).mean.unsqueeze(-1).exp().clamp_min(self.noise_floor)


class MixedHeteroskedasticMultiTaskGP(HeteroskedasticMultiTaskGP):
    """Heteroskedastic long-format multi-task GP for mixed data features."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        task_feature: int,
        *,
        cat_dims: list[int],
        noise_floor: float = 1e-6,
        cont_kernel_factory: ContinuousKernelFactory | None = None,
        task_covar_prior: Prior | _DefaultType | None = DEFAULT,
        output_tasks: list[int] | None = None,
        rank: int | None = None,
        all_tasks: list[int] | None = None,
        outcome_transform: OutcomeTransform | _DefaultType | None = DEFAULT,
        input_transform: InputTransform | None = None,
        validate_task_values: bool = True,
    ) -> None:
        input_dim = train_X.shape[-1]
        resolved_task_feature = normalize_feature_dims(
            [task_feature], input_dim, name="task_feature"
        )[0]
        normalized_cat_dims = normalize_feature_dims(
            cat_dims, input_dim, name="cat_dims", excluded_dims=[resolved_task_feature]
        )
        self.cat_dims = tuple(normalized_cat_dims)
        self._cont_kernel_factory = cont_kernel_factory
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            task_feature=task_feature,
            noise_floor=noise_floor,
            task_covar_prior=task_covar_prior,
            output_tasks=output_tasks,
            rank=rank,
            all_tasks=all_tasks,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
            validate_task_values=validate_task_values,
        )
        data_covar_module = make_mixed_covar_module(
            input_dim=input_dim,
            cat_dims=normalized_cat_dims,
            excluded_dims=[resolved_task_feature],
            batch_shape=train_X.shape[:-2],
            cont_kernel_factory=cont_kernel_factory,
        )
        data_covar_module.active_dims = torch.arange(input_dim, device=train_X.device)
        self.covar_module.kernels[0] = data_covar_module

    def fit_heteroskedastic(self, *, iterations: int = 3) -> MixedHeteroskedasticMultiTaskGP:
        """Alternately fit mixed response and mixed multi-task log-noise GPs."""
        from botorch.fit import fit_gpytorch_mll

        from robotorchan.models.standard.multitask import MixedMultiTaskGP

        if iterations < 1:
            raise ValueError("iterations must be at least 1.")
        train_X = self.raw_train_X
        train_Y = self.raw_train_Y
        for _ in range(iterations):
            fit_gpytorch_mll(self.make_mll())
            with torch.no_grad():
                latent = self(train_X).mean.unsqueeze(-1)
                residual = train_Y - latent
                log_noise = torch.log(residual.square().clamp_min(self.noise_floor))
            noise_model = MixedMultiTaskGP(
                train_X,
                log_noise,
                task_feature=self._heteroskedastic_task_feature,
                cat_dims=list(self.cat_dims),
                cont_kernel_factory=self._cont_kernel_factory,
            )
            fit_gpytorch_mll(noise_model.make_mll())
            with torch.no_grad():
                predicted_noise = noise_model(train_X).mean.unsqueeze(-1).exp()
                predicted_noise = predicted_noise.clamp_min(self.noise_floor)
            self.likelihood.noise_covar.noise = predicted_noise.squeeze(-1)
            self.noise_model = noise_model
            self._noise_model_fitted = True
        return self


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

        from robotorchan.models.standard.single_task import MixedSingleTaskGP

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
