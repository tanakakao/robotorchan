"""Variational GP with an explicit Gaussian contamination likelihood."""

from __future__ import annotations

import math

import torch
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.utils.quadrature import GaussHermiteQuadrature1D
from torch import Tensor, nn

from robotorchan.models.base import RawDataMixin
from robotorchan.models.robust.student_t import (
    _multitask_covar_module,
    _RobustObservationFitMixin,
)
from robotorchan.models.standard.variational import (
    MixedSingleTaskVariationalGP,
    SingleTaskVariationalGP,
)


class ContaminatedSingleTaskGP(_RobustObservationFitMixin, RawDataMixin, nn.Module):
    """Scalar variational GP with fixed nominal/contamination noise components."""

    supports_mll = False

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        contamination_probability: float = 0.05,
        inlier_scale: float = 0.05,
        outlier_scale: float = 0.5,
        num_inducing: int = 32,
        num_likelihood_samples: int = 16,
        beta: float = 1.0,
    ) -> None:
        super().__init__()
        if train_Y.shape[-1] != 1:
            raise ValueError("train_Y must have a single output.")
        if not 0 < contamination_probability < 1:
            raise ValueError("contamination_probability must be strictly between 0 and 1.")
        if inlier_scale <= 0 or outlier_scale <= inlier_scale:
            raise ValueError("Scales must satisfy 0 < inlier_scale < outlier_scale.")
        if num_inducing < 1 or num_likelihood_samples < 1:
            raise ValueError("num_inducing and num_likelihood_samples must be positive.")
        if beta <= 0:
            raise ValueError("beta must be positive.")

        self.contamination_probability = float(contamination_probability)
        self.inlier_scale = float(inlier_scale)
        self.outlier_scale = float(outlier_scale)
        self.num_likelihood_samples = int(num_likelihood_samples)
        self.beta = float(beta)
        inducing = min(int(num_inducing), train_X.shape[-2])
        self.response_model = SingleTaskVariationalGP(
            train_X,
            train_Y,
            inducing_points=inducing,
        )
        self._store_raw_tensor("train_X", train_X.detach().clone())
        self._store_raw_tensor("train_Y", train_Y.detach().clone())
        self._store_raw_tensor("train_Yvar", None)

    @property
    def num_outputs(self) -> int:
        """Number of latent response outputs."""
        return 1

    @property
    def raw_train_X(self) -> Tensor:
        """Caller-supplied training inputs."""
        value = self._get_raw_tensor("train_X")
        if value is None:
            raise RuntimeError("raw_train_X was unexpectedly stored as None.")
        return value

    @property
    def raw_train_Y(self) -> Tensor:
        """Caller-supplied training outcomes."""
        value = self._get_raw_tensor("train_Y")
        if value is None:
            raise RuntimeError("raw_train_Y was unexpectedly stored as None.")
        return value

    @property
    def raw_train_Yvar(self) -> None:
        """No externally supplied observation variance is required."""
        return None

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Load state and clear derived variational caches."""
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self.response_model.model.variational_strategy._clear_cache()
        return result

    def make_mll(self):
        """Reject MLL construction in favor of the explicit mixture loss."""
        raise RuntimeError("Use training_loss() for contamination-mixture inference.")

    def posterior(self, X: Tensor, **kwargs) -> GPyTorchPosterior:
        """Return the latent response posterior for BoTorch acquisitions."""
        self.response_model.model.variational_strategy._clear_cache()
        return self.response_model.posterior(X, **kwargs)

    def _component_log_prob(self, residual: Tensor, scale: float) -> Tensor:
        scale_tensor = residual.new_tensor(scale)
        return (
            -0.5 * math.log(2 * math.pi)
            - torch.log(scale_tensor)
            - 0.5 * (residual / scale_tensor).square()
        )

    def _mixture_log_prob(self, residual: Tensor) -> Tensor:
        inlier = self._component_log_prob(residual, self.inlier_scale)
        outlier = self._component_log_prob(residual, self.outlier_scale)
        log_inlier_weight = residual.new_tensor(math.log1p(-self.contamination_probability))
        log_outlier_weight = residual.new_tensor(math.log(self.contamination_probability))
        components = torch.stack(
            [log_inlier_weight + inlier, log_outlier_weight + outlier],
            dim=0,
        )
        return torch.logsumexp(components, dim=0)

    def training_loss(self, *, num_likelihood_samples: int | None = None) -> Tensor:
        """Return negative ELBO using marginal Gaussian quadrature by default.

        Explicit num_likelihood_samples retains the stochastic Monte Carlo
        estimator for comparison. The likelihood factorizes over observations,
        so only the one-dimensional Gaussian marginals are needed.
        """
        if num_likelihood_samples is not None and num_likelihood_samples < 1:
            raise ValueError("num_likelihood_samples must be positive.")

        X = self.raw_train_X
        Y = self.raw_train_Y.squeeze(-1)
        latent = self.response_model.model(X)
        if num_likelihood_samples is None:
            quadrature = GaussHermiteQuadrature1D(
                num_locs=max(32, self.num_likelihood_samples)
            ).to(X)
            expected_log_likelihood = quadrature(
                lambda values: self._mixture_log_prob(Y - values), latent
            ).sum()
        else:
            latent_samples = latent.rsample(torch.Size([int(num_likelihood_samples)]))
            expected_log_likelihood = self._mixture_log_prob(Y - latent_samples).mean(0).sum()
        kl = self.response_model.model.variational_strategy.kl_divergence().sum()
        return (-expected_log_likelihood + self.beta * kl) / X.shape[-2]

    def fit(
        self, *, steps: int = 100, lr: float = 0.03, warmup_steps: int = 0
    ) -> None:
        """Fit the mixture, optionally warming up with a broader inlier component.

        Warmup temporarily broadens the inlier standard deviation and anneals it
        to the specified value. The final objective and all predictions use the
        original mixture parameters.
        """
        if warmup_steps < 0 or warmup_steps >= steps:
            raise ValueError("warmup_steps must be nonnegative and less than steps.")
        if not 0 < lr < float("inf"):
            raise ValueError("lr must be positive and finite.")
        if warmup_steps == 0:
            super().fit(steps=steps, lr=lr)
            return
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        target_scale = self.inlier_scale
        initial_scale = max(target_scale, min(self.outlier_scale * 0.5, 0.3))
        try:
            for step in range(steps):
                fraction = min(step / warmup_steps, 1.0)
                self.inlier_scale = initial_scale * (target_scale / initial_scale) ** fraction
                optimizer.zero_grad()
                loss = self.training_loss()
                loss.backward()
                optimizer.step()
        finally:
            self.inlier_scale = target_scale
        self.eval()

    def contamination_diagnostic(self, X: Tensor, Y: Tensor) -> Tensor:
        """Approximate contamination probability using the latent posterior mean."""
        if Y.shape[-1] != 1:
            raise ValueError("Y must have a single output.")
        mean = self.posterior(X).mean
        residual = Y - mean
        inlier = self._component_log_prob(residual, self.inlier_scale)
        outlier = self._component_log_prob(residual, self.outlier_scale)
        log_inlier = math.log1p(-self.contamination_probability) + inlier
        log_outlier = math.log(self.contamination_probability) + outlier
        return torch.sigmoid(log_outlier - log_inlier)


class MixedContaminatedSingleTaskGP(ContaminatedSingleTaskGP):
    """Contamination-mixture GP with native mixed continuous/categorical covariance."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        cat_dims: list[int],
        contamination_probability: float = 0.05,
        inlier_scale: float = 0.05,
        outlier_scale: float = 0.5,
        num_inducing: int = 32,
        num_likelihood_samples: int = 16,
        beta: float = 1.0,
    ) -> None:
        nn.Module.__init__(self)
        if train_Y.shape[-1] != 1:
            raise ValueError("train_Y must have a single output.")
        if not 0 < contamination_probability < 1:
            raise ValueError("contamination_probability must be strictly between 0 and 1.")
        if inlier_scale <= 0 or outlier_scale <= inlier_scale:
            raise ValueError("Scales must satisfy 0 < inlier_scale < outlier_scale.")
        if num_inducing < 1 or num_likelihood_samples < 1:
            raise ValueError("num_inducing and num_likelihood_samples must be positive.")
        if beta <= 0:
            raise ValueError("beta must be positive.")
        self.contamination_probability = float(contamination_probability)
        self.inlier_scale = float(inlier_scale)
        self.outlier_scale = float(outlier_scale)
        self.num_likelihood_samples = int(num_likelihood_samples)
        self.beta = float(beta)
        inducing = min(int(num_inducing), train_X.shape[-2])
        self.response_model = MixedSingleTaskVariationalGP(
            train_X,
            train_Y,
            cat_dims=cat_dims,
            inducing_points=inducing,
        )
        self.cat_dims = self.response_model.cat_dims
        self._store_raw_tensor("train_X", train_X)
        self._store_raw_tensor("train_Y", train_Y)
        self._store_raw_tensor("train_Yvar", None)


class ContaminatedMultiTaskGP(ContaminatedSingleTaskGP):
    """Long-format multi-task GP with an explicit Gaussian contamination mixture."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        task_feature: int,
        *,
        contamination_probability: float = 0.05,
        inlier_scale: float = 0.05,
        outlier_scale: float = 0.5,
        num_inducing: int = 32,
        num_likelihood_samples: int = 16,
        beta: float = 1.0,
    ) -> None:
        nn.Module.__init__(self)
        self._initialize_multitask(
            train_X,
            train_Y,
            task_feature,
            cat_dims=None,
            contamination_probability=contamination_probability,
            inlier_scale=inlier_scale,
            outlier_scale=outlier_scale,
            num_inducing=num_inducing,
            num_likelihood_samples=num_likelihood_samples,
            beta=beta,
        )

    def _initialize_multitask(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        task_feature: int,
        *,
        cat_dims: list[int] | None,
        contamination_probability: float,
        inlier_scale: float,
        outlier_scale: float,
        num_inducing: int,
        num_likelihood_samples: int,
        beta: float,
    ) -> None:
        if train_Y.shape[-1] != 1:
            raise ValueError("train_Y must have a single output.")
        if not 0 < contamination_probability < 1:
            raise ValueError("contamination_probability must be strictly between 0 and 1.")
        if inlier_scale <= 0 or outlier_scale <= inlier_scale:
            raise ValueError("Scales must satisfy 0 < inlier_scale < outlier_scale.")
        if num_inducing < 1 or num_likelihood_samples < 1:
            raise ValueError("num_inducing and num_likelihood_samples must be positive.")
        if beta <= 0:
            raise ValueError("beta must be positive.")
        covar_module, task_dim = _multitask_covar_module(train_X, task_feature, cat_dims=cat_dims)
        self.contamination_probability = float(contamination_probability)
        self.inlier_scale = float(inlier_scale)
        self.outlier_scale = float(outlier_scale)
        self.num_likelihood_samples = int(num_likelihood_samples)
        self.beta = float(beta)
        self.task_feature = task_dim
        self.response_model = SingleTaskVariationalGP(
            train_X,
            train_Y,
            covar_module=covar_module,
            inducing_points=min(int(num_inducing), train_X.shape[-2]),
        )
        self._store_raw_tensor("train_X", train_X.detach().clone())
        self._store_raw_tensor("train_Y", train_Y.detach().clone())
        self._store_raw_tensor("train_Yvar", None)


class MixedContaminatedMultiTaskGP(ContaminatedMultiTaskGP):
    """Contamination-mixture multi-task GP for mixed data features."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        task_feature: int,
        *,
        cat_dims: list[int],
        contamination_probability: float = 0.05,
        inlier_scale: float = 0.05,
        outlier_scale: float = 0.5,
        num_inducing: int = 32,
        num_likelihood_samples: int = 16,
        beta: float = 1.0,
    ) -> None:
        nn.Module.__init__(self)
        self._initialize_multitask(
            train_X,
            train_Y,
            task_feature,
            cat_dims=cat_dims,
            contamination_probability=contamination_probability,
            inlier_scale=inlier_scale,
            outlier_scale=outlier_scale,
            num_inducing=num_inducing,
            num_likelihood_samples=num_likelihood_samples,
            beta=beta,
        )
        from robotorchan.models.base import normalize_feature_dims

        self.cat_dims = tuple(
            normalize_feature_dims(
                cat_dims,
                train_X.shape[-1],
                name="cat_dims",
                excluded_dims=[self.task_feature],
            )
        )
