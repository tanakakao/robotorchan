"""Mahalanobis GP components for ALEBO."""

from __future__ import annotations

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models.model import Model
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import Kernel, ScaleKernel
from gpytorch.module import Module
from torch import Tensor

from robotorchan.models.single_task import SingleTaskGP


class MahalanobisRBFKernel(Kernel):
    """RBF kernel with a learned full positive-definite distance metric."""

    has_lengthscale = False

    def __init__(self, ard_num_dims: int, **kwargs: object) -> None:
        super().__init__(**kwargs)
        if ard_num_dims < 1:
            raise ValueError("ard_num_dims must be positive.")
        self.ard_num_dims = ard_num_dims
        n_free = ard_num_dims * (ard_num_dims + 1) // 2
        self.register_parameter(
            name="raw_tril",
            parameter=torch.nn.Parameter(torch.zeros(n_free)),
        )
        self.register_buffer(
            "tril_rows", torch.tril_indices(ard_num_dims, ard_num_dims)[0], persistent=False
        )
        self.register_buffer(
            "tril_cols", torch.tril_indices(ard_num_dims, ard_num_dims)[1], persistent=False
        )

    @property
    def metric_factor(self) -> Tensor:
        """Lower-triangular factor whose Gram matrix is the distance metric."""
        factor = self.raw_tril.new_zeros(self.ard_num_dims, self.ard_num_dims)
        factor[self.tril_rows, self.tril_cols] = self.raw_tril
        diagonal = torch.diagonal(factor)
        positive_diagonal = torch.nn.functional.softplus(diagonal) + 1e-6
        factor = factor - torch.diag_embed(diagonal) + torch.diag_embed(positive_diagonal)
        return factor

    @property
    def metric(self) -> Tensor:
        """Symmetric positive-definite Mahalanobis metric."""
        factor = self.metric_factor
        return factor @ factor.transpose(-2, -1)

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        **params: object,
    ) -> Tensor:
        """Evaluate exp(-0.5 * Mahalanobis squared distance)."""
        del params
        transformed_x1 = x1 @ self.metric_factor
        transformed_x2 = x2 @ self.metric_factor
        if diag:
            squared_distance = (transformed_x1 - transformed_x2).square().sum(dim=-1)
        else:
            difference = transformed_x1.unsqueeze(-2) - transformed_x2.unsqueeze(-3)
            squared_distance = difference.square().sum(dim=-1)
        return torch.exp(-0.5 * squared_distance)


class ALEBOMetricMarginalModel(Model):
    """BoTorch model surface backed by ALEBO metric-marginal predictions."""

    def __init__(
        self,
        base_model: ALEBOGP,
        *,
        metric_samples: Tensor,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.register_buffer("metric_samples", metric_samples.detach().clone())

    @property
    def num_outputs(self) -> int:
        """Return the number of modeled outputs."""
        return 1

    def posterior(
        self,
        X: Tensor,
        output_indices: list[int] | None = None,
        observation_noise: bool | Tensor = False,
        posterior_transform: object | None = None,
        **kwargs: object,
    ) -> GPyTorchPosterior:
        """Return the metric-marginal Gaussian posterior."""
        del kwargs
        if output_indices not in (None, [0]):
            raise NotImplementedError("ALEBO metric marginalization supports one output.")
        if posterior_transform is not None:
            raise NotImplementedError("posterior_transform is not supported yet.")
        if not isinstance(observation_noise, bool):
            raise NotImplementedError("Tensor observation_noise is not supported yet.")
        return self.base_model.metric_marginal_posterior_from_samples(
            X,
            metric_samples=self.metric_samples,
            observation_noise=observation_noise,
        )


class ALEBOGP(SingleTaskGP):
    """Single-task GP using ALEBO's full Mahalanobis RBF geometry."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        *,
        covar_module: Module | None = None,
    ) -> None:
        if train_X.ndim < 2:
            raise ValueError("train_X must have at least two dimensions.")
        if covar_module is None:
            covar_module = ScaleKernel(MahalanobisRBFKernel(ard_num_dims=train_X.shape[-1]))
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            covar_module=covar_module,
        )

    @property
    def mahalanobis_kernel(self) -> MahalanobisRBFKernel:
        """Return the ALEBO Mahalanobis base kernel."""
        if not isinstance(self.covar_module, ScaleKernel) or not isinstance(
            self.covar_module.base_kernel, MahalanobisRBFKernel
        ):
            raise TypeError("ALEBOGP requires a ScaleKernel wrapping MahalanobisRBFKernel.")
        return self.covar_module.base_kernel

    @property
    def metric(self) -> Tensor:
        """Return the current learned Mahalanobis metric."""
        return self.mahalanobis_kernel.metric

    def metric_parameter_vector(self) -> Tensor:
        """Return the unconstrained Mahalanobis parameters as a flat vector."""
        return self.mahalanobis_kernel.raw_tril.detach().clone()

    def metric_log_posterior(self) -> Tensor:
        """Return the exact MLL used as the metric log-posterior objective."""
        mll = self.make_mll()
        output = self(*self.train_inputs)
        target = self.train_targets
        value = mll(output, target)
        return value.sum() if value.ndim else value

    def metric_diagonal_hessian(self) -> Tensor:
        """Evaluate the diagonal Hessian with respect to metric parameters."""
        parameter = self.mahalanobis_kernel.raw_tril
        objective = self.metric_log_posterior()
        gradient = torch.autograd.grad(
            objective,
            parameter,
            create_graph=True,
        )[0]
        diagonal = []
        for index in range(parameter.numel()):
            second = torch.autograd.grad(
                gradient[index],
                parameter,
                retain_graph=index + 1 < parameter.numel(),
            )[0]
            diagonal.append(second[index])
        return torch.stack(diagonal).detach()

    def estimate_metric_laplace_covariance(
        self,
        *,
        min_curvature: float = 1e-8,
    ) -> Tensor:
        """Estimate the diagonal Laplace covariance at the current GP state."""
        return self.metric_laplace_covariance(
            diagonal_hessian=self.metric_diagonal_hessian(),
            min_curvature=min_curvature,
        )

    def metric_laplace_covariance(
        self,
        *,
        diagonal_hessian: Tensor,
        min_curvature: float = 1e-8,
    ) -> Tensor:
        """Construct the diagonal Laplace covariance for metric parameters."""
        mean = self.metric_parameter_vector()
        if diagonal_hessian.shape != mean.shape:
            raise ValueError(f"diagonal_hessian must have shape {tuple(mean.shape)}.")
        if min_curvature <= 0:
            raise ValueError("min_curvature must be positive.")
        curvature = -diagonal_hessian.to(dtype=mean.dtype, device=mean.device)
        if bool((curvature <= 0).any()):
            raise ValueError("diagonal_hessian must be strictly negative at the posterior mode.")
        variance = curvature.clamp_min(min_curvature).reciprocal()
        return torch.diag(variance)

    def sample_metric_parameters(
        self,
        n_samples: int,
        *,
        covariance: Tensor,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Sample metric parameters from a Gaussian Laplace approximation."""
        if n_samples < 1:
            raise ValueError("n_samples must be positive.")
        mean = self.metric_parameter_vector()
        expected_shape = (mean.numel(), mean.numel())
        if covariance.shape != expected_shape:
            raise ValueError(f"covariance must have shape {expected_shape}.")
        covariance = covariance.to(dtype=mean.dtype, device=mean.device)
        chol = torch.linalg.cholesky(covariance)
        noise = torch.randn(
            n_samples,
            mean.numel(),
            dtype=mean.dtype,
            device=mean.device,
            generator=generator,
        )
        return mean.unsqueeze(0) + noise @ chol.transpose(-2, -1)

    def metric_sample_predictions(
        self,
        X: Tensor,
        *,
        metric_samples: Tensor,
        observation_noise: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Evaluate conditional GP moments for sampled metric parameters."""
        parameter = self.mahalanobis_kernel.raw_tril
        expected_shape = (parameter.numel(),)
        if metric_samples.ndim != 2 or metric_samples.shape[1:] != expected_shape:
            raise ValueError(f"metric_samples must have shape [n_samples, {parameter.numel()}].")
        if metric_samples.shape[0] < 1:
            raise ValueError("metric_samples must contain at least one sample.")

        original = parameter.detach().clone()
        means = []
        covariances = []
        try:
            for sample in metric_samples:
                with torch.no_grad():
                    parameter.copy_(sample.to(dtype=parameter.dtype, device=parameter.device))
                posterior = super().posterior(X, observation_noise=observation_noise)
                means.append(posterior.mean)
                covariances.append(posterior.distribution.covariance_matrix)
        finally:
            with torch.no_grad():
                parameter.copy_(original)
        return torch.stack(means), torch.stack(covariances)

    def marginal_metric_moments(
        self,
        X: Tensor,
        *,
        n_metric_samples: int,
        covariance: Tensor | None = None,
        observation_noise: bool = False,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Moment-match predictive mean and full covariance over metric samples."""
        if covariance is None:
            covariance = self.estimate_metric_laplace_covariance()
        samples = self.sample_metric_parameters(
            n_metric_samples,
            covariance=covariance,
            generator=generator,
        )
        means, covariances = self.metric_sample_predictions(
            X,
            metric_samples=samples,
            observation_noise=observation_noise,
        )
        if means.shape[-1] != 1:
            raise NotImplementedError("ALEBO metric marginalization currently supports one output.")
        event_means = means.squeeze(-1)
        mean = event_means.mean(dim=0)
        centered = event_means - mean
        between_metric = torch.einsum("...i,...j->...ij", centered, centered).mean(dim=0)
        predictive_covariance = covariances.mean(dim=0) + between_metric
        predictive_covariance = 0.5 * (
            predictive_covariance + predictive_covariance.transpose(-2, -1)
        )
        eigenvalues = torch.linalg.eigvalsh(predictive_covariance)
        scale = torch.diagonal(predictive_covariance, dim1=-2, dim2=-1).abs().amax(dim=-1)
        floor = torch.finfo(predictive_covariance.dtype).eps * scale.clamp_min(1.0) * 100
        correction = (floor - eigenvalues[..., 0]).clamp_min(0.0)
        identity = torch.eye(
            predictive_covariance.shape[-1],
            dtype=predictive_covariance.dtype,
            device=predictive_covariance.device,
        )
        predictive_covariance = predictive_covariance + correction[..., None, None] * identity
        return mean.unsqueeze(-1), predictive_covariance

    def metric_marginal_posterior_from_samples(
        self,
        X: Tensor,
        *,
        metric_samples: Tensor,
        observation_noise: bool = False,
    ) -> GPyTorchPosterior:
        """Return a moment-matched posterior for fixed metric samples."""
        means, covariances = self.metric_sample_predictions(
            X,
            metric_samples=metric_samples,
            observation_noise=observation_noise,
        )
        mean, predictive_covariance = self._moment_match_metric_covariance(means, covariances)
        distribution = MultivariateNormal(mean.squeeze(-1), predictive_covariance)
        return GPyTorchPosterior(distribution)

    def marginal_metric_posterior(
        self,
        X: Tensor,
        *,
        n_metric_samples: int,
        covariance: Tensor | None = None,
        observation_noise: bool = False,
        generator: torch.Generator | None = None,
    ) -> GPyTorchPosterior:
        """Return a BoTorch Gaussian posterior marginalized over metric uncertainty."""
        if covariance is None:
            covariance = self.estimate_metric_laplace_covariance()
        samples = self.sample_metric_parameters(
            n_metric_samples,
            covariance=covariance,
            generator=generator,
        )
        return self.metric_marginal_posterior_from_samples(
            X,
            metric_samples=samples,
            observation_noise=observation_noise,
        )

    def posterior_with_metric_uncertainty(
        self,
        X: Tensor,
        *,
        n_metric_samples: int,
        covariance: Tensor | None = None,
        observation_noise: bool = False,
        generator: torch.Generator | None = None,
    ) -> GPyTorchPosterior:
        """Return the ALEBO posterior used by acquisition functions."""
        return self.marginal_metric_posterior(
            X,
            n_metric_samples=n_metric_samples,
            covariance=covariance,
            observation_noise=observation_noise,
            generator=generator,
        )

    def acquisition_model(
        self,
        *,
        n_metric_samples: int,
        covariance: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> ALEBOMetricMarginalModel:
        """Return a BoTorch model whose posterior includes metric uncertainty."""
        if covariance is None:
            covariance = self.estimate_metric_laplace_covariance()
        metric_samples = self.sample_metric_parameters(
            n_metric_samples,
            covariance=covariance,
            generator=generator,
        )
        return ALEBOMetricMarginalModel(self, metric_samples=metric_samples)

    @staticmethod
    def moment_match_predictions(
        means: Tensor,
        variances: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Moment-match Gaussian predictions over metric posterior samples."""
        if means.shape != variances.shape:
            raise ValueError("means and variances must have the same shape.")
        if means.ndim < 1 or means.shape[0] < 1:
            raise ValueError("predictions must contain at least one metric sample.")
        if bool((variances < 0).any()):
            raise ValueError("variances must be non-negative.")

        mean = means.mean(dim=0)
        second_moment = (variances + means.square()).mean(dim=0)
        variance = (second_moment - mean.square()).clamp_min(0.0)
        return mean, variance

    def fit(self, **fit_kwargs: object) -> ALEBOGP:
        """Fit ALEBO GP hyperparameters by maximizing the exact marginal likelihood."""
        fit_gpytorch_mll(self.make_mll(), **fit_kwargs)
        return self
