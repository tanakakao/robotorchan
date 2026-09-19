"""Joint variational heteroskedastic Gaussian-process surrogate."""

from __future__ import annotations

import math

import torch
from botorch.posteriors.gpytorch import GPyTorchPosterior
from torch import Tensor, nn

from robotorchan.models.base import RawDataMixin
from robotorchan.models.variational import SingleTaskVariationalGP


class JointHeteroskedasticSingleTaskGP(RawDataMixin, nn.Module):
    """Two-latent variational GP for input-dependent observation variance."""

    supports_mll = False

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_inducing: int = 32,
        noise_floor: float = 1e-6,
        num_mc_samples: int = 16,
        beta_response: float = 1.0,
        beta_noise: float = 1.0,
    ) -> None:
        super().__init__()
        if train_Y.shape[-1] != 1:
            raise ValueError("train_Y must have a single output.")
        if noise_floor <= 0:
            raise ValueError("noise_floor must be positive.")
        if num_inducing < 1 or num_mc_samples < 1:
            raise ValueError("num_inducing and num_mc_samples must be positive.")
        if beta_response <= 0 or beta_noise <= 0:
            raise ValueError("beta_response and beta_noise must be positive.")

        self.noise_floor = float(noise_floor)
        self.num_mc_samples = int(num_mc_samples)
        self.beta_response = float(beta_response)
        self.beta_noise = float(beta_noise)
        inducing = min(int(num_inducing), train_X.shape[-2])

        self.response_model = SingleTaskVariationalGP(
            train_X,
            train_Y,
            inducing_points=inducing,
        )
        initial_log_noise = torch.full_like(train_Y, math.log(self.noise_floor))
        self.noise_model = SingleTaskVariationalGP(
            train_X,
            initial_log_noise,
            inducing_points=inducing,
        )
        self._store_raw_tensor("train_X", train_X.detach().clone())
        self._store_raw_tensor("train_Y", train_Y.detach().clone())
        self._store_raw_tensor("train_Yvar", None)

    @property
    def num_outputs(self) -> int:
        """Number of response outputs."""
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

    def make_mll(self):
        """Reject exact/standard variational MLL construction for the joint objective."""
        raise RuntimeError("Use training_loss() for joint heteroskedastic inference.")

    def posterior(self, X: Tensor, **kwargs) -> GPyTorchPosterior:
        """Return the response-process posterior for BoTorch acquisitions."""
        self.response_model.model.variational_strategy._clear_cache()
        return self.response_model.posterior(X, **kwargs)

    def noise_posterior(self, X: Tensor) -> GPyTorchPosterior:
        """Return the latent posterior over log observation variance."""
        self.noise_model.model.variational_strategy._clear_cache()
        return self.noise_model.posterior(X)

    def predicted_noise(self, X: Tensor) -> Tensor:
        """Return posterior-mean observation variance."""
        posterior = self.noise_posterior(X)
        mean = posterior.mean
        variance = posterior.variance
        return torch.exp(mean + 0.5 * variance).clamp_min(self.noise_floor)

    def training_loss(self, *, num_mc_samples: int | None = None) -> Tensor:
        """Return a Monte Carlo negative ELBO for both latent processes."""
        samples = self.num_mc_samples if num_mc_samples is None else int(num_mc_samples)
        if samples < 1:
            raise ValueError("num_mc_samples must be positive.")

        X = self.raw_train_X
        Y = self.raw_train_Y.squeeze(-1)
        response_dist = self.response_model.model(X)
        noise_dist = self.noise_model.model(X)
        response_samples = response_dist.rsample(torch.Size([samples]))
        log_noise_samples = noise_dist.rsample(torch.Size([samples]))
        variance = log_noise_samples.exp().clamp_min(self.noise_floor)
        log_likelihood = -0.5 * (
            math.log(2 * math.pi) + log_noise_samples + (Y - response_samples).square() / variance
        )
        expected_log_likelihood = log_likelihood.mean(0).sum()

        response_kl = self.response_model.model.variational_strategy.kl_divergence().sum()
        noise_kl = self.noise_model.model.variational_strategy.kl_divergence().sum()
        return (
            -expected_log_likelihood
            + self.beta_response * response_kl
            + self.beta_noise * noise_kl
        ) / X.shape[-2]
