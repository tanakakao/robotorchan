"""Joint variational-autoencoder and Gaussian process models."""

from __future__ import annotations

import torch
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from torch import Tensor, nn

from robotorchan.models.neural_features import make_feature_network
from robotorchan.models.high_dimensional.reduced.joint_neural import JointEncoderGP, MixedJointEncoderGP


class JointVAEGP(JointEncoderGP):
    """Exact GP jointly trained with a variational latent input representation.

    GP predictions use the posterior-mean latent code ``mu(X)`` by default. The
    VAE representation can also be marginalized with
    :meth:`uncertainty_aware_posterior`, which moment-matches GP predictions
    over Monte Carlo samples from ``q(z | X)``. Use :meth:`training_loss` for
    joint GP, reconstruction, and KL optimization.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        latent_dim: int,
        *,
        beta: float = 1.0,
        reconstruction_weight: float = 1.0,
        **kwargs,
    ) -> None:
        if beta < 0:
            raise ValueError("beta must be non-negative.")
        if reconstruction_weight < 0:
            raise ValueError("reconstruction_weight must be non-negative.")
        super().__init__(train_X=train_X, train_Y=train_Y, latent_dim=latent_dim, **kwargs)
        self.beta = float(beta)
        self.reconstruction_weight = float(reconstruction_weight)

        encoder = self.encoder
        if not isinstance(encoder[-1], nn.Linear):
            raise RuntimeError("JointVAEGP requires a linear encoder output layer.")
        hidden_dim = encoder[-1].in_features
        self.encoder_body = nn.Sequential(*list(encoder.children())[:-1])
        self.mu_head = encoder[-1]
        self.logvar_head = nn.Linear(hidden_dim, self.latent_dim).to(
            device=train_X.device,
            dtype=train_X.dtype,
        )
        self.encoder = nn.Sequential(self.encoder_body, self.mu_head)
        self.decoder = self._make_decoder(
            train_X.shape[-1],
            device=train_X.device,
            dtype=train_X.dtype,
        )

    def _make_decoder(
        self,
        output_dim: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> nn.Sequential:
        return make_feature_network(
            self.latent_dim,
            output_dim,
            self.hidden_dims,
            self.activation,
            reverse=True,
            device=device,
            dtype=dtype,
        )

    def encode_distribution(self, X: Tensor) -> tuple[Tensor, Tensor]:
        """Return latent posterior mean and log variance for original-space X."""
        if X.shape[-1] != self.raw_train_X.shape[-1]:
            raise ValueError(
                f"Expected final dimension {self.raw_train_X.shape[-1]}, got {X.shape[-1]}."
            )
        standardized = (X - self.x_mean) / self.x_scale
        hidden = self.encoder_body(standardized)
        return self.mu_head(hidden), self.logvar_head(hidden)

    def encode(self, X: Tensor) -> Tensor:
        """Return the deterministic posterior-mean latent representation."""
        mu, _ = self.encode_distribution(X)
        return mu

    def sample_latent(self, X: Tensor, n_samples: int) -> Tensor:
        """Draw reparameterized samples from q(z | X)."""
        if n_samples <= 0:
            raise ValueError("n_samples must be a positive integer.")
        mu, logvar = self.encode_distribution(X)
        std = torch.exp(0.5 * logvar)
        noise = torch.randn((n_samples, *std.shape), device=std.device, dtype=std.dtype)
        return mu.unsqueeze(0) + std.unsqueeze(0) * noise

    def reconstruct(self, X: Tensor, *, sample: bool = False) -> Tensor:
        """Reconstruct original-scale X from its VAE latent representation."""
        mu, logvar = self.encode_distribution(X)
        if sample:
            std = torch.exp(0.5 * logvar)
            latent = mu + std * torch.randn_like(std)
        else:
            latent = mu
        standardized = self.decoder(latent)
        return standardized * self.x_scale + self.x_mean

    def reconstruction_loss(self, X: Tensor | None = None) -> Tensor:
        """Return standardized reconstruction MSE using reparameterized z."""
        if X is None:
            X = self.raw_train_X
        target = (X - self.x_mean) / self.x_scale
        mu, logvar = self.encode_distribution(X)
        std = torch.exp(0.5 * logvar)
        latent = mu + std * torch.randn_like(std)
        return torch.nn.functional.mse_loss(self.decoder(latent), target)

    def kl_loss(self, X: Tensor | None = None) -> Tensor:
        """Return mean KL divergence from q(z | X) to the unit Gaussian."""
        if X is None:
            X = self.raw_train_X
        mu, logvar = self.encode_distribution(X)
        return -0.5 * torch.mean(1.0 + logvar - mu.square() - logvar.exp())

    def training_loss(self) -> Tensor:
        """Return negative GP MLL plus reconstruction and KL regularization."""
        loss = super().training_loss()
        if self.reconstruction_weight != 0.0:
            loss = loss + self.reconstruction_weight * self.reconstruction_loss()
        if self.beta != 0.0:
            loss = loss + self.beta * self.kl_loss()
        return loss

    def _training_noise(self) -> Tensor:
        """Return observation noise aligned with the transformed training targets."""
        noise = getattr(self.likelihood, "noise", None)
        if noise is None:
            raise NotImplementedError(
                "uncertainty_aware_posterior requires a likelihood exposing observation noise."
            )
        noise = noise.to(device=self.raw_train_X.device, dtype=self.raw_train_X.dtype)
        n_train = self.raw_train_X.shape[-2]
        if noise.numel() == 1:
            return noise.reshape(1).expand(n_train)
        noise = noise.reshape(-1)
        if noise.numel() != n_train:
            raise NotImplementedError(
                "uncertainty_aware_posterior currently supports scalar or per-observation noise."
            )
        return noise

    def _posterior_moments_from_latent(self, latent_X: Tensor) -> tuple[Tensor, Tensor]:
        """Compute exact scalar-GP posterior moments for one latent q-batch."""
        if self.num_outputs != 1:
            raise NotImplementedError(
                "uncertainty_aware_posterior currently supports single-output JointVAEGP."
            )
        train_Z = self.encode(self.raw_train_X)
        train_mean = self.mean_module(train_Z)
        test_mean = self.mean_module(latent_X)
        train_covar = self.covar_module(train_Z).to_dense()
        cross_covar = self.covar_module(train_Z, latent_X).to_dense()
        test_covar = self.covar_module(latent_X).to_dense()
        noise = self._training_noise()
        train_covar = train_covar + torch.diag_embed(noise)
        residual = self.train_targets - train_mean
        solve_residual = torch.linalg.solve(train_covar, residual.unsqueeze(-1)).squeeze(-1)
        posterior_mean = test_mean + cross_covar.transpose(-1, -2) @ solve_residual
        solve_cross = torch.linalg.solve(train_covar, cross_covar)
        posterior_covar = test_covar - cross_covar.transpose(-1, -2) @ solve_cross
        return posterior_mean, posterior_covar

    def uncertainty_aware_posterior(
        self,
        X: Tensor,
        *,
        n_latent_samples: int = 32,
    ) -> GPyTorchPosterior:
        """Moment-match GP predictions marginalized over ``q(z | X)``.

        This propagates VAE latent uncertainty into both predictive mean and
        covariance. The returned Gaussian matches the first two moments of the
        Monte Carlo mixture and remains differentiable with respect to ``X``.
        """
        if n_latent_samples <= 0:
            raise ValueError("n_latent_samples must be a positive integer.")
        if X.ndim < 2:
            raise ValueError("X must have shape [..., q, d].")

        latent_samples = self.sample_latent(X, n_latent_samples)
        batch_shape = X.shape[:-2]
        q = X.shape[-2]
        flat_latent = latent_samples.reshape(n_latent_samples, -1, q, self.latent_dim)
        component_means: list[Tensor] = []
        component_covars: list[Tensor] = []
        for batch_index in range(flat_latent.shape[1]):
            batch_means: list[Tensor] = []
            batch_covars: list[Tensor] = []
            for sample_index in range(n_latent_samples):
                mean, covar = self._posterior_moments_from_latent(
                    flat_latent[sample_index, batch_index]
                )
                batch_means.append(mean)
                batch_covars.append(covar)
            means = torch.stack(batch_means)
            covars = torch.stack(batch_covars)
            mean = means.mean(dim=0)
            centered = means - mean
            between = torch.einsum("si,sj->sij", centered, centered).mean(dim=0)
            covar = covars.mean(dim=0) + between
            component_means.append(mean)
            component_covars.append(covar)

        posterior_mean = torch.stack(component_means).reshape(*batch_shape, q)
        posterior_covar = torch.stack(component_covars).reshape(*batch_shape, q, q)
        posterior_covar = 0.5 * (posterior_covar + posterior_covar.transpose(-1, -2))
        jitter = torch.finfo(posterior_covar.dtype).eps * 10
        posterior_covar = posterior_covar + jitter * torch.eye(
            q,
            device=posterior_covar.device,
            dtype=posterior_covar.dtype,
        )
        posterior = GPyTorchPosterior(MultivariateNormal(posterior_mean, posterior_covar))
        if self.outcome_transform is not None:
            posterior = self.outcome_transform.untransform_posterior(posterior)
        return posterior


class MixedJointVAEGP(MixedJointEncoderGP):
    """Joint VAE GP with continuous latent representation and native categories."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        latent_dim: int,
        cat_dims: list[int],
        *,
        beta: float = 1.0,
        reconstruction_weight: float = 1.0,
        **kwargs,
    ) -> None:
        if beta < 0:
            raise ValueError("beta must be non-negative.")
        if reconstruction_weight < 0:
            raise ValueError("reconstruction_weight must be non-negative.")
        super().__init__(train_X, train_Y, latent_dim, cat_dims, **kwargs)
        self.beta = float(beta)
        self.reconstruction_weight = float(reconstruction_weight)
        encoder = self.encoder
        if not isinstance(encoder[-1], nn.Linear):
            raise RuntimeError("MixedJointVAEGP requires a linear encoder output layer.")
        hidden_dim = encoder[-1].in_features
        self.encoder_body = nn.Sequential(*list(encoder.children())[:-1])
        self.mu_head = encoder[-1]
        self.logvar_head = nn.Linear(hidden_dim, self.latent_dim).to(
            device=train_X.device,
            dtype=train_X.dtype,
        )
        self.encoder = nn.Sequential(self.encoder_body, self.mu_head)
        self.decoder = self._make_decoder(
            len(self._mixed_cont_dims),
            device=train_X.device,
            dtype=train_X.dtype,
        )

    def _make_decoder(
        self,
        output_dim: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> nn.Sequential:
        return make_feature_network(
            self.latent_dim,
            output_dim,
            self.hidden_dims,
            self.activation,
            reverse=True,
            device=device,
            dtype=dtype,
        )

    def encode_distribution(self, X: Tensor) -> tuple[Tensor, Tensor]:
        continuous = self._continuous(X)
        standardized = (continuous - self.x_mean) / self.x_scale
        hidden = self.encoder_body(standardized)
        return self.mu_head(hidden), self.logvar_head(hidden)

    def encode(self, X: Tensor) -> Tensor:
        mu, _ = self.encode_distribution(X)
        categorical = X[..., list(self._mixed_cat_dims)].to(mu)
        return torch.cat((mu, categorical), dim=-1)

    def reconstruction_loss(self, X: Tensor | None = None) -> Tensor:
        if X is None:
            X = self.raw_train_X
        continuous = self._continuous(X)
        target = (continuous - self.x_mean) / self.x_scale
        mu, logvar = self.encode_distribution(X)
        std = torch.exp(0.5 * logvar)
        latent = mu + std * torch.randn_like(std)
        return torch.nn.functional.mse_loss(self.decoder(latent), target)

    def kl_loss(self, X: Tensor | None = None) -> Tensor:
        if X is None:
            X = self.raw_train_X
        mu, logvar = self.encode_distribution(X)
        return -0.5 * torch.mean(1.0 + logvar - mu.square() - logvar.exp())

    def training_loss(self) -> Tensor:
        loss = super().training_loss()
        if self.reconstruction_weight != 0.0:
            loss = loss + self.reconstruction_weight * self.reconstruction_loss()
        if self.beta != 0.0:
            loss = loss + self.beta * self.kl_loss()
        return loss
