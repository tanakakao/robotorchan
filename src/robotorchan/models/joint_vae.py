"""Joint variational-autoencoder and Gaussian process models."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from robotorchan.models.joint_neural import _ACTIVATIONS, JointEncoderGP


class JointVAEGP(JointEncoderGP):
    """Exact GP jointly trained with a variational latent input representation.

    GP predictions use the posterior-mean latent code ``mu(X)``. The training
    objective can additionally regularize that representation with VAE input
    reconstruction and KL divergence terms.
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
        layers: list[nn.Module] = []
        previous = self.latent_dim
        for width in reversed(self.hidden_dims):
            layers.extend([nn.Linear(previous, width), _ACTIVATIONS[self.activation]()])
            previous = width
        layers.append(nn.Linear(previous, output_dim))
        return nn.Sequential(*layers).to(device=device, dtype=dtype)

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

    def joint_loss(self) -> Tensor:
        """Return negative GP MLL plus reconstruction and KL regularization."""
        self.train()
        self.likelihood.train()
        output = self(self.raw_train_X)
        loss = -self.make_mll()(output, self.train_targets)
        if self.reconstruction_weight != 0.0:
            loss = loss + self.reconstruction_weight * self.reconstruction_loss()
        if self.beta != 0.0:
            loss = loss + self.beta * self.kl_loss()
        return loss
