"""Joint neural representation models for multi-task Gaussian processes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from gpytorch.distributions import MultivariateNormal
from torch import Tensor, nn

from robotorchan.models.base import normalize_feature_dims
from robotorchan.models.multitask import KroneckerMultiTaskGP, MultiTaskGP

_ACTIVATIONS: dict[str, Callable[[], nn.Module]] = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
}


def _make_network(
    input_dim: int,
    output_dim: int,
    hidden_dims: tuple[int, ...],
    activation: str,
    *,
    reverse: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> nn.Sequential:
    widths = tuple(reversed(hidden_dims)) if reverse else hidden_dims
    layers: list[nn.Module] = []
    previous = input_dim
    for width in widths:
        layers.extend([nn.Linear(previous, width), _ACTIVATIONS[activation]()])
        previous = width
    layers.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*layers).to(device=device, dtype=dtype)


def _validate_config(
    input_dim: int,
    latent_dim: int,
    hidden_dims: tuple[int, ...],
    activation: str,
    eps: float,
) -> None:
    if latent_dim <= 0 or latent_dim > input_dim:
        raise ValueError("latent_dim must be positive and not exceed the encoded input dimension.")
    if any(width <= 0 for width in hidden_dims):
        raise ValueError("hidden_dims must contain only positive integers.")
    if activation not in _ACTIVATIONS:
        raise ValueError(f"Unsupported activation {activation!r}.")
    if eps <= 0:
        raise ValueError("eps must be positive.")


class JointEncoderMultiTaskGP(MultiTaskGP):
    """Long-format multi-task GP with a GP-MLL-trained neural encoder."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        task_feature: int,
        latent_dim: int,
        *,
        hidden_dims: tuple[int, ...] = (64, 32),
        activation: str = "gelu",
        standardize: bool = True,
        eps: float = 1e-8,
        random_state: int = 0,
        **kwargs: Any,
    ) -> None:
        input_dim = train_X.shape[-1]
        task_feature = normalize_feature_dims([task_feature], input_dim, name="task_feature")[0]
        data_dims = tuple(i for i in range(input_dim) if i != task_feature)
        _validate_config(len(data_dims), latent_dim, hidden_dims, activation, eps)
        data_X = train_X[..., list(data_dims)]
        x_mean = data_X.mean(dim=0) if standardize else torch.zeros_like(data_X[0])
        x_scale = (
            data_X.std(dim=0, unbiased=False).clamp_min(eps)
            if standardize
            else torch.ones_like(data_X[0])
        )
        with torch.random.fork_rng():
            torch.manual_seed(random_state)
            encoder = _make_network(
                len(data_dims),
                latent_dim,
                hidden_dims,
                activation,
                reverse=False,
                device=train_X.device,
                dtype=train_X.dtype,
            )
        latent = encoder((data_X - x_mean) / x_scale).detach()
        reduced_X = torch.cat([latent, train_X[..., task_feature : task_feature + 1]], dim=-1)
        super().__init__(reduced_X, train_Y, task_feature=latent_dim, **kwargs)
        self.encoder = encoder
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.original_task_feature = task_feature
        self.data_dims = data_dims
        self._original_input_dim = input_dim
        self.register_buffer("x_mean", x_mean.detach().clone())
        self.register_buffer("x_scale", x_scale.detach().clone())
        self._store_supervised_training_data(train_X, train_Y)

    def encode(self, X: Tensor) -> Tensor:
        """Encode data columns while preserving task identity."""
        if X.shape[-1] != self._original_input_dim:
            raise ValueError(
                f"Expected final dimension {self._original_input_dim}, got {X.shape[-1]}."
            )
        data = X[..., list(self.data_dims)]
        latent = self.encoder((data - self.x_mean) / self.x_scale)
        task = X[..., self.original_task_feature : self.original_task_feature + 1]
        return torch.cat([latent, task], dim=-1)

    def forward(self, X: Tensor) -> MultivariateNormal:
        if X.shape[-1] == self.latent_dim + 1:
            return super().forward(X)
        return super().forward(self.encode(X))

    def posterior(self, X: Tensor, *args: Any, **kwargs: Any):
        return super().posterior(self.encode(X), *args, **kwargs)

    def training_loss(self) -> Tensor:
        self.train()
        self.likelihood.train()
        return -self.make_mll()(self(self.raw_train_X), self.train_targets)


class JointEncoderKroneckerMultiTaskGP(KroneckerMultiTaskGP):
    """Kronecker multi-task GP with a GP-MLL-trained neural encoder."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        latent_dim: int,
        *,
        hidden_dims: tuple[int, ...] = (64, 32),
        activation: str = "gelu",
        standardize: bool = True,
        eps: float = 1e-8,
        random_state: int = 0,
        **kwargs: Any,
    ) -> None:
        input_dim = train_X.shape[-1]
        _validate_config(input_dim, latent_dim, hidden_dims, activation, eps)
        x_mean = train_X.mean(dim=0) if standardize else torch.zeros_like(train_X[0])
        x_scale = (
            train_X.std(dim=0, unbiased=False).clamp_min(eps)
            if standardize
            else torch.ones_like(train_X[0])
        )
        with torch.random.fork_rng():
            torch.manual_seed(random_state)
            encoder = _make_network(
                input_dim,
                latent_dim,
                hidden_dims,
                activation,
                reverse=False,
                device=train_X.device,
                dtype=train_X.dtype,
            )
        latent = encoder((train_X - x_mean) / x_scale).detach()
        super().__init__(latent, train_Y, **kwargs)
        self.encoder = encoder
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.activation = activation
        self._original_input_dim = input_dim
        self.register_buffer("x_mean", x_mean.detach().clone())
        self.register_buffer("x_scale", x_scale.detach().clone())
        self._store_supervised_training_data(train_X, train_Y)
        self.set_train_data(
            inputs=train_X.detach().clone(), targets=self.train_targets, strict=False
        )

    def encode(self, X: Tensor) -> Tensor:
        if X.shape[-1] != self._original_input_dim:
            raise ValueError(
                f"Expected final dimension {self._original_input_dim}, got {X.shape[-1]}."
            )
        return self.encoder((X - self.x_mean) / self.x_scale)

    def forward(self, X: Tensor):
        return super().forward(self.encode(X))

    def posterior(self, X: Tensor, *args: Any, **kwargs: Any):
        return super().posterior(self.encode(X), *args, **kwargs)

    def training_loss(self) -> Tensor:
        self.train()
        self.likelihood.train()
        return -self.make_mll()(self(self.raw_train_X), self.train_targets)


class _HybridMixin:
    reconstruction_weight: float
    decoder: nn.Sequential

    def reconstruction_loss(self, X: Tensor | None = None) -> Tensor:
        if X is None:
            X = self.raw_train_X
        if isinstance(self, JointEncoderMultiTaskGP):
            source = X[..., list(self.data_dims)]
            latent = self.encode(X)[..., : self.latent_dim]
        else:
            source = X
            latent = self.encode(X)
        target = (source - self.x_mean) / self.x_scale
        return torch.nn.functional.mse_loss(self.decoder(latent), target)

    def training_loss(self) -> Tensor:
        loss = super().training_loss()
        if self.reconstruction_weight == 0.0:
            return loss
        return loss + self.reconstruction_weight * self.reconstruction_loss()


class HybridAutoEncoderMultiTaskGP(_HybridMixin, JointEncoderMultiTaskGP):
    """Long-format joint encoder GP with reconstruction regularization."""

    def __init__(self, *args: Any, reconstruction_weight: float = 1.0, **kwargs: Any) -> None:
        if reconstruction_weight < 0:
            raise ValueError("reconstruction_weight must be non-negative.")
        super().__init__(*args, **kwargs)
        self.reconstruction_weight = float(reconstruction_weight)
        self.decoder = _make_network(
            self.latent_dim,
            len(self.data_dims),
            self.hidden_dims,
            self.activation,
            reverse=True,
            device=self.raw_train_X.device,
            dtype=self.raw_train_X.dtype,
        )


class HybridAutoEncoderKroneckerMultiTaskGP(_HybridMixin, JointEncoderKroneckerMultiTaskGP):
    """Kronecker joint encoder GP with reconstruction regularization."""

    def __init__(self, *args: Any, reconstruction_weight: float = 1.0, **kwargs: Any) -> None:
        if reconstruction_weight < 0:
            raise ValueError("reconstruction_weight must be non-negative.")
        super().__init__(*args, **kwargs)
        self.reconstruction_weight = float(reconstruction_weight)
        self.decoder = _make_network(
            self.latent_dim,
            self._original_input_dim,
            self.hidden_dims,
            self.activation,
            reverse=True,
            device=self.raw_train_X.device,
            dtype=self.raw_train_X.dtype,
        )


class _JointVAEMixin(_HybridMixin):
    beta: float
    logvar_head: nn.Linear

    def encode_distribution(self, X: Tensor) -> tuple[Tensor, Tensor]:
        source = X[..., list(self.data_dims)] if isinstance(
            self, JointEncoderMultiTaskGP
        ) else X
        standardized = (source - self.x_mean) / self.x_scale
        body = nn.Sequential(*list(self.encoder.children())[:-1])
        hidden = body(standardized)
        return self.encoder[-1](hidden), self.logvar_head(hidden)

    def kl_loss(self, X: Tensor | None = None) -> Tensor:
        if X is None:
            X = self.raw_train_X
        mu, logvar = self.encode_distribution(X)
        return -0.5 * torch.mean(1.0 + logvar - mu.square() - logvar.exp())

    def training_loss(self) -> Tensor:
        loss = super().training_loss()
        if self.beta == 0.0:
            return loss
        return loss + self.beta * self.kl_loss()


class JointVAEMultiTaskGP(_JointVAEMixin, HybridAutoEncoderMultiTaskGP):
    """Long-format jointly trained VAE representation and multi-task GP."""

    def __init__(self, *args: Any, beta: float = 1.0, **kwargs: Any) -> None:
        if beta < 0:
            raise ValueError("beta must be non-negative.")
        super().__init__(*args, **kwargs)
        self.beta = float(beta)
        final = self.encoder[-1]
        if not isinstance(final, nn.Linear):
            raise RuntimeError("Joint VAE requires a linear encoder output layer.")
        self.logvar_head = nn.Linear(final.in_features, self.latent_dim).to(
            device=self.raw_train_X.device,
            dtype=self.raw_train_X.dtype,
        )


class JointVAEKroneckerMultiTaskGP(_JointVAEMixin, HybridAutoEncoderKroneckerMultiTaskGP):
    """Kronecker jointly trained VAE representation and multi-task GP."""

    def __init__(self, *args: Any, beta: float = 1.0, **kwargs: Any) -> None:
        if beta < 0:
            raise ValueError("beta must be non-negative.")
        super().__init__(*args, **kwargs)
        self.beta = float(beta)
        final = self.encoder[-1]
        if not isinstance(final, nn.Linear):
            raise RuntimeError("Joint VAE requires a linear encoder output layer.")
        self.logvar_head = nn.Linear(final.in_features, self.latent_dim).to(
            device=self.raw_train_X.device,
            dtype=self.raw_train_X.dtype,
        )
