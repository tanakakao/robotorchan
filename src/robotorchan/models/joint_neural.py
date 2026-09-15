"""Jointly trained neural feature extractor and Gaussian process models."""

from __future__ import annotations

from collections.abc import Callable

import torch
from botorch.models import SingleTaskGP as BoTorchSingleTaskGP
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.utils.types import DEFAULT, _DefaultType
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import Likelihood
from torch import Tensor, nn

from robotorchan.models.base import ExactGPModelMixin

_ACTIVATIONS: dict[str, Callable[[], nn.Module]] = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
}


class JointEncoderGP(ExactGPModelMixin, BoTorchSingleTaskGP):
    """Exact GP whose neural encoder is optimized jointly through the GP MLL."""

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
        train_Yvar: Tensor | None = None,
        likelihood: Likelihood | None = None,
        outcome_transform: OutcomeTransform | _DefaultType | None = DEFAULT,
    ) -> None:
        if latent_dim <= 0:
            raise ValueError("latent_dim must be a positive integer.")
        if latent_dim > train_X.shape[-1]:
            raise ValueError("latent_dim cannot exceed the original input dimension.")
        if any(width <= 0 for width in hidden_dims):
            raise ValueError("hidden_dims must contain only positive integers.")
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"Unsupported activation {activation!r}. Choose from {sorted(_ACTIVATIONS)}."
            )
        if eps <= 0:
            raise ValueError("eps must be positive.")

        raw_train_X = train_X.detach().clone()
        raw_train_Y = train_Y.detach().clone()
        raw_train_Yvar = None if train_Yvar is None else train_Yvar.detach().clone()

        self.latent_dim = int(latent_dim)
        self.hidden_dims = tuple(int(width) for width in hidden_dims)
        self.activation = activation
        self.standardize = bool(standardize)
        self.eps = float(eps)
        self.random_state = int(random_state)

        if self.standardize:
            x_mean = train_X.mean(dim=0)
            x_scale = train_X.std(dim=0, unbiased=False).clamp_min(self.eps)
        else:
            x_mean = torch.zeros_like(train_X[0])
            x_scale = torch.ones_like(train_X[0])

        cuda_devices: list[int] = []
        if train_X.device.type == "cuda":
            device_index = train_X.device.index
            if device_index is None:
                device_index = torch.cuda.current_device()
            cuda_devices = [device_index]

        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(self.random_state)
            encoder = self._make_encoder(
                train_X.shape[-1],
                device=train_X.device,
                dtype=train_X.dtype,
            )

        standardized_X = (train_X - x_mean) / x_scale
        latent_X = encoder(standardized_X).detach()
        super().__init__(
            train_X=latent_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            likelihood=likelihood,
            outcome_transform=outcome_transform,
        )

        self.encoder = encoder
        self.register_buffer("x_mean", x_mean.detach().clone())
        self.register_buffer("x_scale", x_scale.detach().clone())
        self._store_supervised_training_data(
            train_X=raw_train_X,
            train_Y=raw_train_Y,
            train_Yvar=raw_train_Yvar,
        )
        self.set_train_data(inputs=raw_train_X, targets=self.train_targets, strict=False)

    def _make_encoder(
        self,
        input_dim: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> nn.Sequential:
        layers: list[nn.Module] = []
        previous = input_dim
        for width in self.hidden_dims:
            layers.extend([nn.Linear(previous, width), _ACTIVATIONS[self.activation]()])
            previous = width
        layers.append(nn.Linear(previous, self.latent_dim))
        return nn.Sequential(*layers).to(device=device, dtype=dtype)

    def encode(self, X: Tensor) -> Tensor:
        """Map original-space inputs to the jointly learned latent space."""
        if X.shape[-1] != self.raw_train_X.shape[-1]:
            raise ValueError(
                f"Expected final dimension {self.raw_train_X.shape[-1]}, got {X.shape[-1]}."
            )
        return self.encoder((X - self.x_mean) / self.x_scale)

    def forward(self, X: Tensor) -> MultivariateNormal:
        """Evaluate the GP after applying the learnable encoder."""
        latent_X = self.encode(X)
        mean_x = self.mean_module(latent_X)
        covar_x = self.covar_module(latent_X)
        return MultivariateNormal(mean_x, covar_x)


class HybridAutoEncoderGP(JointEncoderGP):
    """Joint encoder-GP model with autoencoder reconstruction regularization.

    The GP marginal log likelihood trains the predictive latent representation,
    while a decoder regularizes that representation to retain information about
    the original inputs. Use :meth:`hybrid_loss` for joint optimization.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        latent_dim: int,
        *,
        reconstruction_weight: float = 1.0,
        **kwargs,
    ) -> None:
        if reconstruction_weight < 0:
            raise ValueError("reconstruction_weight must be non-negative.")
        super().__init__(train_X=train_X, train_Y=train_Y, latent_dim=latent_dim, **kwargs)
        self.reconstruction_weight = float(reconstruction_weight)
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

    def reconstruct(self, X: Tensor) -> Tensor:
        """Reconstruct original-scale inputs from their latent representation."""
        standardized = self.decoder(self.encode(X))
        return standardized * self.x_scale + self.x_mean

    def reconstruction_loss(self, X: Tensor | None = None) -> Tensor:
        """Return standardized input reconstruction MSE."""
        if X is None:
            X = self.raw_train_X
        target = (X - self.x_mean) / self.x_scale
        return torch.nn.functional.mse_loss(self.decoder(self.encode(X)), target)

    def hybrid_loss(self) -> Tensor:
        """Return negative exact MLL plus weighted reconstruction loss."""
        self.train()
        self.likelihood.train()
        output = self(self.raw_train_X)
        negative_mll = -self.make_mll()(output, self.train_targets)
        return negative_mll + self.reconstruction_weight * self.reconstruction_loss()
