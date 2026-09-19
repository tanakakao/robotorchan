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

from robotorchan.models.base import (
    ExactGPModelMixin,
    make_mixed_covar_module,
    normalize_feature_dims,
)

_ACTIVATIONS: dict[str, Callable[[], nn.Module]] = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
}


class JointEncoderGP(ExactGPModelMixin, BoTorchSingleTaskGP):
    """Exact GP whose neural encoder is optimized jointly through the GP MLL.

    Unlike frozen-reducer models, the encoder is part of the predictive model and
    receives gradients from the GP objective. :meth:`training_loss` is the common
    optimization contract for all jointly trained neural GP variants.
    """

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
        feature_extractor: nn.Module | None = None,
    ) -> None:
        if latent_dim <= 0:
            raise ValueError("latent_dim must be a positive integer.")
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
            if feature_extractor is None:
                encoder = self._make_encoder(
                    train_X.shape[-1],
                    device=train_X.device,
                    dtype=train_X.dtype,
                )
            else:
                encoder = feature_extractor.to(device=train_X.device, dtype=train_X.dtype)

        standardized_X = (train_X - x_mean) / x_scale
        latent_X = encoder(standardized_X)
        if latent_X.shape[:-1] != standardized_X.shape[:-1]:
            raise ValueError("feature_extractor must preserve all non-feature input dimensions.")
        if latent_X.shape[-1] != self.latent_dim:
            raise ValueError(
                "feature_extractor output dimension must equal latent_dim; "
                f"expected {self.latent_dim}, got {latent_X.shape[-1]}."
            )
        latent_X = latent_X.detach()
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

    def training_loss(self) -> Tensor:
        """Return the scalar loss used to jointly optimize encoder and GP.

        This base implementation is the negative exact GP marginal log
        likelihood. Subclasses extend the same contract with representation
        regularizers such as reconstruction or KL losses.
        """
        self.train()
        self.likelihood.train()
        output = self(self.raw_train_X)
        return -self.make_mll()(output, self.train_targets)


class HybridAutoEncoderGP(JointEncoderGP):
    """Joint encoder-GP model with autoencoder reconstruction regularization.

    The GP marginal log likelihood trains the predictive latent representation,
    while a decoder regularizes that representation to retain information about
    the original inputs. Use :meth:`training_loss` for joint optimization.
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

    def training_loss(self) -> Tensor:
        """Return negative GP MLL plus weighted reconstruction loss."""
        loss = super().training_loss()
        if self.reconstruction_weight == 0.0:
            return loss
        return loss + self.reconstruction_weight * self.reconstruction_loss()


class MixedJointEncoderGP(JointEncoderGP):
    """Joint encoder GP with continuous-only representation and native categories."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        latent_dim: int,
        cat_dims: list[int],
        **kwargs,
    ) -> None:
        input_dim = train_X.shape[-1]
        normalized = normalize_feature_dims(cat_dims, input_dim, name="cat_dims")
        cat_set = set(normalized)
        cont_dims = tuple(i for i in range(input_dim) if i not in cat_set)
        if not cont_dims:
            raise ValueError("Mixed joint representation requires a continuous dimension.")
        self._mixed_input_dim = input_dim
        self._mixed_cat_dims = normalized
        self._mixed_cont_dims = cont_dims
        continuous_X = train_X[..., list(cont_dims)]
        if latent_dim > continuous_X.shape[-1]:
            raise ValueError("latent_dim cannot exceed the continuous input dimension.")
        super().__init__(continuous_X, train_Y, latent_dim, **kwargs)
        self._store_supervised_training_data(train_X=train_X, train_Y=train_Y)
        reduced_cat_dims = list(range(latent_dim, latent_dim + len(normalized)))
        self.covar_module = make_mixed_covar_module(
            input_dim=latent_dim + len(normalized),
            cat_dims=reduced_cat_dims,
        )
        self.set_train_data(
            inputs=train_X.detach().clone(),
            targets=self.train_targets,
            strict=False,
        )

    @property
    def cat_dims(self) -> list[int]:
        return list(self._mixed_cat_dims)

    def _continuous(self, X: Tensor) -> Tensor:
        if X.shape[-1] != self._mixed_input_dim:
            raise ValueError(
                f"Expected final dimension {self._mixed_input_dim}, got {X.shape[-1]}."
            )
        return X[..., list(self._mixed_cont_dims)]

    def encode(self, X: Tensor) -> Tensor:
        """Encode continuous columns and append untouched categorical columns."""
        continuous = self._continuous(X)
        latent = self.encoder((continuous - self.x_mean) / self.x_scale)
        categorical = X[..., list(self._mixed_cat_dims)].to(latent)
        return torch.cat((latent, categorical), dim=-1)


class MixedHybridAutoEncoderGP(MixedJointEncoderGP):
    """Mixed joint encoder GP with continuous-only reconstruction regularization."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        latent_dim: int,
        cat_dims: list[int],
        *,
        reconstruction_weight: float = 1.0,
        **kwargs,
    ) -> None:
        if reconstruction_weight < 0:
            raise ValueError("reconstruction_weight must be non-negative.")
        super().__init__(train_X, train_Y, latent_dim, cat_dims, **kwargs)
        self.reconstruction_weight = float(reconstruction_weight)
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
        layers: list[nn.Module] = []
        previous = self.latent_dim
        for width in reversed(self.hidden_dims):
            layers.extend([nn.Linear(previous, width), _ACTIVATIONS[self.activation]()])
            previous = width
        layers.append(nn.Linear(previous, output_dim))
        return nn.Sequential(*layers).to(device=device, dtype=dtype)

    def reconstruct(self, X: Tensor) -> Tensor:
        latent = self.encode(X)[..., : self.latent_dim]
        standardized = self.decoder(latent)
        return standardized * self.x_scale + self.x_mean

    def reconstruction_loss(self, X: Tensor | None = None) -> Tensor:
        if X is None:
            X = self.raw_train_X
        continuous = self._continuous(X)
        target = (continuous - self.x_mean) / self.x_scale
        latent = self.encode(X)[..., : self.latent_dim]
        return torch.nn.functional.mse_loss(self.decoder(latent), target)

    def training_loss(self) -> Tensor:
        loss = super().training_loss()
        if self.reconstruction_weight == 0.0:
            return loss
        return loss + self.reconstruction_weight * self.reconstruction_loss()
