"""Joint neural representation learning for binary GP classification."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from robotorchan.models.classification.binary.standard.single_task import (\n    BinarySingleTaskGPClassifier,\n)
from robotorchan.models.expressive.neural_features import (
    make_feature_network,
    validate_feature_output,
    validate_neural_feature_config,
)


class JointEncoderBinaryGPClassifier(BinarySingleTaskGPClassifier):
    """Binary variational GP classifier with a jointly optimized neural encoder."""

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
        feature_extractor: nn.Module | None = None,
    ) -> None:
        """Initialize the neural representation and variational latent GP."""
        validate_neural_feature_config(latent_dim, hidden_dims, activation, eps)
        raw_train_X = train_X.detach().clone()
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
                encoder = make_feature_network(
                    train_X.shape[-1],
                    self.latent_dim,
                    self.hidden_dims,
                    self.activation,
                    device=train_X.device,
                    dtype=train_X.dtype,
                )
            else:
                encoder = feature_extractor.to(device=train_X.device, dtype=train_X.dtype)

        standardized_X = (train_X - x_mean) / x_scale
        latent_X = encoder(standardized_X)
        validate_feature_output(latent_X, standardized_X, self.latent_dim)
        super().__init__(train_X=latent_X.detach(), train_Y=train_Y)
        self.encoder = encoder
        self.register_buffer("x_mean", x_mean.detach().clone())
        self.register_buffer("x_scale", x_scale.detach().clone())
        self._store_raw_tensor("train_X", raw_train_X)

    def encode(self, X: Tensor) -> Tensor:
        """Map original-space inputs into the learned latent representation."""
        if X.shape[-1] != self.raw_train_X.shape[-1]:
            raise ValueError(
                f"Expected final dimension {self.raw_train_X.shape[-1]}, got {X.shape[-1]}."
            )
        return self.encoder((X - self.x_mean) / self.x_scale)

    def posterior(self, X: Tensor, **kwargs: object):
        """Evaluate the latent GP posterior after neural feature extraction."""
        return super().posterior(self.encode(X), **kwargs)

    def training_loss(self) -> Tensor:
        """Return negative variational ELBO for joint encoder-GP optimization."""
        self.train()
        self.likelihood.train()
        latent_X = self.encode(self.raw_train_X)
        output = self.model(latent_X)
        target = self.raw_train_Y.to(dtype=latent_X.dtype, device=latent_X.device)
        return -self.make_mll()(output, target)
