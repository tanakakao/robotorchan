"""Joint neural-feature GP classifiers for unordered and ordinal labels.

The encoder is optimized together with the existing variational GP(s).  The
observation likelihood, probability posterior, and AL APIs are inherited.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from robotorchan.models.classification.multiclass.standard.single_task import (
    MulticlassSingleTaskGPClassifier,
)
from robotorchan.models.classification.ordinal.standard.single_task import (
    OrdinalSingleTaskGPClassifier,
)
from robotorchan.models.expressive.neural_features import (
    make_feature_network,
    validate_feature_output,
    validate_neural_feature_config,
)


def _prepare_features(
    train_X: Tensor,
    latent_dim: int,
    hidden_dims: tuple[int, ...],
    activation: str,
    standardize: bool,
    eps: float,
    random_state: int,
    feature_extractor: nn.Module | None,
) -> tuple[nn.Module, Tensor, Tensor, Tensor]:
    validate_neural_feature_config(latent_dim, hidden_dims, activation, eps)
    if train_X.ndim != 2 or train_X.shape[0] < 2:
        raise ValueError("train_X must have shape [n, d] with n >= 2.")
    mean = train_X.mean(0) if standardize else torch.zeros_like(train_X[0])
    scale = (
        train_X.std(0, unbiased=False).clamp_min(eps)
        if standardize
        else torch.ones_like(train_X[0])
    )
    devices = []
    if train_X.is_cuda:
        devices = [
            train_X.device.index
            if train_X.device.index is not None
            else torch.cuda.current_device()
        ]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(random_state)
        encoder = (
            make_feature_network(
                train_X.shape[-1],
                latent_dim,
                hidden_dims,
                activation,
                device=train_X.device,
                dtype=train_X.dtype,
            )
            if feature_extractor is None
            else feature_extractor.to(device=train_X.device, dtype=train_X.dtype)
        )
    with torch.no_grad():
        features = encoder((train_X - mean) / scale)
    validate_feature_output(features, train_X, latent_dim)
    return encoder, mean, scale, features.detach()


class _JointEncoderInputs:
    """Original-coordinate input mapping; no classification semantics."""

    encoder: nn.Module
    latent_dim: int

    def _install_encoder(
        self,
        train_X: Tensor,
        encoder: nn.Module,
        mean: Tensor,
        scale: Tensor,
        latent_dim: int,
    ) -> None:
        self.latent_dim = latent_dim
        self.encoder = encoder
        self.register_buffer("x_mean", mean.detach().clone())
        self.register_buffer("x_scale", scale.detach().clone())
        self._store_raw_tensor("train_X", train_X.detach().clone())

    def encode(self, X: Tensor) -> Tensor:
        if X.shape[-1] != self.raw_train_X.shape[-1]:
            raise ValueError("Input feature dimension does not match training data.")
        features = self.encoder((X - self.x_mean) / self.x_scale)
        validate_feature_output(features, X, self.latent_dim)
        return features


class JointEncoderOrdinalGPClassifier(_JointEncoderInputs, OrdinalSingleTaskGPClassifier):
    """Ordered-probit variational GP with jointly learned neural input features."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_classes: int,
        latent_dim: int,
        hidden_dims: tuple[int, ...] = (64, 32),
        activation: str = "gelu",
        standardize: bool = True,
        eps: float = 1e-8,
        random_state: int = 0,
        feature_extractor: nn.Module | None = None,
    ) -> None:
        encoder, mean, scale, features = _prepare_features(
            train_X,
            latent_dim,
            hidden_dims,
            activation,
            standardize,
            eps,
            random_state,
            feature_extractor,
        )
        super().__init__(features, train_Y, num_classes=num_classes)
        self._install_encoder(train_X, encoder, mean, scale, latent_dim)

    def posterior(self, X: Tensor, **kwargs: object):
        return super().posterior(self.encode(X), **kwargs)

    def fit(
        self,
        *,
        epochs: int = 60,
        lr: float = 0.03,
        batch_size: int | None = None,
    ) -> None:
        if epochs <= 0 or lr <= 0:
            raise ValueError("epochs and lr must be positive.")
        n = self.raw_train_X.shape[-2]
        batch_size = n if batch_size is None else batch_size
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        loader = DataLoader(
            TensorDataset(self.raw_train_X, self.raw_train_Y),
            batch_size=batch_size,
            shuffle=True,
        )
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        mll = self.make_mll(num_data=n)
        for _ in range(epochs):
            for batch_X, batch_Y in loader:
                optimizer.zero_grad()
                output = self.model(self.encode(batch_X))
                loss = -mll(output, batch_Y.to(output.mean.dtype))
                loss.backward()
                optimizer.step()
        self.eval()


class JointEncoderMulticlassGPClassifier(
    _JointEncoderInputs,
    MulticlassSingleTaskGPClassifier,
):
    """Independent class GPs with a shared, jointly learned neural encoder."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_classes: int,
        latent_dim: int,
        hidden_dims: tuple[int, ...] = (64, 32),
        activation: str = "gelu",
        standardize: bool = True,
        eps: float = 1e-8,
        random_state: int = 0,
        feature_extractor: nn.Module | None = None,
    ) -> None:
        encoder, mean, scale, features = _prepare_features(
            train_X,
            latent_dim,
            hidden_dims,
            activation,
            standardize,
            eps,
            random_state,
            feature_extractor,
        )
        super().__init__(features, train_Y, num_classes=num_classes)
        self._install_encoder(train_X, encoder, mean, scale, latent_dim)

    def posterior(self, X: Tensor, **kwargs: object):
        return super().posterior(self.encode(X), **kwargs)

    def fit(
        self,
        *,
        epochs: int = 60,
        lr: float = 0.03,
        batch_size: int | None = None,
    ) -> None:
        if epochs <= 0 or lr <= 0:
            raise ValueError("epochs and lr must be positive.")
        n = self.raw_train_X.shape[-2]
        batch_size = n if batch_size is None else batch_size
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        loader = DataLoader(
            TensorDataset(self.raw_train_X, self.raw_train_Y),
            batch_size=batch_size,
            shuffle=True,
        )
        self.train()
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.class_models.parameters()),
            lr=lr,
        )
        for _ in range(epochs):
            for batch_X, batch_Y in loader:
                optimizer.zero_grad()
                features = self.encode(batch_X)
                samples = torch.stack(
                    [gp.model(features).rsample() for gp in self.class_models],
                    dim=-1,
                )
                log_probs = samples.log_softmax(dim=-1)
                log_likelihood = log_probs.gather(
                    -1,
                    batch_Y.long().reshape(-1, 1),
                ).mean()
                kl = sum(
                    gp.model.variational_strategy.kl_divergence().sum()
                    for gp in self.class_models
                )
                loss = -log_likelihood + kl / n
                loss.backward()
                optimizer.step()
        self.eval()
