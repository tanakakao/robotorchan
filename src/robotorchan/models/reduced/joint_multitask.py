"""Joint neural representation models for multi-task Gaussian processes."""

from __future__ import annotations

from typing import Any

import torch
from gpytorch.distributions import MultivariateNormal
from torch import Tensor, nn

from robotorchan.models.base import make_mixed_covar_module, normalize_feature_dims
from robotorchan.models.multitask import KroneckerMultiTaskGP, MultiTaskGP
from robotorchan.models.neural_features import (
    make_feature_network,
    validate_feature_output,
    validate_neural_feature_config,
)


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
        feature_extractor: nn.Module | None = None,
        **kwargs: Any,
    ) -> None:
        input_dim = train_X.shape[-1]
        task_feature = normalize_feature_dims([task_feature], input_dim, name="task_feature")[0]
        data_dims = tuple(i for i in range(input_dim) if i != task_feature)
        validate_neural_feature_config(latent_dim, hidden_dims, activation, eps)
        data_X = train_X[..., list(data_dims)]
        x_mean = data_X.mean(dim=0) if standardize else torch.zeros_like(data_X[0])
        x_scale = (
            data_X.std(dim=0, unbiased=False).clamp_min(eps)
            if standardize
            else torch.ones_like(data_X[0])
        )
        with torch.random.fork_rng():
            torch.manual_seed(random_state)
            encoder = (
                make_feature_network(
                    len(data_dims),
                    latent_dim,
                    hidden_dims,
                    activation,
                    device=train_X.device,
                    dtype=train_X.dtype,
                )
                if feature_extractor is None
                else feature_extractor.to(device=train_X.device, dtype=train_X.dtype)
            )
        latent = encoder((data_X - x_mean) / x_scale)
        validate_feature_output(latent, data_X, latent_dim)
        latent = latent.detach()
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
        return super().forward(X)

    def posterior(self, X: Tensor, *args: Any, **kwargs: Any):
        return super().posterior(self.encode(X), *args, **kwargs)

    def training_loss(self) -> Tensor:
        self.train()
        self.likelihood.train()
        encoded_X = self.encode(self.raw_train_X)
        self.set_train_data(inputs=encoded_X, targets=self.train_targets, strict=False)
        function_dist = super().forward(encoded_X)
        task_indices = encoded_X[..., -1:].long()
        return -self.make_mll()(function_dist, self.train_targets, task_indices)


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
        feature_extractor: nn.Module | None = None,
        **kwargs: Any,
    ) -> None:
        input_dim = train_X.shape[-1]
        validate_neural_feature_config(latent_dim, hidden_dims, activation, eps)
        x_mean = train_X.mean(dim=0) if standardize else torch.zeros_like(train_X[0])
        x_scale = (
            train_X.std(dim=0, unbiased=False).clamp_min(eps)
            if standardize
            else torch.ones_like(train_X[0])
        )
        with torch.random.fork_rng():
            torch.manual_seed(random_state)
            encoder = (
                make_feature_network(
                    input_dim,
                    latent_dim,
                    hidden_dims,
                    activation,
                    device=train_X.device,
                    dtype=train_X.dtype,
                )
                if feature_extractor is None
                else feature_extractor.to(device=train_X.device, dtype=train_X.dtype)
            )
        latent = encoder((train_X - x_mean) / x_scale)
        validate_feature_output(latent, train_X, latent_dim)
        latent = latent.detach()
        super().__init__(latent, train_Y, **kwargs)
        self.encoder = encoder
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.activation = activation
        self._original_input_dim = input_dim
        self.register_buffer("x_mean", x_mean.detach().clone())
        self.register_buffer("x_scale", x_scale.detach().clone())
        self._store_supervised_training_data(train_X, train_Y)

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
        latent = self.encode(self.raw_train_X)
        self.set_train_data(
            inputs=latent.detach().clone(),
            targets=self.train_targets,
            strict=False,
        )
        return -self.make_mll()(super().forward(latent), self.train_targets)


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
        self.decoder = make_feature_network(
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
        self.decoder = make_feature_network(
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
        source = X[..., list(self.data_dims)] if isinstance(self, JointEncoderMultiTaskGP) else X
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



class MixedJointEncoderMultiTaskGP(JointEncoderMultiTaskGP):
    """Long-format mixed-input DKL preserving categories and task identity."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        task_feature: int,
        latent_dim: int,
        cat_dims: list[int],
        **kwargs: Any,
    ) -> None:
        input_dim = train_X.shape[-1]
        task_dim = normalize_feature_dims([task_feature], input_dim, name="task_feature")[0]
        cats = normalize_feature_dims(
            cat_dims,
            input_dim,
            name="cat_dims",
            excluded_dims=[task_dim],
        )
        cat_set = set(cats)
        continuous_dims = tuple(
            i for i in range(input_dim) if i != task_dim and i not in cat_set
        )
        if not continuous_dims:
            raise ValueError("Mixed DKL multi-task model requires a continuous data dimension.")
        continuous_X = train_X[..., list(continuous_dims)]
        if latent_dim > continuous_X.shape[-1]:
            raise ValueError("latent_dim cannot exceed the continuous input dimension.")

        reduced_train_X = torch.cat(
            (
                continuous_X,
                train_X[..., list(cats)],
                train_X[..., task_dim : task_dim + 1],
            ),
            dim=-1,
        )
        reduced_task_feature = reduced_train_X.shape[-1] - 1
        super().__init__(
            reduced_train_X,
            train_Y,
            task_feature=reduced_task_feature,
            latent_dim=latent_dim,
            **kwargs,
        )

        self._mixed_original_input_dim = input_dim
        self._mixed_original_task_feature = task_dim
        self._mixed_cat_dims = tuple(cats)
        self._mixed_continuous_dims = continuous_dims
        self.original_task_feature = task_dim
        self.data_dims = tuple(i for i in range(input_dim) if i != task_dim)
        self._original_input_dim = input_dim
        continuous_mean = continuous_X.mean(dim=0)
        continuous_scale = continuous_X.std(dim=0, unbiased=False).clamp_min(self.eps)
        if not self.standardize:
            continuous_mean = torch.zeros_like(continuous_X[0])
            continuous_scale = torch.ones_like(continuous_X[0])
        self.x_mean = continuous_mean.detach().clone()
        self.x_scale = continuous_scale.detach().clone()
        self._store_supervised_training_data(train_X, train_Y)

        reduced_cat_dims = list(range(latent_dim, latent_dim + len(cats)))
        data_covar = make_mixed_covar_module(
            input_dim=latent_dim + len(cats),
            cat_dims=reduced_cat_dims,
        )
        data_covar.active_dims = torch.arange(
            latent_dim + len(cats) + 1,
            device=train_X.device,
        )
        self.covar_module.kernels[0] = data_covar

    @property
    def cat_dims(self) -> tuple[int, ...]:
        return self._mixed_cat_dims

    @property
    def continuous_dims(self) -> tuple[int, ...]:
        return self._mixed_continuous_dims

    def encode(self, X: Tensor) -> Tensor:
        if X.shape[-1] != self._mixed_original_input_dim:
            raise ValueError(
                f"Expected final dimension {self._mixed_original_input_dim}, got {X.shape[-1]}."
            )
        continuous = X[..., list(self._mixed_continuous_dims)]
        latent = self.encoder((continuous - self.x_mean) / self.x_scale)
        categorical = X[..., list(self._mixed_cat_dims)].to(latent)
        task = X[
            ..., self._mixed_original_task_feature : self._mixed_original_task_feature + 1
        ].to(latent)
        return torch.cat((latent, categorical, task), dim=-1)

    def training_loss(self) -> Tensor:
        self.train()
        self.likelihood.train()
        encoded_X = self.encode(self.raw_train_X)
        self.set_train_data(inputs=encoded_X, targets=self.train_targets, strict=False)
        function_dist = MultiTaskGP.forward(self, encoded_X)
        task_indices = encoded_X[..., -1:].long()
        return -self.make_mll()(function_dist, self.train_targets, task_indices)
