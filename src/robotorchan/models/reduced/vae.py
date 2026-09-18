"""Variational-autoencoder GP wrappers."""

from __future__ import annotations

from typing import Any

from torch import Tensor

from robotorchan.models.neural_reduction import VAEInputReducer
from robotorchan.models.reduced.base import ReducedGP
from robotorchan.models.supervised_neural_reduction import SupervisedVAEInputReducer


class VAEGP(ReducedGP):
    """Single-task GP using a frozen VAE posterior-mean input representation."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        latent_dim: int,
        *,
        hidden_dims: tuple[int, ...] = (64, 32),
        activation: str = "gelu",
        epochs: int = 200,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int | None = None,
        standardize: bool = True,
        eps: float = 1e-8,
        random_state: int = 0,
        beta: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            input_reducer=VAEInputReducer(
                latent_dim=latent_dim,
                hidden_dims=hidden_dims,
                activation=activation,
                epochs=epochs,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                batch_size=batch_size,
                standardize=standardize,
                eps=eps,
                random_state=random_state,
                beta=beta,
            ),
            **kwargs,
        )


class SupervisedVAEGP(ReducedGP):
    """Single-task GP using a supervised frozen VAE representation."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        latent_dim: int,
        *,
        hidden_dims: tuple[int, ...] = (64, 32),
        activation: str = "gelu",
        epochs: int = 200,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int | None = None,
        standardize: bool = True,
        eps: float = 1e-8,
        random_state: int = 0,
        beta: float = 1.0,
        supervised_weight: float = 1.0,
        standardize_y: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            input_reducer=SupervisedVAEInputReducer(
                latent_dim=latent_dim,
                hidden_dims=hidden_dims,
                activation=activation,
                epochs=epochs,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                batch_size=batch_size,
                standardize=standardize,
                eps=eps,
                random_state=random_state,
                beta=beta,
                supervised_weight=supervised_weight,
                standardize_y=standardize_y,
            ),
            **kwargs,
        )
