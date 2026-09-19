"""Supervised neural GP wrappers."""

from __future__ import annotations

from typing import Any

from torch import Tensor

from robotorchan.models.reduced.base import ReducedGP
from robotorchan.models.reduced.mixed import MixedReducedGP
from robotorchan.reduction import SupervisedAutoEncoderInputReducer


class SupervisedAutoEncoderGP(ReducedGP):
    """Single-task GP using a frozen outcome-aware autoencoder representation.

    The autoencoder is pretrained once from ``train_X`` and ``train_Y`` using
    reconstruction and auxiliary outcome-prediction losses. Its encoder is then
    frozen and used as the deterministic input reducer for ``ReducedGP``.
    """

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
        supervised_weight: float = 1.0,
        standardize_y: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            input_reducer=SupervisedAutoEncoderInputReducer(
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
                supervised_weight=supervised_weight,
                standardize_y=standardize_y,
            ),
            **kwargs,
        )


class MixedSupervisedAutoEncoderGP(MixedReducedGP):
    """Mixed GP with an outcome-aware frozen AE over continuous inputs only."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        latent_dim: int,
        cat_dims: list[int],
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
        supervised_weight: float = 1.0,
        standardize_y: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            input_reducer=SupervisedAutoEncoderInputReducer(
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
                supervised_weight=supervised_weight,
                standardize_y=standardize_y,
            ),
            cat_dims=cat_dims,
            **kwargs,
        )
