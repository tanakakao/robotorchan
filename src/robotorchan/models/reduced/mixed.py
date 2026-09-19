"""Mixed-input reduction utilities for reduced GP models."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from botorch.models import MixedSingleTaskGP as BoTorchMixedSingleTaskGP
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.utils.types import DEFAULT, _DefaultType
from gpytorch.kernels import Kernel
from gpytorch.likelihoods import Likelihood
from torch import Tensor

from robotorchan.models.base import ExactGPModelMixin, normalize_feature_dims
from robotorchan.reduction import (
    AutoEncoderInputReducer,
    SupervisedAutoEncoderInputReducer,
    SupervisedVAEInputReducer,
    VAEInputReducer,
)
from robotorchan.reduction.base import InputReducer
from robotorchan.reduction.input import (
    PCAInputReducer,
    PLSInputReducer,
    RandomProjectionInputReducer,
)


@dataclass(frozen=True)
class MixedInputLayout:
    """Describe how original mixed inputs map to the reduced GP input space.

    Continuous variables are reduced into a latent block. Categorical variables
    bypass the reducer unchanged and are appended after that latent block.
    """

    input_dim: int
    cat_dims: tuple[int, ...]
    cont_dims: tuple[int, ...]
    latent_dim: int

    @classmethod
    def from_cat_dims(
        cls,
        *,
        input_dim: int,
        cat_dims: list[int],
        latent_dim: int,
    ) -> MixedInputLayout:
        """Validate dimensions and construct a mixed-input layout."""
        if input_dim < 1:
            raise ValueError("input_dim must be positive.")
        if latent_dim < 1:
            raise ValueError("latent_dim must be positive.")
        normalized = normalize_feature_dims(cat_dims, input_dim, name="cat_dims")
        cat_set = set(normalized)
        cont_dims = tuple(dim for dim in range(input_dim) if dim not in cat_set)
        if not cont_dims:
            raise ValueError("Mixed input reduction requires at least one continuous dimension.")
        return cls(
            input_dim=input_dim,
            cat_dims=normalized,
            cont_dims=cont_dims,
            latent_dim=latent_dim,
        )

    @property
    def reduced_input_dim(self) -> int:
        """Dimensionality consumed by the downstream mixed GP."""
        return self.latent_dim + len(self.cat_dims)

    @property
    def reduced_cat_dims(self) -> list[int]:
        """Categorical dimensions after continuous latent coordinates."""
        return list(range(self.latent_dim, self.reduced_input_dim))

    def continuous(self, X: Tensor) -> Tensor:
        """Select continuous columns from original-space inputs."""
        self._validate_original_input(X)
        return X[..., list(self.cont_dims)]

    def categorical(self, X: Tensor) -> Tensor:
        """Select categorical columns from original-space inputs."""
        self._validate_original_input(X)
        return X[..., list(self.cat_dims)]

    def combine(self, latent_X: Tensor, original_X: Tensor) -> Tensor:
        """Append untouched categorical values to reduced continuous inputs."""
        self._validate_original_input(original_X)
        if latent_X.shape[:-1] != original_X.shape[:-1]:
            raise ValueError("latent_X and original_X must have matching leading dimensions.")
        if latent_X.shape[-1] != self.latent_dim:
            raise ValueError(
                f"Expected latent dimension {self.latent_dim}, got {latent_X.shape[-1]}."
            )
        categorical = self.categorical(original_X).to(
            dtype=latent_X.dtype,
            device=latent_X.device,
        )
        return torch.cat((latent_X, categorical), dim=-1)

    def _validate_original_input(self, X: Tensor) -> None:
        if X.shape[-1] != self.input_dim:
            raise ValueError(f"Expected final input dimension {self.input_dim}, got {X.shape[-1]}.")


class MixedInputReducer:
    """Reduce only continuous columns while preserving categorical columns."""

    def __init__(
        self,
        reducer: InputReducer,
        *,
        input_dim: int,
        cat_dims: list[int],
    ) -> None:
        self.reducer = reducer
        latent_dim = self._configured_output_dim(reducer)
        self.layout = MixedInputLayout.from_cat_dims(
            input_dim=input_dim,
            cat_dims=cat_dims,
            latent_dim=latent_dim,
        )

    @staticmethod
    def _configured_output_dim(reducer: InputReducer) -> int:
        """Read the configured latent size without requiring a fitted reducer."""
        if reducer.is_fitted:
            return reducer.output_dim

        configured_dim = getattr(reducer, "n_components", None)
        if configured_dim is None:
            configured_dim = getattr(reducer, "latent_dim", None)
        if configured_dim is None:
            raise ValueError(
                "The wrapped reducer must expose its configured output dimension "
                "through n_components or latent_dim before fitting, or already be fitted."
            )
        latent_dim = int(configured_dim)
        if latent_dim < 1:
            raise ValueError("The reducer output dimension must be positive.")
        return latent_dim

    @property
    def is_fitted(self) -> bool:
        """Whether the wrapped continuous reducer has been fitted."""
        return self.reducer.is_fitted

    @property
    def output_dim(self) -> int:
        """Total reduced dimension including categorical passthrough columns."""
        return self.layout.reduced_input_dim

    @property
    def cat_dims(self) -> list[int]:
        """Categorical indices expected by the downstream mixed GP."""
        return self.layout.reduced_cat_dims

    def fit(self, X: Tensor, Y: Tensor | None = None) -> MixedInputReducer:
        """Fit the wrapped reducer on continuous columns only."""
        self.reducer.fit(self.layout.continuous(X), Y)
        if self.reducer.output_dim != self.layout.latent_dim:
            raise RuntimeError(
                "Reducer output dimension changed during fit: "
                f"expected {self.layout.latent_dim}, got {self.reducer.output_dim}."
            )
        return self

    def transform(self, X: Tensor) -> Tensor:
        """Transform continuous columns and append categorical columns unchanged."""
        latent_X = self.reducer.transform(self.layout.continuous(X))
        return self.layout.combine(latent_X, X)

    def fit_transform(self, X: Tensor, Y: Tensor | None = None) -> Tensor:
        """Fit on continuous columns and transform the full mixed input."""
        self.fit(X, Y)
        return self.transform(X)


class MixedReducedGP(ExactGPModelMixin, BoTorchMixedSingleTaskGP):
    """Mixed single-task GP with reduction applied only to continuous inputs.

    Public inputs remain in the original mixed space. Continuous columns are
    projected into a frozen latent representation while categorical columns
    bypass the reducer unchanged and are appended after the latent block.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        input_reducer: InputReducer,
        cat_dims: list[int],
        train_Yvar: Tensor | None = None,
        cont_kernel_factory: Callable[[torch.Size, int, list[int]], Kernel] | None = None,
        likelihood: Likelihood | None = None,
        outcome_transform: OutcomeTransform | _DefaultType | None = DEFAULT,
        input_transform: InputTransform | None = None,
    ) -> None:
        raw_train_X = train_X.detach().clone()
        raw_train_Y = train_Y.detach().clone()
        raw_train_Yvar = None if train_Yvar is None else train_Yvar.detach().clone()

        mixed_reducer = MixedInputReducer(
            input_reducer,
            input_dim=train_X.shape[-1],
            cat_dims=cat_dims,
        )
        if mixed_reducer.is_fitted:
            reduced_train_X = mixed_reducer.transform(train_X)
        else:
            reduced_train_X = mixed_reducer.fit_transform(train_X, train_Y)

        self._original_input_dim_value = train_X.shape[-1]
        self._original_cat_dims_value = mixed_reducer.layout.cat_dims
        self._mixed_layout = mixed_reducer.layout

        super().__init__(
            train_X=reduced_train_X,
            train_Y=train_Y,
            cat_dims=mixed_reducer.cat_dims,
            train_Yvar=train_Yvar,
            cont_kernel_factory=cont_kernel_factory,
            likelihood=likelihood,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
        )
        self.input_reducer = input_reducer
        self._store_supervised_training_data(
            train_X=raw_train_X,
            train_Y=raw_train_Y,
            train_Yvar=raw_train_Yvar,
        )

    @property
    def original_input_dim(self) -> int:
        """Input dimensionality expected by the public model interface."""
        return self._original_input_dim_value

    @property
    def reduced_input_dim(self) -> int:
        """Input dimensionality used by the underlying mixed GP."""
        return self._mixed_layout.reduced_input_dim

    @property
    def original_cat_dims(self) -> list[int]:
        """Categorical dimensions in the original public input space."""
        return list(self._original_cat_dims_value)

    @property
    def reduced_cat_dims(self) -> list[int]:
        """Categorical dimensions in the reduced GP input space."""
        return self._mixed_layout.reduced_cat_dims

    def _transform_original_inputs(self, X: Tensor) -> Tensor:
        latent_X = self.input_reducer.transform(self._mixed_layout.continuous(X))
        return self._mixed_layout.combine(latent_X, X)

    def _prepare_inputs(self, X: Tensor) -> Tensor:
        if X.shape[-1] == self.original_input_dim:
            return self._transform_original_inputs(X)
        if X.shape[-1] == self.reduced_input_dim:
            return X
        raise ValueError(
            "Expected final input dimension "
            f"{self.original_input_dim} (original) or {self.reduced_input_dim} (reduced), "
            f"got {X.shape[-1]}."
        )

    def posterior(
        self,
        X: Tensor,
        output_indices: list[int] | None = None,
        observation_noise: bool | Tensor = False,
        posterior_transform: Any | None = None,
    ):
        """Evaluate the posterior from original-space or reduced inputs."""
        return super().posterior(
            self._prepare_inputs(X),
            output_indices=output_indices,
            observation_noise=observation_noise,
            posterior_transform=posterior_transform,
        )

    def condition_on_observations(self, X: Tensor, Y: Tensor, **kwargs: Any):
        """Condition on observations supplied in the original mixed space."""
        return super().condition_on_observations(
            X=self._prepare_inputs(X),
            Y=Y,
            **kwargs,
        )

    def load_state_dict(
        self,
        state_dict: dict[str, Tensor],
        strict: bool = True,
        assign: bool = False,
    ):
        """Load parameters and resynchronize reducer-dependent training inputs."""
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        train_X = self._transform_original_inputs(self.raw_train_X)
        if hasattr(self, "input_transform"):
            train_X = self.transform_inputs(train_X)
        self.set_train_data(inputs=train_X, targets=self.train_targets, strict=False)
        return result


class MixedAutoEncoderGP(MixedReducedGP):
    """Mixed GP using a frozen autoencoder on continuous inputs."""

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
        **kwargs: Any,
    ) -> None:
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            input_reducer=AutoEncoderInputReducer(
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
            ),
            cat_dims=cat_dims,
            **kwargs,
        )


class MixedVAEGP(MixedReducedGP):
    """Mixed GP using a frozen VAE posterior-mean representation on continuous inputs."""

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
            cat_dims=cat_dims,
            **kwargs,
        )


class MixedSupervisedAutoEncoderGP(MixedReducedGP):
    """Mixed GP using an outcome-aware frozen autoencoder on continuous inputs."""

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


class MixedSupervisedVAEGP(MixedReducedGP):
    """Mixed GP using an outcome-aware frozen VAE on continuous inputs."""

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
            cat_dims=cat_dims,
            **kwargs,
        )
