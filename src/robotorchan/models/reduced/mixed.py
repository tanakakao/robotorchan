"""Mixed-input reduction utilities for reduced GP models."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from robotorchan.reduction.base import InputReducer


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
        normalized = tuple(int(dim) for dim in cat_dims)
        if not normalized:
            raise ValueError("cat_dims must contain at least one categorical dimension.")
        if len(set(normalized)) != len(normalized):
            raise ValueError("cat_dims must not contain duplicates.")
        if any(dim < 0 or dim >= input_dim for dim in normalized):
            raise ValueError(f"cat_dims must be within [0, {input_dim}).")
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

        n_components = getattr(reducer, "n_components", None)
        if n_components is None:
            raise ValueError(
                "The wrapped reducer must expose its configured output dimension "
                "before fitting or already be fitted."
            )
        latent_dim = int(n_components)
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
