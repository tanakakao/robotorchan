"""Reconstruction capabilities for latent-space acquisition optimization.

These adapters deliberately live in the search layer rather than on every
input reducer. Only reducers with a mathematically defined reconstruction are
accepted. The latent bounds returned here are axis-aligned outer bounds of the
linear image of the original box; they are not an exact representation of the
resulting zonotope.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor

from robotorchan.reduction import (
    InputReducer,
    PCAInputReducer,
    RandomProjectionInputReducer,
    ReducerNotFittedError,
)


class LatentReconstruction(ABC):
    """Capability for mapping between original and latent search spaces."""

    def __init__(self, reducer: InputReducer) -> None:
        if not reducer.is_fitted:
            raise ReducerNotFittedError(
                f"{type(reducer).__name__} must be fitted before latent reconstruction."
            )
        self.reducer = reducer

    @property
    def input_dim(self) -> int:
        """Dimension of the public/original input space."""
        return self.reducer.input_dim

    @property
    def latent_dim(self) -> int:
        """Dimension of the latent search space."""
        return self.reducer.output_dim

    def transform(self, X: Tensor) -> Tensor:
        """Project original-space inputs into the fitted latent space."""
        return self.reducer.transform(X)

    def reconstruct(self, Z: Tensor) -> Tensor:
        """Reconstruct latent points in the original/public input space."""
        self._validate_latent(Z)
        original_shape = Z.shape
        restored = self._reconstruct_2d(Z.reshape(-1, self.latent_dim))
        expected_shape = (Z.numel() // self.latent_dim, self.input_dim)
        if restored.shape != expected_shape:
            raise RuntimeError(
                "Latent reconstruction returned an invalid shape: "
                f"expected {expected_shape}, got {tuple(restored.shape)}."
            )
        return restored.reshape(*original_shape[:-1], self.input_dim)

    def transform_bounds(self, bounds: Tensor) -> Tensor:
        """Return coordinate-wise outer bounds of the transformed input box."""
        self._validate_bounds(bounds)
        return self._transform_bounds(bounds)

    def _validate_latent(self, Z: Tensor) -> None:
        if Z.ndim < 1:
            raise ValueError("Z must have at least one dimension.")
        if Z.shape[-1] != self.latent_dim:
            raise ValueError(
                f"Expected latent final dimension {self.latent_dim}, got {Z.shape[-1]}."
            )

    def _validate_bounds(self, bounds: Tensor) -> None:
        if bounds.ndim != 2 or bounds.shape != (2, self.input_dim):
            raise ValueError(f"bounds must have shape [2, {self.input_dim}].")
        if not (bounds[0] < bounds[1]).all():
            raise ValueError("Every lower bound must be strictly below its upper bound.")

    @abstractmethod
    def _reconstruct_2d(self, Z: Tensor) -> Tensor:
        """Reconstruct a two-dimensional latent tensor."""

    @abstractmethod
    def _transform_bounds(self, bounds: Tensor) -> Tensor:
        """Transform a box into coordinate-wise latent outer bounds."""

    @staticmethod
    def _linear_outer_bounds(
        lower: Tensor,
        upper: Tensor,
        matrix: Tensor,
        *,
        offset: Tensor | None = None,
    ) -> Tensor:
        """Compute exact coordinate extrema for a linear image of a box."""
        positive = matrix.clamp_min(0)
        negative = matrix.clamp_max(0)
        latent_lower = lower @ positive + upper @ negative
        latent_upper = upper @ positive + lower @ negative
        if offset is not None:
            latent_lower = latent_lower + offset
            latent_upper = latent_upper + offset
        return torch.stack([latent_lower, latent_upper])


class PCAReconstruction(LatentReconstruction):
    """Exact reconstruction capability for a fitted PCA input reducer.

    Reconstruction maps latent coordinates back to the retained PCA subspace.
    When PCA is dimensionality reducing, this is the orthogonal projection in
    original coordinates rather than recovery of discarded components.
    """

    reducer: PCAInputReducer

    def __init__(self, reducer: PCAInputReducer) -> None:
        if not isinstance(reducer, PCAInputReducer):
            raise TypeError("PCAReconstruction requires PCAInputReducer.")
        super().__init__(reducer)

    def _reconstruct_2d(self, Z: Tensor) -> Tensor:
        assert self.reducer.components is not None
        assert self.reducer.mean is not None
        X = Z @ self.reducer.components.transpose(-2, -1)
        if self.reducer.center:
            X = X + self.reducer.mean
        return X

    def _transform_bounds(self, bounds: Tensor) -> Tensor:
        assert self.reducer.components is not None
        assert self.reducer.mean is not None
        lower, upper = bounds
        if self.reducer.center:
            lower = lower - self.reducer.mean
            upper = upper - self.reducer.mean
        return self._linear_outer_bounds(lower, upper, self.reducer.components)


class RandomProjectionReconstruction(LatentReconstruction):
    """Minimum-norm reconstruction for a fitted Gaussian random projection.

    Random projection is generally many-to-one. ``reconstruct`` therefore uses
    the Moore-Penrose pseudoinverse and does not claim to recover the original
    point. Reconstructed points may also violate original box constraints; the
    search strategy is responsible for feasibility handling.
    """

    reducer: RandomProjectionInputReducer

    def __init__(self, reducer: RandomProjectionInputReducer) -> None:
        if not isinstance(reducer, RandomProjectionInputReducer):
            raise TypeError("RandomProjectionReconstruction requires RandomProjectionInputReducer.")
        super().__init__(reducer)

    def _reconstruct_2d(self, Z: Tensor) -> Tensor:
        assert self.reducer.projection is not None
        return Z @ torch.linalg.pinv(self.reducer.projection)

    def _transform_bounds(self, bounds: Tensor) -> Tensor:
        assert self.reducer.projection is not None
        return self._linear_outer_bounds(bounds[0], bounds[1], self.reducer.projection)
