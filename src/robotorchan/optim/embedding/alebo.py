"""Adaptive linear embedding Bayesian optimization (ALEBO) foundation."""

from __future__ import annotations

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from torch import Tensor

from robotorchan.optim.base import SearchResult, SearchStrategy


def _make_alebo_embedding(
    input_dim: int,
    embedding_dim: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator | None,
) -> Tensor:
    """Create a fixed orthonormal-row linear embedding."""
    gaussian = torch.randn(
        input_dim,
        embedding_dim,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    basis, _ = torch.linalg.qr(gaussian, mode="reduced")
    return basis.transpose(-2, -1).contiguous()


class ALEBOStrategy(SearchStrategy):
    """Foundation for Adaptive Linear Embedding Bayesian Optimization.

    Phase 1 defines the fixed linear subspace and public-space projection
    contract. ALEBO-specific acquisition optimization and GP geometry are
    intentionally deferred rather than approximated with REMBO behavior.
    """

    def __init__(
        self,
        bounds: Tensor,
        *,
        embedding_dim: int,
        seed: int | None = None,
    ) -> None:
        super().__init__(bounds)
        if embedding_dim < 1 or embedding_dim > self.input_dim:
            raise ValueError("embedding_dim must be between 1 and input_dim.")

        generator = None
        if seed is not None:
            generator = torch.Generator(device=bounds.device)
            generator.manual_seed(seed)

        self.embedding = _make_alebo_embedding(
            self.input_dim,
            embedding_dim,
            dtype=bounds.dtype,
            device=bounds.device,
            generator=generator,
        )

    @property
    def embedding_dim(self) -> int:
        """Dimension of the fixed ALEBO search subspace."""
        return self.embedding.shape[0]

    def project(self, Z: Tensor) -> Tensor:
        """Map embedded coordinates to feasible public-space candidates."""
        if Z.shape[-1] != self.embedding_dim:
            raise ValueError("Z last dimension must equal embedding_dim.")
        raw_normalized = Z @ self.embedding
        normalized = raw_normalized.clamp(-1.0, 1.0)
        center = self.bounds.mean(dim=0)
        half_range = 0.5 * (self.bounds[1] - self.bounds[0])
        return center + half_range * normalized

    def optimize(
        self,
        acq_function: AcquisitionFunction,
        *,
        q: int = 1,
    ) -> SearchResult:
        """Optimize an acquisition function with ALEBO."""
        del acq_function, q
        raise NotImplementedError(
            "ALEBO acquisition optimization is not implemented in Phase 1."
        )
