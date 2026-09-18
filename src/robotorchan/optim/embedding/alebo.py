"""Adaptive Linear Embedding Bayesian Optimization (ALEBO)."""

from __future__ import annotations

from typing import Any

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.optim import optimize_acqf
from botorch.utils.sampling import HitAndRunPolytopeSampler
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
    """Sample ALEBO projection columns uniformly from the unit hypersphere."""
    projection = torch.randn(
        embedding_dim,
        input_dim,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    return projection / projection.norm(dim=0, keepdim=True).clamp_min(torch.finfo(dtype).eps)


class ALEBOStrategy(SearchStrategy):
    """Optimize an acquisition function in a fixed ALEBO linear subspace.

    Candidates are restricted to the embedded polytope whose linear projection
    remains inside the public input bounds, so no clipping is applied.
    """

    def __init__(
        self,
        bounds: Tensor,
        *,
        embedding_dim: int,
        seed: int | None = None,
        num_restarts: int = 10,
        options: dict[str, Any] | None = None,
        sequential: bool = False,
    ) -> None:
        super().__init__(bounds)
        if embedding_dim < 1 or embedding_dim > self.input_dim:
            raise ValueError("embedding_dim must be between 1 and input_dim.")
        if num_restarts < 1:
            raise ValueError("num_restarts must be at least 1.")
        generator = None
        if seed is not None:
            generator = torch.Generator(device=bounds.device)
            generator.manual_seed(seed)

        self.num_restarts = num_restarts
        self.seed = seed
        self.options = None if options is None else dict(options)
        self.sequential = sequential

        self.embedding = _make_alebo_embedding(
            self.input_dim,
            embedding_dim,
            dtype=bounds.dtype,
            device=bounds.device,
            generator=generator,
        )
        self.embedding_pinv = torch.linalg.pinv(self.embedding)

    @property
    def embedding_dim(self) -> int:
        """Dimension of the fixed ALEBO search subspace."""
        return self.embedding.shape[0]

    @property
    def linear_constraints(self) -> tuple[Tensor, Tensor]:
        """Return A and b for the ALEBO polytope A @ z <= b."""
        coefficients = self.embedding_pinv
        A = torch.cat((coefficients, -coefficients), dim=0)
        b = torch.ones(
            2 * self.input_dim,
            dtype=self.bounds.dtype,
            device=self.bounds.device,
        )
        return A, b

    def is_feasible(self, Z: Tensor, *, atol: float = 1e-10) -> Tensor:
        """Return a mask indicating membership in the ALEBO polytope."""
        if Z.shape[-1] != self.embedding_dim:
            raise ValueError("Z last dimension must equal embedding_dim.")
        normalized = Z @ self.embedding_pinv.transpose(-2, -1)
        return (normalized.abs() <= 1.0 + atol).all(dim=-1)

    def project(self, Z: Tensor) -> Tensor:
        """Map feasible embedded coordinates linearly to public input units."""
        if Z.shape[-1] != self.embedding_dim:
            raise ValueError("Z last dimension must equal embedding_dim.")
        if not bool(self.is_feasible(Z).all()):
            raise ValueError("Z must satisfy the ALEBO embedding polytope constraints.")
        normalized = Z @ self.embedding_pinv.transpose(-2, -1)
        normalized = normalized.clamp(-1.0, 1.0)
        center = self.bounds.mean(dim=0)
        half_range = 0.5 * (self.bounds[1] - self.bounds[0])
        return center + half_range * normalized

    def sample_feasible(
        self,
        n: int,
        *,
        seed: int | None = None,
    ) -> Tensor:
        """Sample embedded points directly from the feasible ALEBO polytope."""
        if n < 1:
            raise ValueError("n must be at least 1.")
        A, b = self.linear_constraints
        sampler = HitAndRunPolytopeSampler(
            inequality_constraints=(A, b.unsqueeze(-1)),
            interior_point=self.bounds.new_zeros((self.embedding_dim, 1)),
        )
        with torch.random.fork_rng(devices=[]):
            if seed is not None:
                torch.manual_seed(seed)
            samples = sampler.draw(n=n)
        return samples.to(dtype=self.bounds.dtype, device=self.bounds.device)

    def optimize(
        self,
        acq_function: AcquisitionFunction,
        *,
        q: int = 1,
    ) -> SearchResult:
        """Optimize an embedded-space acquisition over the ALEBO polytope."""
        if q < 1:
            raise ValueError("q must be at least 1.")

        A, b = self.linear_constraints
        inequality_constraints = [
            (
                torch.arange(self.embedding_dim, device=self.bounds.device),
                -A[row],
                -b[row],
            )
            for row in range(A.shape[0])
        ]
        radius = torch.linalg.vector_norm(self.embedding, ord=1, dim=1).max()
        embedded_bounds = torch.stack(
            [
                self.bounds.new_full((self.embedding_dim,), -radius),
                self.bounds.new_full((self.embedding_dim,), radius),
            ]
        )
        batch_initial_conditions = self.sample_feasible(
            self.num_restarts * q,
            seed=self.seed,
        ).reshape(self.num_restarts, q, self.embedding_dim)
        embedded_candidates, acquisition_value = optimize_acqf(
            acq_function=acq_function,
            bounds=embedded_bounds,
            q=q,
            num_restarts=self.num_restarts,
            raw_samples=None,
            batch_initial_conditions=batch_initial_conditions,
            options=self.options,
            inequality_constraints=inequality_constraints,
            sequential=self.sequential,
        )
        candidates = self.project(embedded_candidates)

        return SearchResult(
            candidates=candidates,
            acquisition_value=acquisition_value.reshape(()),
            metadata={
                "embedded_candidates": embedded_candidates.detach(),
                "embedding": self.embedding.detach().clone(),
            },
        )
