"""Hash embedding Bayesian optimization (HeSBO) search strategy."""

from __future__ import annotations

from typing import Any

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.optim import optimize_acqf
from torch import Tensor

from robotorchan.optim.base import SearchResult, SearchStrategy


def _project_to_original_box(Z: Tensor, embedding: Tensor, bounds: Tensor) -> tuple[Tensor, Tensor]:
    """Project hash-embedded coordinates through a normalized ``[-1, 1]^d`` box."""
    if Z.shape[-1] != embedding.shape[-1]:
        raise ValueError("Z last dimension must equal embedding_dim.")
    raw_normalized = Z @ embedding.transpose(-2, -1)
    normalized = raw_normalized.clamp(-1.0, 1.0)
    center = bounds.mean(dim=0)
    half_range = 0.5 * (bounds[1] - bounds[0])
    X = center + half_range * normalized
    clipping_distance = (raw_normalized - normalized).norm(dim=-1)
    return X, clipping_distance


class _HeSBOAcquisition(AcquisitionFunction):
    """Evaluate an original-space acquisition through a fixed hash embedding."""

    def __init__(
        self,
        acq_function: AcquisitionFunction,
        embedding: Tensor,
        bounds: Tensor,
    ) -> None:
        super().__init__(model=acq_function.model)
        self.acq_function = acq_function
        self.register_buffer("embedding", embedding.detach().clone())
        self.register_buffer("original_bounds", bounds.detach().clone())

    def forward(self, Z: Tensor) -> Tensor:
        X, _ = _project_to_original_box(Z, self.embedding, self.original_bounds)
        return self.acq_function(X)


class HeSBOStrategy(SearchStrategy):
    """Optimize an original-space acquisition through a sparse hash embedding.

    Each original input dimension is assigned to exactly one embedded dimension
    with a random sign. The surrogate and acquisition remain in the public input
    space; only acquisition optimization is performed in the lower-dimensional
    embedded space.
    """

    def __init__(
        self,
        bounds: Tensor,
        *,
        embedding_dim: int,
        embedded_bound: float = 1.0,
        seed: int | None = None,
        num_restarts: int = 10,
        raw_samples: int = 512,
        options: dict[str, Any] | None = None,
        sequential: bool = False,
    ) -> None:
        super().__init__(bounds)
        if embedding_dim < 1 or embedding_dim > self.input_dim:
            raise ValueError("embedding_dim must be between 1 and input_dim.")
        if embedded_bound <= 0:
            raise ValueError("embedded_bound must be positive.")
        if num_restarts < 1:
            raise ValueError("num_restarts must be at least 1.")
        if raw_samples < 1:
            raise ValueError("raw_samples must be at least 1.")

        generator = None
        if seed is not None:
            generator = torch.Generator(device=bounds.device)
            generator.manual_seed(seed)
        assignments = torch.randint(
            embedding_dim,
            (self.input_dim,),
            device=bounds.device,
            generator=generator,
        )
        signs = torch.randint(
            0,
            2,
            (self.input_dim,),
            device=bounds.device,
            generator=generator,
        )
        signs = signs.to(dtype=bounds.dtype).mul_(2).sub_(1)
        embedding = torch.zeros(
            self.input_dim,
            embedding_dim,
            dtype=bounds.dtype,
            device=bounds.device,
        )
        rows = torch.arange(self.input_dim, device=bounds.device)
        embedding[rows, assignments] = signs

        self.embedding = embedding
        self.embedded_bounds = torch.stack(
            [
                torch.full(
                    (embedding_dim,),
                    -embedded_bound,
                    dtype=bounds.dtype,
                    device=bounds.device,
                ),
                torch.full(
                    (embedding_dim,),
                    embedded_bound,
                    dtype=bounds.dtype,
                    device=bounds.device,
                ),
            ]
        )
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples
        self.options = None if options is None else dict(options)
        self.sequential = sequential

    @property
    def embedding_dim(self) -> int:
        """Dimension optimized by HeSBO."""
        return self.embedding.shape[-1]

    def project(self, Z: Tensor) -> Tensor:
        """Project embedded coordinates to feasible original-space candidates."""
        X, _ = _project_to_original_box(Z, self.embedding, self.bounds)
        return X

    def optimize(self, acq_function: AcquisitionFunction, *, q: int = 1) -> SearchResult:
        """Optimize ``acq_function`` in the fixed sparse embedded search space."""
        if q < 1:
            raise ValueError("q must be at least 1.")

        embedded_acq = _HeSBOAcquisition(acq_function, self.embedding, self.bounds)
        embedded_candidates, _ = optimize_acqf(
            acq_function=embedded_acq,
            bounds=self.embedded_bounds,
            q=q,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
            options=self.options,
            sequential=self.sequential,
        )
        candidates, clipping_distance = _project_to_original_box(
            embedded_candidates,
            self.embedding,
            self.bounds,
        )
        with torch.no_grad():
            acquisition_value = acq_function(candidates)

        return SearchResult(
            candidates=candidates,
            acquisition_value=acquisition_value,
            metadata={
                "embedded_candidates": embedded_candidates.detach(),
                "clipping_distance": clipping_distance.detach(),
                "embedding": self.embedding.detach().clone(),
            },
        )
