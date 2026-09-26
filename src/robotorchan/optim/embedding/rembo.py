"""Random embedding Bayesian optimization (REMBO) search strategy."""

from __future__ import annotations

from typing import Any

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.optim import optimize_acqf
from torch import Tensor

from robotorchan.optim.base import SearchResult, SearchStrategy
from robotorchan.optim.constraints import (
    CandidateConstraints,
    reject_unmapped_candidate_constraints,
)


def _project_to_original_box(Z: Tensor, embedding: Tensor, bounds: Tensor) -> tuple[Tensor, Tensor]:
    """Project embedded coordinates through a normalized ``[-1, 1]^d`` box."""
    if Z.shape[-1] != embedding.shape[-1]:
        raise ValueError("Z last dimension must equal embedding_dim.")
    raw_normalized = Z @ embedding.transpose(-2, -1)
    normalized = raw_normalized.clamp(-1.0, 1.0)
    center = bounds.mean(dim=0)
    half_range = 0.5 * (bounds[1] - bounds[0])
    X = center + half_range * normalized
    clipping_distance = (raw_normalized - normalized).norm(dim=-1)
    return X, clipping_distance


class _REMBOAcquisition(AcquisitionFunction):
    """Evaluate an original-space acquisition through a fixed random embedding."""

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

    def project(self, Z: Tensor) -> tuple[Tensor, Tensor]:
        """Map embedded points to the original box and return clipping distance."""
        return _project_to_original_box(Z, self.embedding, self.original_bounds)

    def forward(self, Z: Tensor) -> Tensor:
        X, _ = self.project(Z)
        return self.acq_function(X)


class REMBOStrategy(SearchStrategy):
    """Optimize an original-space acquisition through a fixed random embedding.

    REMBO reduces only the acquisition search space. It does not transform or
    replace the surrogate model, so the acquisition function always receives
    candidates in the public/original input space. Embedded coordinates are
    projected into a normalized ``[-1, 1]^d`` box before being unnormalized to
    the configured public bounds, making the embedding independent of input units.
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
        constraints: CandidateConstraints | None = None,
    ) -> None:
        super().__init__(bounds)
        reject_unmapped_candidate_constraints(
            constraints,
            strategy_name=self.__class__.__name__,
        )
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
        embedding = torch.randn(
            self.input_dim,
            embedding_dim,
            dtype=bounds.dtype,
            device=bounds.device,
            generator=generator,
        )
        embedding = embedding / embedding.norm(dim=0, keepdim=True).clamp_min(
            torch.finfo(bounds.dtype).eps
        )
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
        """Dimension optimized by REMBO."""
        return self.embedding.shape[-1]

    def project(self, Z: Tensor) -> Tensor:
        """Project embedded coordinates to feasible original-space candidates."""
        X, _ = _project_to_original_box(Z, self.embedding, self.bounds)
        return X

    def optimize(
        self,
        acq_function: AcquisitionFunction,
        *,
        q: int = 1,
    ) -> SearchResult:
        """Optimize ``acq_function`` in the fixed embedded search space."""
        if q < 1:
            raise ValueError("q must be at least 1.")

        embedded_acq = _REMBOAcquisition(acq_function, self.embedding, self.bounds)
        embedded_candidates, _ = optimize_acqf(
            acq_function=embedded_acq,
            bounds=self.embedded_bounds,
            q=q,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
            options=self.options,
            sequential=self.sequential,
        )

        candidates, clipping_distance = embedded_acq.project(embedded_candidates)
        with torch.no_grad():
            acquisition_value = acq_function(candidates).reshape(())

        return SearchResult(
            candidates=candidates,
            acquisition_value=acquisition_value,
            metadata={
                "embedded_candidates": embedded_candidates.detach(),
                "clipping_distance": clipping_distance.detach(),
                "embedding": self.embedding.detach().clone(),
            },
        )
