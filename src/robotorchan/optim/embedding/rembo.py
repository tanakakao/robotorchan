"""Random embedding Bayesian optimization (REMBO) search strategy."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.optim import optimize_acqf
from torch import Tensor

from robotorchan.optim.base import SearchResult, SearchStrategy


class _REMBOAcquisition(AcquisitionFunction):
    """Evaluate an original-space acquisition through a fixed random embedding."""

    def __init__(
        self,
        acq_function: AcquisitionFunction,
        embedding: Tensor,
        center: Tensor,
        bounds: Tensor,
    ) -> None:
        super().__init__(model=acq_function.model)
        self.acq_function = acq_function
        self.register_buffer("embedding", embedding.detach().clone())
        self.register_buffer("center", center.detach().clone())
        self.register_buffer("original_bounds", bounds.detach().clone())

    def project(self, Z: Tensor) -> tuple[Tensor, Tensor]:
        """Map embedded points to the original box and return clipping distance."""
        raw_X = self.center + Z @ self.embedding.transpose(-2, -1)
        X = torch.maximum(raw_X, self.original_bounds[0])
        X = torch.minimum(X, self.original_bounds[1])
        clipping_distance = (raw_X - X).norm(dim=-1)
        return X, clipping_distance

    def forward(self, Z: Tensor) -> Tensor:
        X, _ = self.project(Z)
        return self.acq_function(X)


class REMBOStrategy(SearchStrategy):
    """Optimize an original-space acquisition through a fixed random embedding.

    REMBO reduces only the acquisition search space. It does not transform or
    replace the surrogate model, so the acquisition function always receives
    candidates in the public/original input space.

    Args:
        bounds: Original-space continuous box bounds with shape ``[2, d]``.
        embedding_dim: Dimension of the random embedded search space.
        embedded_bound: Symmetric embedded box ``[-b, b]^k``.
        seed: Seed used only to construct the fixed Gaussian embedding.
        num_restarts: Number of multistart acquisition optimization restarts.
        raw_samples: Number of raw samples used to initialize restarts.
        options: Optional options forwarded to ``optimize_acqf``.
        sequential: Whether to optimize a q-batch sequentially.
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

        generator = torch.Generator(device=bounds.device)
        if seed is not None:
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
        self.center = bounds.mean(dim=0)
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
        acquisition = _ProjectionOnly(self.embedding, self.center, self.bounds)
        return acquisition.project(Z)

    def optimize(
        self,
        acq_function: AcquisitionFunction,
        *,
        q: int = 1,
    ) -> SearchResult:
        """Optimize ``acq_function`` in the fixed embedded search space."""
        if q < 1:
            raise ValueError("q must be at least 1.")

        embedded_acq = _REMBOAcquisition(
            acq_function,
            self.embedding,
            self.center,
            self.bounds,
        )
        start = perf_counter()
        embedded_candidates, _ = optimize_acqf(
            acq_function=embedded_acq,
            bounds=self.embedded_bounds,
            q=q,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
            options=self.options,
            sequential=self.sequential,
        )
        elapsed = perf_counter() - start

        candidates, clipping_distance = embedded_acq.project(embedded_candidates)
        with torch.no_grad():
            acquisition_value = acq_function(candidates)

        return SearchResult(
            candidates=candidates,
            acquisition_value=acquisition_value,
            optimization_time=elapsed,
            metadata={
                "embedded_candidates": embedded_candidates.detach(),
                "clipping_distance": clipping_distance.detach(),
                "embedding": self.embedding.detach().clone(),
            },
        )


class _ProjectionOnly:
    """Small helper used by the public projection method without an acquisition."""

    def __init__(self, embedding: Tensor, center: Tensor, bounds: Tensor) -> None:
        self.embedding = embedding
        self.center = center
        self.bounds = bounds

    def project(self, Z: Tensor) -> Tensor:
        if Z.shape[-1] != self.embedding.shape[-1]:
            raise ValueError("Z last dimension must equal embedding_dim.")
        raw_X = self.center + Z @ self.embedding.transpose(-2, -1)
        X = torch.maximum(raw_X, self.bounds[0])
        return torch.minimum(X, self.bounds[1])
