"""Acquisition optimization in a fitted latent input space."""

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
from robotorchan.optim.latent.reconstruction import LatentReconstruction


class _ReconstructedAcquisition(AcquisitionFunction):
    """Evaluate an original-space acquisition through latent reconstruction."""

    def __init__(
        self,
        acq_function: AcquisitionFunction,
        reconstruction: LatentReconstruction,
        bounds: Tensor,
    ) -> None:
        super().__init__(model=acq_function.model)
        self.acq_function = acq_function
        self.reconstruction = reconstruction
        self.register_buffer("original_bounds", bounds.detach().clone())

    def forward(self, Z: Tensor) -> Tensor:
        X = self.reconstruction.reconstruct(Z)
        X = torch.maximum(X, self.original_bounds[0])
        X = torch.minimum(X, self.original_bounds[1])
        return self.acq_function(X)


class LatentSpaceStrategy(SearchStrategy):
    """Optimize an acquisition function through a fitted latent reconstruction.

    The acquisition function keeps its original/public input contract. Latent
    points are reconstructed and clamped to the original box before every
    acquisition evaluation. The selected candidate is therefore always returned
    in the original input space.
    """

    def __init__(
        self,
        bounds: Tensor,
        reconstruction: LatentReconstruction,
        *,
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
        if reconstruction.input_dim != self.input_dim:
            raise ValueError(
                "Reconstruction input dimension must match the original bounds dimension."
            )
        if num_restarts < 1:
            raise ValueError("num_restarts must be at least 1.")
        if raw_samples < 1:
            raise ValueError("raw_samples must be at least 1.")
        self.reconstruction = reconstruction
        self.latent_bounds = reconstruction.transform_bounds(self.bounds)
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples
        self.options = None if options is None else dict(options)
        self.sequential = sequential

    @property
    def latent_dim(self) -> int:
        """Dimension optimized by the latent strategy."""
        return self.reconstruction.latent_dim

    def optimize(
        self,
        acq_function: AcquisitionFunction,
        *,
        q: int = 1,
    ) -> SearchResult:
        if q < 1:
            raise ValueError("q must be at least 1.")

        latent_acq = _ReconstructedAcquisition(
            acq_function=acq_function,
            reconstruction=self.reconstruction,
            bounds=self.bounds,
        )
        latent_candidates, _ = optimize_acqf(
            acq_function=latent_acq,
            bounds=self.latent_bounds,
            q=q,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
            options=self.options,
            sequential=self.sequential,
        )

        reconstructed = self.reconstruction.reconstruct(latent_candidates)
        candidates = torch.maximum(reconstructed, self.bounds[0])
        candidates = torch.minimum(candidates, self.bounds[1])
        with torch.no_grad():
            acquisition_value = acq_function(candidates).reshape(())

        violation = (reconstructed - candidates).norm(dim=-1)
        projected_latent = self.reconstruction.transform(candidates)
        latent_distortion = (projected_latent - latent_candidates).norm(dim=-1)

        return SearchResult(
            candidates=candidates,
            acquisition_value=acquisition_value,
            metadata={
                "latent_candidates": latent_candidates.detach(),
                "reconstructed_candidates": reconstructed.detach(),
                "feasibility_projection_distance": violation.detach(),
                "latent_round_trip_distance": latent_distortion.detach(),
            },
        )
