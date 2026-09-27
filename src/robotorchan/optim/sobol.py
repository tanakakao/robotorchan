"""Sobol candidate search for acquisition functions."""

from __future__ import annotations

from botorch.acquisition.acquisition import AcquisitionFunction
from torch import Tensor

from robotorchan.optim.backends.sampling import optimize_acqf_sampling
from robotorchan.optim.base import SearchResult, SearchStrategy


class SobolSearchStrategy(SearchStrategy):
    """Optimize an acquisition function over scrambled Sobol q-batches."""

    def __init__(
        self,
        bounds: Tensor,
        *,
        num_samples: int = 4096,
        seed: int | None = None,
    ) -> None:
        super().__init__(bounds)
        if num_samples < 1:
            raise ValueError("num_samples must be at least 1.")
        self.num_samples = num_samples
        self.seed = seed

    def optimize(
        self,
        acq_function: AcquisitionFunction,
        *,
        q: int = 1,
    ) -> SearchResult:
        """Return the highest-acquisition Sobol q-batch in public input space."""
        candidates, acquisition_value = optimize_acqf_sampling(
            acq_function,
            self.bounds,
            q,
            num_samples=self.num_samples,
            method="sobol",
            seed=self.seed,
        )
        return SearchResult(
            candidates=candidates,
            acquisition_value=acquisition_value,
            metadata={"num_samples": self.num_samples, "q": q, "sampler": "sobol"},
        )
