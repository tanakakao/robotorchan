"""Random candidate search for acquisition functions."""

from __future__ import annotations

from botorch.acquisition.acquisition import AcquisitionFunction
from torch import Tensor

from robotorchan.optim.backends.sampling import (
    sample_candidate_batches,
    select_best_sampled_batch,
)
from robotorchan.optim.base import SearchResult, SearchStrategy


class RandomSearchStrategy(SearchStrategy):
    """Optimize an acquisition function over uniformly sampled candidate batches."""

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
        """Return the highest-acquisition random q-batch in public input space."""
        samples = self._sample_candidate_batches(q)
        candidates, acquisition_value = select_best_sampled_batch(
            acq_function,
            samples,
        )
        return SearchResult(
            candidates=candidates,
            acquisition_value=acquisition_value,
            metadata={"num_samples": self.num_samples, "q": q},
        )

    def _sample_candidate_batches(self, q: int) -> Tensor:
        """Draw random q-batches through the shared sampling backend."""
        return sample_candidate_batches(
            self.bounds,
            self.num_samples,
            q,
            method="random",
            seed=self.seed,
        )
