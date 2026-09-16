"""Random candidate search for acquisition functions."""

from __future__ import annotations

from time import perf_counter

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from torch import Tensor

from robotorchan.optim.base import SearchResult, SearchStrategy


class RandomSearchStrategy(SearchStrategy):
    """Optimize an acquisition function over uniformly sampled candidates.

    This strategy is intended as a simple high-dimensional acquisition-search
    baseline. It samples points uniformly inside the configured public-space box,
    evaluates the acquisition function on those points, and returns the best
    candidates. The true objective is never evaluated by the strategy.

    Args:
        bounds: Continuous box bounds with shape ``[2, d]``.
        num_samples: Number of random public-space points evaluated per search.
        seed: Optional local random seed. A dedicated generator is used so the
            strategy does not modify PyTorch's global random-number state.
    """

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
        """Return the highest-acquisition random points in public input space."""
        if q < 1:
            raise ValueError("q must be at least 1.")
        if q > self.num_samples:
            raise ValueError("q must not exceed num_samples.")

        start = perf_counter()
        samples = self._sample_candidates()
        with torch.no_grad():
            values = acq_function(samples.unsqueeze(-2))
        scores = self._as_point_scores(values)
        selected = torch.topk(scores, k=q, largest=True, sorted=True).indices
        candidates = samples[selected]
        acquisition_value = scores[selected]
        elapsed = perf_counter() - start

        return SearchResult(
            candidates=candidates,
            acquisition_value=acquisition_value,
            optimization_time=elapsed,
            metadata={"num_samples": self.num_samples},
        )

    def _sample_candidates(self) -> Tensor:
        generator = None
        if self.seed is not None:
            generator = torch.Generator(device=self.bounds.device)
            generator.manual_seed(self.seed)
        unit = torch.rand(
            self.num_samples,
            self.input_dim,
            dtype=self.bounds.dtype,
            device=self.bounds.device,
            generator=generator,
        )
        return self.bounds[0] + (self.bounds[1] - self.bounds[0]) * unit

    def _as_point_scores(self, values: Tensor) -> Tensor:
        scores = values.squeeze()
        if scores.ndim != 1 or scores.shape[0] != self.num_samples:
            raise ValueError(
                "RandomSearchStrategy requires an acquisition function that returns "
                "one scalar value per q=1 candidate."
            )
        return scores
