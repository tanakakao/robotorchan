"""Random candidate search for acquisition functions."""

from __future__ import annotations

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from torch import Tensor

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
        if q < 1:
            raise ValueError("q must be at least 1.")

        samples = self._sample_candidate_batches(q)
        with torch.no_grad():
            values = acq_function(samples)
        scores = self._as_batch_scores(values)
        selected = scores.argmax()

        return SearchResult(
            candidates=samples[selected],
            acquisition_value=scores[selected],
            metadata={"num_samples": self.num_samples, "q": q},
        )

    def _sample_candidate_batches(self, q: int) -> Tensor:
        generator = None
        if self.seed is not None:
            generator = torch.Generator(device=self.bounds.device)
            generator.manual_seed(self.seed)
        unit = torch.rand(
            self.num_samples,
            q,
            self.input_dim,
            dtype=self.bounds.dtype,
            device=self.bounds.device,
            generator=generator,
        )
        return self.bounds[0] + (self.bounds[1] - self.bounds[0]) * unit

    def _as_batch_scores(self, values: Tensor) -> Tensor:
        if values.numel() != self.num_samples:
            raise ValueError(
                "RandomSearchStrategy requires an acquisition function that returns "
                "one scalar value per sampled q-batch."
            )
        return values.reshape(self.num_samples)
