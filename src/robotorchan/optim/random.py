"""Random candidate search for acquisition functions."""

from __future__ import annotations

from botorch.acquisition.acquisition import AcquisitionFunction
from torch import Tensor

from robotorchan.optim.backends.sampling import optimize_acqf_sampling
from robotorchan.optim.base import SearchResult, SearchStrategy


class RandomSearchStrategy(SearchStrategy):
    """Optimize an acquisition function over uniformly sampled candidate batches.

    This strategy is intended as a simple high-dimensional acquisition-search
    baseline. It samples ``num_samples`` independent q-batches uniformly inside
    the configured public-space box, evaluates the acquisition function once per
    batch, and returns the batch with the largest acquisition value. The true
    objective is never evaluated by the strategy.

    Args:
        bounds: Continuous box bounds with shape ``[2, d]``.
        num_samples: Number of random q-batches evaluated per search.
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
        """Return the highest-acquisition random q-batch in public input space."""
        if q < 1:
            raise ValueError("q must be at least 1.")

        candidates, acquisition_value = optimize_acqf_sampling(
            acq_function,
            self.bounds,
            q,
            num_samples=self.num_samples,
            method="random",
            seed=self.seed,
        )

        return SearchResult(
            candidates=candidates,
            acquisition_value=acquisition_value,
            metadata={"num_samples": self.num_samples, "q": q, "sampler": "random"},
        )

