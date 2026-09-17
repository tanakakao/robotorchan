"""Acquisition optimization in the public/original input space."""

from __future__ import annotations

from typing import Any

from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.optim import optimize_acqf
from torch import Tensor

from robotorchan.optim.base import SearchResult, SearchStrategy


class OriginalSpaceStrategy(SearchStrategy):
    """Optimize a BoTorch acquisition function directly in original space.

    This strategy is a thin robotorchan adapter around ``botorch.optim.optimize_acqf``.
    It does not transform candidates or inspect the surrogate model. Therefore the
    acquisition function sees exactly the same public input space as a direct
    ``optimize_acqf`` call.

    Args:
        bounds: Continuous box bounds with shape ``[2, d]``.
        num_restarts: Number of multistart optimization restarts.
        raw_samples: Number of raw samples used to initialize the restarts.
        options: Optional optimizer options forwarded to ``optimize_acqf``.
        sequential: Whether to optimize a q-batch sequentially.
    """

    def __init__(
        self,
        bounds: Tensor,
        *,
        num_restarts: int = 10,
        raw_samples: int = 512,
        options: dict[str, Any] | None = None,
        sequential: bool = False,
    ) -> None:
        super().__init__(bounds)
        if num_restarts < 1:
            raise ValueError("num_restarts must be at least 1.")
        if raw_samples < 1:
            raise ValueError("raw_samples must be at least 1.")
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples
        self.options = None if options is None else dict(options)
        self.sequential = sequential

    def optimize(
        self,
        acq_function: AcquisitionFunction,
        *,
        q: int = 1,
    ) -> SearchResult:
        """Optimize ``acq_function`` within the configured original-space bounds."""
        if q < 1:
            raise ValueError("q must be at least 1.")

        candidates, acquisition_value = optimize_acqf(
            acq_function=acq_function,
            bounds=self.bounds,
            q=q,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
            options=self.options,
            sequential=self.sequential,
        )

        return SearchResult(
            candidates=candidates,
            acquisition_value=acquisition_value,
            metadata={},
        )
