"""Acquisition optimization over enumerated mixed-variable assignments."""

from __future__ import annotations

from typing import Any

from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.optim import optimize_acqf_mixed
from torch import Tensor

from robotorchan.optim.base import SearchResult, SearchStrategy
from robotorchan.optim.constraints import CandidateConstraints


class MixedSpaceStrategy(SearchStrategy):
    """Optimize over continuous variables and enumerated discrete assignments."""

    def __init__(
        self,
        bounds: Tensor,
        *,
        fixed_features_list: list[dict[int, float]],
        num_restarts: int = 10,
        raw_samples: int = 512,
        options: dict[str, Any] | None = None,
        constraints: CandidateConstraints | None = None,
    ) -> None:
        super().__init__(bounds)
        if not fixed_features_list:
            raise ValueError("fixed_features_list must not be empty.")
        if num_restarts < 1:
            raise ValueError("num_restarts must be at least 1.")
        if raw_samples < 1:
            raise ValueError("raw_samples must be at least 1.")
        self.fixed_features_list = [dict(features) for features in fixed_features_list]
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples
        self.options = None if options is None else dict(options)
        self.constraints = constraints or CandidateConstraints()

    def optimize(
        self,
        acq_function: AcquisitionFunction,
        *,
        q: int = 1,
    ) -> SearchResult:
        """Optimize the acquisition over the configured mixed search space."""
        if q < 1:
            raise ValueError("q must be at least 1.")

        candidates, acquisition_value = optimize_acqf_mixed(
            acq_function=acq_function,
            bounds=self.bounds,
            q=q,
            num_restarts=self.num_restarts,
            fixed_features_list=self.fixed_features_list,
            raw_samples=self.raw_samples,
            options=self.options,
            inequality_constraints=(
                list(self.constraints.inequality_constraints)
                if self.constraints.inequality_constraints
                else None
            ),
            equality_constraints=(
                list(self.constraints.equality_constraints)
                if self.constraints.equality_constraints
                else None
            ),
        )
        return SearchResult(
            candidates=candidates,
            acquisition_value=acquisition_value,
            metadata={},
        )
