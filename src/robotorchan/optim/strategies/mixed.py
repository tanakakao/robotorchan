"""Acquisition optimization over enumerated mixed-variable assignments."""

from __future__ import annotations

from typing import Any

from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.optim.initializers import TGenInitialConditions
from torch import Tensor

from robotorchan.optim.backends.botorch import optimize_acqf_mixed_botorch
from robotorchan.optim.base import SearchResult, SearchStrategy
from robotorchan.optim.constraints.contracts import CandidateConstraints


class MixedSpaceStrategy(SearchStrategy):
    """Optimize over continuous variables and enumerated discrete assignments.

    Initial-condition generation remains BoTorch-native. ``fixed_features_list``
    enumerates the discrete assignments, so categorical coordinates are fixed
    during each continuous subproblem rather than relaxed into free variables.
    For q > 1, BoTorch uses greedy sequential mixed optimization and regenerates
    initial conditions for each q=1 step. Nonlinear constraints therefore need
    an ``ic_generator`` for this sequential path.
    """

    def __init__(
        self,
        bounds: Tensor,
        *,
        fixed_features_list: list[dict[int, float]],
        num_restarts: int = 10,
        raw_samples: int | None = 512,
        options: dict[str, Any] | None = None,
        constraints: CandidateConstraints | None = None,
        batch_initial_conditions: Tensor | None = None,
        ic_generator: TGenInitialConditions | None = None,
        ic_gen_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(bounds)
        if not fixed_features_list:
            raise ValueError("fixed_features_list must not be empty.")
        if num_restarts < 1:
            raise ValueError("num_restarts must be at least 1.")
        if raw_samples is not None and raw_samples < 1:
            raise ValueError("raw_samples must be at least 1 when provided.")
        self.fixed_features_list = [dict(features) for features in fixed_features_list]
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples
        self.options = None if options is None else dict(options)
        self.constraints = constraints or CandidateConstraints()
        self.batch_initial_conditions = batch_initial_conditions
        self.ic_generator = ic_generator
        self.ic_gen_kwargs = None if ic_gen_kwargs is None else dict(ic_gen_kwargs)

    def optimize(
        self,
        acq_function: AcquisitionFunction,
        *,
        q: int = 1,
    ) -> SearchResult:
        """Optimize the acquisition over the configured mixed search space."""
        if q < 1:
            raise ValueError("q must be at least 1.")
        candidates, acquisition_value = optimize_acqf_mixed_botorch(
            acq_function=acq_function,
            bounds=self.bounds,
            q=q,
            num_restarts=self.num_restarts,
            fixed_features_list=self.fixed_features_list,
            raw_samples=self.raw_samples,
            options=self.options,
            constraints=self.constraints,
            batch_initial_conditions=self.batch_initial_conditions,
            ic_generator=self.ic_generator,
            ic_gen_kwargs=self.ic_gen_kwargs,
        )

        return SearchResult(
            candidates=candidates,
            acquisition_value=acquisition_value,
            metadata={"optimizer": "botorch_mixed"},
        )
