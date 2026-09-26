"""Acquisition optimization in the public/original input space."""

from __future__ import annotations

from typing import Any

from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.optim import optimize_acqf
from torch import Tensor

from robotorchan.optim.base import SearchResult, SearchStrategy
from robotorchan.optim.constraints import CandidateConstraints


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
        constraints: Optional candidate-space constraints using BoTorch-native
            optimizer contracts.
        fixed_features: Optional feature values fixed during optimization. This is
            suitable for target-fidelity optimization without changing the public
            candidate coordinates.
        batch_initial_conditions: Optional BoTorch initial conditions with shape
            ``[num_restarts, q, d]``. BoTorch requires feasible initial conditions
            when nonlinear inequality constraints are used.
    """

    def __init__(
        self,
        bounds: Tensor,
        *,
        num_restarts: int = 10,
        raw_samples: int = 512,
        options: dict[str, Any] | None = None,
        sequential: bool = False,
        constraints: CandidateConstraints | None = None,
        fixed_features: dict[int, float] | None = None,
        batch_initial_conditions: Tensor | None = None,
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
        self.constraints = constraints or CandidateConstraints()
        self.fixed_features = None if fixed_features is None else dict(fixed_features)
        self.batch_initial_conditions = batch_initial_conditions

    def optimize(
        self,
        acq_function: AcquisitionFunction,
        *,
        q: int = 1,
    ) -> SearchResult:
        """Optimize ``acq_function`` within the configured original-space bounds."""
        if q < 1:
            raise ValueError("q must be at least 1.")
        if self.constraints.has_nonlinear_constraints and self.batch_initial_conditions is None:
            raise ValueError(
                "Nonlinear candidate constraints require feasible "
                "batch_initial_conditions."
            )

        candidates, acquisition_value = optimize_acqf(
            acq_function=acq_function,
            bounds=self.bounds,
            q=q,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
            options=self.options,
            sequential=self.sequential,
            fixed_features=self.fixed_features,
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
            batch_initial_conditions=self.batch_initial_conditions,
            nonlinear_inequality_constraints=(
                list(self.constraints.nonlinear_inequality_constraints)
                if self.constraints.nonlinear_inequality_constraints
                else None
            ),
        )

        return SearchResult(
            candidates=candidates,
            acquisition_value=acquisition_value,
            metadata={},
        )
