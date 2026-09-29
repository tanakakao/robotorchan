"""Acquisition optimization in the public/original input space."""

from __future__ import annotations

from typing import Any

from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.optim.initializers import TGenInitialConditions
from torch import Tensor

from robotorchan.optim.backends import optimize_acqf_botorch
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
        raw_samples: Number of raw samples used to initialize the restarts. May be
            ``None`` when ``batch_initial_conditions`` are provided or a custom
            ``ic_generator`` supplies the restart points.
        options: Optional optimizer options forwarded to ``optimize_acqf``.
        sequential: Whether to optimize a q-batch sequentially. For ``q > 1``,
            BoTorch greedily solves ``q`` single-candidate problems and generates
            fresh initial conditions for each step.
        constraints: Optional candidate-space constraints using BoTorch-native
            optimizer contracts.
        fixed_features: Optional feature values fixed during optimization. This is
            suitable for target-fidelity optimization without changing the public
            candidate coordinates.
        batch_initial_conditions: Optional BoTorch initial conditions with shape
            ``[num_restarts, q, d]``. BoTorch requires feasible initial conditions
            when nonlinear inequality constraints are used. For joint optimization,
            the standard shape is ``[num_restarts, q, d]``. BoTorch does not reuse
            these conditions across greedy steps when ``sequential=True`` and ``q > 1``.
        ic_generator: Optional BoTorch-compatible initial-condition generator. This is
            required for nonlinear constraints when explicit initial conditions are
            not supplied.
        ic_gen_kwargs: Optional keyword arguments for ``ic_generator``.
    """

    def __init__(
        self,
        bounds: Tensor,
        *,
        num_restarts: int = 10,
        raw_samples: int | None = 512,
        options: dict[str, Any] | None = None,
        sequential: bool = False,
        constraints: CandidateConstraints | None = None,
        fixed_features: dict[int, float] | None = None,
        batch_initial_conditions: Tensor | None = None,
        ic_generator: TGenInitialConditions | None = None,
        ic_gen_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(bounds)
        if num_restarts < 1:
            raise ValueError("num_restarts must be at least 1.")
        if raw_samples is not None and raw_samples < 1:
            raise ValueError("raw_samples must be at least 1 when provided.")
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples
        self.options = None if options is None else dict(options)
        self.sequential = sequential
        self.constraints = constraints or CandidateConstraints()
        self.fixed_features = None if fixed_features is None else dict(fixed_features)
        self.batch_initial_conditions = batch_initial_conditions
        self.ic_generator = ic_generator
        self.ic_gen_kwargs = None if ic_gen_kwargs is None else dict(ic_gen_kwargs)

    def optimize(
        self,
        acq_function: AcquisitionFunction,
        *,
        q: int = 1,
    ) -> SearchResult:
        """Optimize ``acq_function`` within the configured original-space bounds."""
        if q < 1:
            raise ValueError("q must be at least 1.")
        candidates, acquisition_value = optimize_acqf_botorch(
            acq_function=acq_function,
            bounds=self.bounds,
            q=q,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
            options=self.options,
            sequential=self.sequential,
            constraints=self.constraints,
            fixed_features=self.fixed_features,
            batch_initial_conditions=self.batch_initial_conditions,
            ic_generator=self.ic_generator,
            ic_gen_kwargs=self.ic_gen_kwargs,
        )

        return SearchResult(
            candidates=candidates,
            acquisition_value=acquisition_value,
            metadata={"optimizer": "botorch"},
        )
