"""Stateful trust-region Bayesian optimization (TuRBO) search strategy."""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.optim import optimize_acqf
from torch import Tensor

from robotorchan.optim.base import SearchResult, SearchStrategy


@dataclass(frozen=True)
class TuRBOState:
    """Persistent state controlling the TuRBO trust-region length."""

    length: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    success_counter: int = 0
    failure_counter: int = 0
    success_tolerance: int = 3
    failure_tolerance: int = 4
    best_value: float = float("-inf")
    restart_triggered: bool = False

    def __post_init__(self) -> None:
        if not 0 < self.length_min <= self.length_max:
            raise ValueError("TuRBO lengths must satisfy 0 < length_min <= length_max.")
        if self.length <= 0 or self.length > self.length_max:
            raise ValueError("length must satisfy 0 < length <= length_max.")
        if self.length < self.length_min and not self.restart_triggered:
            raise ValueError("length below length_min requires restart_triggered=True.")
        if self.success_tolerance < 1:
            raise ValueError("success_tolerance must be at least 1.")
        if self.failure_tolerance < 1:
            raise ValueError("failure_tolerance must be at least 1.")


def update_turbo_state(
    state: TuRBOState,
    values: Tensor,
    *,
    relative_improvement: float = 1e-3,
) -> TuRBOState:
    """Return the next TuRBO state after observing objective values.

    The state is immutable so callers explicitly retain the returned state. An
    improvement is measured against the best objective value seen previously.
    """
    if values.numel() < 1:
        raise ValueError("values must contain at least one observation.")
    if relative_improvement < 0:
        raise ValueError("relative_improvement must be non-negative.")

    candidate_best = float(values.max().item())
    threshold = relative_improvement * max(1.0, abs(state.best_value))
    success = candidate_best > state.best_value + threshold

    success_counter = state.success_counter + 1 if success else 0
    failure_counter = 0 if success else state.failure_counter + 1
    length = state.length

    if success_counter >= state.success_tolerance:
        length = min(2.0 * length, state.length_max)
        success_counter = 0
    elif failure_counter >= state.failure_tolerance:
        length = length / 2.0
        failure_counter = 0

    return replace(
        state,
        length=length,
        success_counter=success_counter,
        failure_counter=failure_counter,
        best_value=max(state.best_value, candidate_best),
        restart_triggered=length < state.length_min,
    )


class TuRBOStrategy(SearchStrategy):
    """Optimize an acquisition function inside a stateful TuRBO trust region.

    The strategy owns the geometric search state but never evaluates the true
    objective. After evaluating returned candidates, callers update the state
    with :meth:`update_state` and the observed objective values.

    Args:
        bounds: Public/original-space box bounds with shape ``[2, d]``.
        state: Initial TuRBO state. A default state is created when omitted.
        num_restarts: Number of acquisition optimization restarts.
        raw_samples: Number of raw samples used to initialize restarts.
        options: Optional options forwarded to ``optimize_acqf``.
        sequential: Whether to optimize a q-batch sequentially.
    """

    def __init__(
        self,
        bounds: Tensor,
        *,
        state: TuRBOState | None = None,
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
        self.state = TuRBOState() if state is None else state
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples
        self.options = None if options is None else dict(options)
        self.sequential = sequential

    def trust_region_bounds(self, center: Tensor) -> Tensor:
        """Return feasible trust-region bounds centered on ``center``."""
        if center.shape != (self.input_dim,):
            raise ValueError("center must have shape [input_dim].")
        if torch.any(center < self.bounds[0]) or torch.any(center > self.bounds[1]):
            raise ValueError("center must lie inside bounds.")

        half_width = 0.5 * self.state.length * (self.bounds[1] - self.bounds[0])
        lower = torch.maximum(center - half_width, self.bounds[0])
        upper = torch.minimum(center + half_width, self.bounds[1])
        return torch.stack([lower, upper])

    def update_state(
        self,
        values: Tensor,
        *,
        relative_improvement: float = 1e-3,
    ) -> TuRBOState:
        """Update and return the persistent state from observed objective values."""
        self.state = update_turbo_state(
            self.state,
            values,
            relative_improvement=relative_improvement,
        )
        return self.state

    def optimize(
        self,
        acq_function: AcquisitionFunction,
        *,
        q: int = 1,
    ) -> SearchResult:
        """Optimize the acquisition inside the current trust region."""
        if q < 1:
            raise ValueError("q must be at least 1.")
        if self.state.restart_triggered:
            raise RuntimeError("TuRBO restart is required before further optimization.")

        center = self._select_center(acq_function)
        trust_bounds = self.trust_region_bounds(center)
        start = perf_counter()
        candidates, acquisition_value = optimize_acqf(
            acq_function=acq_function,
            bounds=trust_bounds,
            q=q,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
            options=self.options,
            sequential=self.sequential,
        )
        elapsed = perf_counter() - start

        return SearchResult(
            candidates=candidates,
            acquisition_value=acquisition_value,
            optimization_time=elapsed,
            metadata={
                "trust_region_center": center.detach(),
                "trust_region_bounds": trust_bounds.detach(),
                "trust_region_length": self.state.length,
                "success_counter": self.state.success_counter,
                "failure_counter": self.state.failure_counter,
            },
        )

    def _select_center(self, acq_function: AcquisitionFunction) -> Tensor:
        model = getattr(acq_function, "model", None)
        train_inputs = getattr(model, "train_inputs", None)
        train_targets = getattr(model, "train_targets", None)
        if not train_inputs or train_targets is None:
            raise ValueError("TuRBOStrategy requires an acquisition model with training data.")

        train_X = train_inputs[0]
        if train_X.ndim != 2 or train_X.shape[-1] != self.input_dim:
            raise ValueError("TuRBOStrategy requires original-space two-dimensional train_X.")
        scores = train_targets.reshape(train_X.shape[0], -1).mean(dim=-1)
        center = train_X[scores.argmax()]
        return center.detach().to(dtype=self.bounds.dtype, device=self.bounds.device)
