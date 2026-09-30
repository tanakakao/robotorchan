"""Stateful trust-region Bayesian optimization (TuRBO) search strategy."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from torch import Tensor
from torch.quasirandom import SobolEngine

from robotorchan.acquisition.sampling import select_thompson_candidates
from robotorchan.optim.base import SearchResult, SearchStrategy
from robotorchan.optim.dispatch import OptimizerName, optimize_acqf


@dataclass(frozen=True)
class TuRBOState:
    """Persistent state controlling the TuRBO trust-region length."""

    dim: int = 1
    batch_size: int = 1
    length: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    success_counter: int = 0
    failure_counter: int = 0
    success_tolerance: int = 3
    failure_tolerance: int | None = None
    best_value: float = float("-inf")
    restart_triggered: bool = False

    def __post_init__(self) -> None:
        if self.dim < 1:
            raise ValueError("dim must be at least 1.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        if self.failure_tolerance is None:
            failure_tolerance = math.ceil(max(4.0 / self.batch_size, self.dim / self.batch_size))
            object.__setattr__(self, "failure_tolerance", failure_tolerance)
        if not 0 < self.length_min <= self.length_max:
            raise ValueError("TuRBO lengths must satisfy 0 < length_min <= length_max.")
        if self.length <= 0 or self.length > self.length_max:
            raise ValueError("length must satisfy 0 < length <= length_max.")
        if self.length < self.length_min and not self.restart_triggered:
            raise ValueError("length below length_min requires restart_triggered=True.")
        if self.success_tolerance < 1:
            raise ValueError("success_tolerance must be at least 1.")
        if self.failure_tolerance is None or self.failure_tolerance < 1:
            raise ValueError("failure_tolerance must be at least 1.")


def _is_turbo_improvement(
    candidate_best: float,
    best_value: float,
    *,
    relative_improvement: float,
) -> bool:
    """Return whether an observation is a numerically meaningful improvement."""
    if not math.isfinite(best_value):
        return True
    threshold = relative_improvement * max(1.0, abs(best_value))
    return candidate_best > best_value + threshold


def update_turbo_state(
    state: TuRBOState,
    values: Tensor,
    *,
    relative_improvement: float = 1e-3,
) -> TuRBOState:
    """Return the next TuRBO state after observing objective values."""
    if values.numel() < 1:
        raise ValueError("values must contain at least one observation.")
    if relative_improvement < 0:
        raise ValueError("relative_improvement must be non-negative.")

    candidate_best = float(values.max().item())
    success = _is_turbo_improvement(
        candidate_best,
        state.best_value,
        relative_improvement=relative_improvement,
    )

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


def _normalized_dimension_weights(
    dimension_weights: Tensor | None,
    *,
    dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    """Return positive dimension weights with geometric mean one."""
    if dimension_weights is None:
        return torch.ones(dim, dtype=dtype, device=device)
    weights = dimension_weights.to(dtype=dtype, device=device).reshape(-1)
    if weights.shape != (dim,):
        raise ValueError("dimension_weights must have shape [dim].")
    if not torch.all(torch.isfinite(weights)):
        raise ValueError("dimension_weights must be finite.")
    if torch.any(weights <= 0):
        raise ValueError("dimension_weights must be strictly positive.")
    log_weights = torch.log(weights)
    return torch.exp(log_weights - log_weights.mean())


def turbo_trust_region_bounds(
    center: Tensor,
    bounds: Tensor,
    *,
    length: float,
    dimension_weights: Tensor | None = None,
) -> Tensor:
    """Return clipped TuRBO bounds in the public input space.

    Length is interpreted relative to each global input range. Optional
    dimension weights are normalized to geometric mean one, matching TuRBO's
    ARD geometry while keeping this function independent from model internals.
    """
    if bounds.ndim != 2 or bounds.shape[0] != 2:
        raise ValueError("bounds must have shape [2, dim].")
    if center.shape != (bounds.shape[-1],):
        raise ValueError("center must have shape [dim].")
    if not math.isfinite(length) or length <= 0:
        raise ValueError("length must be finite and positive.")
    if torch.any(bounds[0] >= bounds[1]):
        raise ValueError("bounds must satisfy lower < upper in every dimension.")
    if not torch.all(torch.isfinite(center)) or not torch.all(torch.isfinite(bounds)):
        raise ValueError("center and bounds must be finite.")

    center = center.to(dtype=bounds.dtype, device=bounds.device)
    if torch.any(center < bounds[0]) or torch.any(center > bounds[1]):
        raise ValueError("center must lie inside bounds.")

    weights = _normalized_dimension_weights(
        dimension_weights,
        dim=bounds.shape[-1],
        dtype=bounds.dtype,
        device=bounds.device,
    )
    half_width = 0.5 * length * weights * (bounds[1] - bounds[0])
    lower = torch.maximum(center - half_width, bounds[0])
    upper = torch.minimum(center + half_width, bounds[1])
    return torch.stack([lower, upper])


def generate_turbo_thompson_choices(
    center: Tensor,
    trust_region_bounds: Tensor,
    *,
    n_candidates: int,
    seed: int | None = None,
    perturbation_probability: float | None = None,
) -> Tensor:
    """Generate a TuRBO Thompson-sampling candidate pool inside local bounds."""
    if trust_region_bounds.ndim != 2 or trust_region_bounds.shape[0] != 2:
        raise ValueError("trust_region_bounds must have shape [2, dim].")
    dim = trust_region_bounds.shape[-1]
    if center.shape != (dim,):
        raise ValueError("center must have shape [dim].")
    if n_candidates < 1:
        raise ValueError("n_candidates must be at least 1.")
    center = center.to(
        dtype=trust_region_bounds.dtype,
        device=trust_region_bounds.device,
    )
    if torch.any(center < trust_region_bounds[0]) or torch.any(center > trust_region_bounds[1]):
        raise ValueError("center must lie inside trust_region_bounds.")

    probability = min(20.0 / dim, 1.0)
    if perturbation_probability is not None:
        probability = perturbation_probability
    if not 0.0 < probability <= 1.0:
        raise ValueError("perturbation_probability must satisfy 0 < p <= 1.")

    sobol = SobolEngine(dimension=dim, scramble=True, seed=seed)
    perturbations = sobol.draw(n_candidates).to(
        dtype=trust_region_bounds.dtype,
        device=trust_region_bounds.device,
    )
    perturbations = trust_region_bounds[0] + (
        trust_region_bounds[1] - trust_region_bounds[0]
    ) * perturbations

    generator = torch.Generator(device=trust_region_bounds.device)
    if seed is not None:
        generator.manual_seed(seed)
    mask = torch.rand(
        n_candidates,
        dim,
        dtype=trust_region_bounds.dtype,
        device=trust_region_bounds.device,
        generator=generator,
    ) <= probability
    empty_rows = torch.where(mask.sum(dim=1) == 0)[0]
    if empty_rows.numel() > 0:
        forced_dims = torch.randint(
            dim,
            (empty_rows.numel(),),
            device=trust_region_bounds.device,
            generator=generator,
        )
        mask[empty_rows, forced_dims] = True

    choices = center.expand(n_candidates, dim).clone()
    choices[mask] = perturbations[mask]
    return choices


class TuRBOStrategy(SearchStrategy):
    """Optimize an acquisition function inside a stateful TuRBO trust region.

    The incumbent is maintained explicitly in public/original input space. This
    keeps the strategy independent from the surrogate's internal representation,
    including reduced-space surrogate models.
    """

    def __init__(
        self,
        bounds: Tensor,
        *,
        center: Tensor,
        state: TuRBOState | None = None,
        num_restarts: int = 10,
        raw_samples: int = 512,
        options: dict[str, Any] | None = None,
        sequential: bool = False,
        dimension_weights: Tensor | None = None,
        optimizer: OptimizerName = "botorch",
        seed: int | None = None,
        optimizer_options: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(bounds)
        if num_restarts < 1:
            raise ValueError("num_restarts must be at least 1.")
        if raw_samples < 1:
            raise ValueError("raw_samples must be at least 1.")
        self.center = self._validate_center(center)
        self.state = TuRBOState(dim=self.input_dim, batch_size=1) if state is None else state
        if self.state.dim != self.input_dim:
            raise ValueError("state.dim must match the strategy input dimension.")
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples
        self.options = None if options is None else dict(options)
        self.sequential = sequential
        self.optimizer = optimizer
        self.seed = seed
        self.optimizer_options = None if optimizer_options is None else dict(optimizer_options)
        self.dimension_weights = _normalized_dimension_weights(
            dimension_weights,
            dim=self.input_dim,
            dtype=self.bounds.dtype,
            device=self.bounds.device,
        )

    def _validate_center(self, center: Tensor) -> Tensor:
        center = center.to(dtype=self.bounds.dtype, device=self.bounds.device)
        if center.shape != (self.input_dim,):
            raise ValueError("center must have shape [input_dim].")
        if torch.any(center < self.bounds[0]) or torch.any(center > self.bounds[1]):
            raise ValueError("center must lie inside bounds.")
        return center.detach().clone()

    def trust_region_bounds(self, center: Tensor | None = None) -> Tensor:
        """Return feasible trust-region bounds around the current incumbent."""
        current_center = self.center if center is None else self._validate_center(center)
        return turbo_trust_region_bounds(
            current_center,
            self.bounds,
            length=self.state.length,
            dimension_weights=self.dimension_weights,
        )

    def update_state(
        self,
        values: Tensor,
        *,
        candidates: Tensor | None = None,
        relative_improvement: float = 1e-3,
    ) -> TuRBOState:
        """Update state and move the incumbent to the best improving candidate."""
        previous_best = self.state.best_value
        next_state = update_turbo_state(
            self.state,
            values,
            relative_improvement=relative_improvement,
        )
        if candidates is not None:
            if candidates.ndim != 2 or candidates.shape != (values.numel(), self.input_dim):
                raise ValueError("candidates must have shape [values.numel(), input_dim].")
            best_index = values.reshape(-1).argmax()
            candidate_best = float(values.reshape(-1)[best_index].item())
            if _is_turbo_improvement(
                candidate_best,
                previous_best,
                relative_improvement=relative_improvement,
            ):
                self.center = self._validate_center(candidates[best_index])
        self.state = next_state
        return self.state


    def thompson_sample(
        self,
        model: Any,
        *,
        q: int = 1,
        n_candidates: int | None = None,
        perturbation_probability: float | None = None,
        objective: Any | None = None,
    ) -> SearchResult:
        """Select TuRBO candidates by posterior sampling over a local Sobol pool."""
        if q < 1:
            raise ValueError("q must be at least 1.")
        if self.state.restart_triggered:
            raise RuntimeError("TuRBO restart is required before further candidate generation.")
        candidate_count = (
            min(5000, max(2000, 200 * self.input_dim))
            if n_candidates is None
            else n_candidates
        )
        if candidate_count < q:
            raise ValueError("n_candidates must be at least q.")

        trust_bounds = self.trust_region_bounds()
        choices = generate_turbo_thompson_choices(
            self.center,
            trust_bounds,
            n_candidates=candidate_count,
            seed=self.seed,
            perturbation_probability=perturbation_probability,
        )
        candidates = select_thompson_candidates(
            model,
            choices,
            num_samples=q,
            replacement=False,
            objective=objective,
        )
        return SearchResult(
            candidates=candidates,
            acquisition_value=None,
            metadata={
                "trust_region_center": self.center.detach().clone(),
                "trust_region_bounds": trust_bounds.detach(),
                "trust_region_length": self.state.length,
                "success_counter": self.state.success_counter,
                "failure_counter": self.state.failure_counter,
                "candidate_generation": "thompson",
                "n_candidates": candidate_count,
            },
        )

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

        trust_bounds = self.trust_region_bounds()
        candidates, acquisition_value = optimize_acqf(
            acq_function=acq_function,
            bounds=trust_bounds,
            q=q,
            optimizer=self.optimizer,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
            options=self.options,
            sequential=self.sequential,
            seed=self.seed,
            optimizer_options=self.optimizer_options,
        )

        return SearchResult(
            candidates=candidates,
            acquisition_value=acquisition_value,
            metadata={
                "trust_region_center": self.center.detach().clone(),
                "trust_region_bounds": trust_bounds.detach(),
                "trust_region_length": self.state.length,
                "success_counter": self.state.success_counter,
                "failure_counter": self.state.failure_counter,
                "optimizer": self.optimizer,
            },
        )
