"""Single-objective continuous candidate generation for Study."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from inspect import signature
from typing import Any

import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from torch import Tensor
from torch.quasirandom import SobolEngine

from robotorchan.models.standard.single_task import SingleTaskGP
from robotorchan.optim.base import SearchStrategy
from robotorchan.optim.constraints.contracts import CandidateConstraints
from robotorchan.optim.dispatch import optimize_acqf
from robotorchan.optim.domains.variable_space import MixedVariableSpace
from robotorchan.study.feasibility import is_feasible
from robotorchan.study.search_space import SearchSpace
from robotorchan.study.trial import Trial


class CandidateGenerator:
    """Connect Study history to existing models, acquisitions, and optimizers."""

    def __init__(
        self,
        space: SearchSpace,
        *,
        direction: str = "maximize",
        model: Callable[..., Any] = SingleTaskGP,
        model_kwargs: Mapping[str, Any] | None = None,
        acquisition: Callable[..., Any] = qLogExpectedImprovement,
        acquisition_kwargs: Mapping[str, Any] | None = None,
        optimizer: str | SearchStrategy = "botorch",
        optimizer_kwargs: Mapping[str, Any] | None = None,
        n_initial_trials: int = 3,
        seed: int | None = None,
        variable_space: MixedVariableSpace | None = None,
        constraints: CandidateConstraints | None = None,
    ) -> None:
        if direction not in {"maximize", "minimize"}:
            raise ValueError("direction must be 'maximize' or 'minimize'.")
        if n_initial_trials < 1:
            raise ValueError("n_initial_trials must be at least 1.")
        if not callable(model) or not callable(acquisition):
            raise TypeError("model and acquisition must be callable.")
        if not isinstance(optimizer, (str, SearchStrategy)):
            raise TypeError("optimizer must be a backend name or SearchStrategy.")
        self.space = space
        if variable_space is not None:
            bounds = torch.tensor(list(space.bounds.values()), dtype=torch.double).T
            if not torch.allclose(variable_space.bounds.to(dtype=bounds.dtype), bounds):
                raise ValueError("variable_space bounds must match Study bounds.")
        self.variable_space = variable_space
        self.constraints = constraints
        self.direction = direction
        self.model = model
        self.model_kwargs = dict(model_kwargs or {})
        self.acquisition = acquisition
        self.acquisition_kwargs = dict(acquisition_kwargs or {})
        self.optimizer = optimizer
        self.optimizer_kwargs = dict(optimizer_kwargs or {})
        self.n_initial_trials = n_initial_trials
        self._sobol = SobolEngine(len(space.bounds), scramble=True, seed=seed)
        self._random = torch.Generator()
        if seed is not None:
            self._random.manual_seed(seed)

    def _bounds(self) -> Tensor:
        return torch.tensor(list(self.space.bounds.values()), dtype=torch.double).T

    def propose(self, completed: list[Trial]) -> dict[str, float]:
        """Generate one candidate using the existing q-batch implementation."""
        return self.propose_batch(completed, q=1)[0]

    def propose_batch(
        self,
        completed: list[Trial],
        *,
        q: int,
        fixed_features: Mapping[int, float] | None = None,
    ) -> list[dict[str, float]]:
        """Generate q joint candidates from completed observations."""
        if isinstance(q, bool) or not isinstance(q, int) or q < 1:
            raise ValueError("q must be a positive integer.")
        bounds = self._bounds()
        fixed = dict(fixed_features or {})
        if any(type(dim) is not int or dim < 0 or dim >= bounds.shape[1] for dim in fixed):
            raise ValueError("Fixed feature indices must refer to Study dimensions.")
        for dim, value in fixed.items():
            if not isinstance(value, (int, float)) or not torch.isfinite(torch.tensor(value)):
                raise ValueError("Fixed feature values must be finite numbers.")
            if not bounds[0, dim] <= value <= bounds[1, dim]:
                raise ValueError("Fixed feature value is outside Study bounds.")
        if self.variable_space is not None:
            self.variable_space.validate_fixed_features(fixed)
        if len(completed) < self.n_initial_trials:
            candidate = None
            for _ in range(2048):
                points = (
                    torch.rand(
                        q, len(self.space.bounds), generator=self._random, dtype=bounds.dtype
                    )
                    if self.constraints is not None and self.constraints.has_constraints
                    else self._sobol.draw(q).to(dtype=bounds.dtype)
                )
                proposal = bounds[0] + (bounds[1] - bounds[0]) * points
                if self.variable_space is not None:
                    for dim in self.variable_space.integer_dims:
                        proposal[:, dim] = proposal[:, dim].round()
                    for dim, values in self.variable_space.categorical_values.items():
                        index = torch.clamp(
                            (points[:, dim] * len(values)).long(), max=len(values) - 1
                        )
                        proposal[:, dim] = torch.tensor(values, dtype=bounds.dtype)[index]
                for dim, value in fixed.items():
                    proposal[:, dim] = value
                if is_feasible(proposal, self.constraints, variable_space=self.variable_space):
                    candidate = proposal
                    break
            if candidate is None:
                raise RuntimeError(
                    "Unable to generate a feasible initial q-batch after 2048 attempts."
                )
        else:
            names = tuple(self.space.bounds)
            train_x = torch.tensor(
                [[trial.params[name] for name in names] for trial in completed],
                dtype=torch.double,
            )
            values = torch.tensor([[trial.value] for trial in completed], dtype=torch.double)
            train_y = values if self.direction == "maximize" else -values
            model = self.model(train_x, train_y, **self.model_kwargs)
            fit = getattr(model, "fit", None)
            if not callable(fit):
                raise TypeError("Study model must expose fit().")
            fit()
            kwargs = dict(self.acquisition_kwargs)
            parameters = signature(self.acquisition).parameters
            if "model" in parameters:
                kwargs.setdefault("model", model)
            if "best_f" in parameters:
                kwargs.setdefault("best_f", train_y.max().item())
            if "X_baseline" in parameters:
                kwargs.setdefault("X_baseline", train_x)
            acq = self.acquisition(**kwargs)
            if isinstance(self.optimizer, SearchStrategy):
                if fixed:
                    raise ValueError("Context-fixed proposals require a named optimizer backend.")
                if self.constraints is not None and self.constraints.has_constraints:
                    raise ValueError("CandidateConstraints requires a named optimizer backend.")
                if self.variable_space is not None:
                    raise ValueError("Mixed variable_space requires a named optimizer backend.")
                if not torch.allclose(self.optimizer.bounds.to(dtype=bounds.dtype), bounds):
                    raise ValueError("SearchStrategy bounds must match Study bounds.")
                if self.optimizer_kwargs:
                    raise ValueError("optimizer_kwargs cannot accompany SearchStrategy.")
                candidate = self.optimizer.optimize(acq, q=q).candidates
            else:
                options = dict(self.optimizer_kwargs)
                options.setdefault("num_restarts", 5)
                options.setdefault("raw_samples", 64)
                if fixed and "fixed_features" in options:
                    raise ValueError("optimizer_kwargs must not override context fixed features.")
                candidate, _ = optimize_acqf(
                    acq,
                    bounds,
                    q=q,
                    optimizer=self.optimizer,
                    variable_space=self.variable_space,
                    constraints=self.constraints,
                    fixed_features=fixed or None,
                    **options,
                )
        if candidate.shape != (q, len(self.space.bounds)):
            raise ValueError("Optimizer must return candidates with shape [q, d].")
        for dim, value in fixed.items():
            if not torch.allclose(
                candidate[:, dim], torch.full_like(candidate[:, dim], value), atol=1e-8, rtol=0
            ):
                raise ValueError("Optimizer did not preserve the supplied context.")
        if not is_feasible(candidate, self.constraints, variable_space=self.variable_space):
            raise ValueError("Optimizer returned candidates violating CandidateConstraints.")
        results = []
        for point in candidate.detach().tolist():
            params = dict(zip(self.space.bounds, point, strict=True))
            self.space.validate(params)
            if self.variable_space is not None:
                vector = torch.tensor(point, dtype=bounds.dtype)
                for dim in self.variable_space.integer_dims:
                    if vector[dim] != vector[dim].round():
                        raise ValueError("Optimizer returned a non-integer candidate.")
                for dim, values in self.variable_space.categorical_values.items():
                    if point[dim] not in values:
                        raise ValueError("Optimizer returned an invalid categorical candidate.")
            results.append(params)
        return results
