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
from robotorchan.optim.dispatch import optimize_acqf
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
        self.direction = direction
        self.model = model
        self.model_kwargs = dict(model_kwargs or {})
        self.acquisition = acquisition
        self.acquisition_kwargs = dict(acquisition_kwargs or {})
        self.optimizer = optimizer
        self.optimizer_kwargs = dict(optimizer_kwargs or {})
        self.n_initial_trials = n_initial_trials
        self._sobol = SobolEngine(len(space.bounds), scramble=True, seed=seed)

    def _bounds(self) -> Tensor:
        return torch.tensor(list(self.space.bounds.values()), dtype=torch.double).T

    def propose(self, completed: list[Trial]) -> dict[str, float]:
        """Generate a single candidate, preserving declared parameter order."""
        bounds = self._bounds()
        if len(completed) < self.n_initial_trials:
            point = self._sobol.draw(1).to(dtype=bounds.dtype)
            candidate = bounds[0] + (bounds[1] - bounds[0]) * point
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
                if not torch.allclose(self.optimizer.bounds.to(dtype=bounds.dtype), bounds):
                    raise ValueError("SearchStrategy bounds must match Study bounds.")
                if self.optimizer_kwargs:
                    raise ValueError("optimizer_kwargs cannot accompany SearchStrategy.")
                candidate = self.optimizer.optimize(acq, q=1).candidates
            else:
                options = dict(self.optimizer_kwargs)
                options.setdefault("num_restarts", 5)
                options.setdefault("raw_samples", 64)
                candidate, _ = optimize_acqf(
                    acq, bounds, q=1, optimizer=self.optimizer, **options
                )
        if candidate.shape != (1, len(self.space.bounds)):
            raise ValueError("Optimizer must return candidates with shape [1, d].")
        point = candidate[0].detach().tolist()
        return dict(zip(self.space.bounds, point, strict=True))
