"""Sequential ask/tell workflow for conservative safe Bayesian optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from torch import Tensor

from robotorchan.optim.safe import optimize_safe_acqf
from robotorchan.optim.safe_expansion import propose_safe_expansion
from robotorchan.semantics.safety import SafetySpecification

SafeSelectionMode = Literal["objective", "expansion"]


@dataclass
class SafeBOWorkflow:
    """Coordinate safe candidate selection and observed safety feedback.

    The caller fits and updates the objective acquisition and safety model
    between ask/tell iterations. This workflow does not retrain GP models.
    """

    specification: SafetySpecification
    bounds: Tensor

    def __post_init__(self) -> None:
        if self.bounds.ndim != 2 or self.bounds.shape != (2, self.specification.initial_X.shape[1]):
            raise ValueError("bounds must have shape [2, d] matching initial_X.")
        if not bool(
            (self.specification.initial_X >= self.bounds[0]).all()
            and (self.specification.initial_X <= self.bounds[1]).all()
        ):
            raise ValueError("Observed safe seeds must lie within bounds.")
        self._observed_X = self.specification.initial_X.clone()
        self._observed_constraints = self.specification.observed_constraints.clone()
        self._pending: Tensor | None = None

    @property
    def observed_X(self) -> Tensor:
        return self._observed_X.clone()

    @property
    def observed_constraints(self) -> Tensor:
        return self._observed_constraints.clone()

    def ask(
        self,
        safety_model: object,
        *,
        mode: SafeSelectionMode = "objective",
        acq_function: AcquisitionFunction | None = None,
        num_samples: int = 1024,
        seed: int | None = None,
    ) -> Tensor:
        """Propose one posterior-safe candidate; requires tell before next ask."""
        if self._pending is not None:
            raise RuntimeError("Call tell() before requesting another candidate.")
        kwargs = {
            "beta": self.specification.beta,
            "num_samples": num_samples,
            "seed": seed,
        }
        if mode == "objective":
            if acq_function is None:
                raise ValueError("acq_function is required for objective selection.")
            candidate, _ = optimize_safe_acqf(
                acq_function, safety_model, self.bounds, self.specification.constraints, **kwargs
            )
        elif mode == "expansion":
            candidate, _ = propose_safe_expansion(
                safety_model, self.bounds, self.specification.constraints, **kwargs
            )
        else:
            raise ValueError("mode must be 'objective' or 'expansion'.")
        self._pending = candidate.detach().clone()
        return candidate.clone()

    def tell(self, candidate: Tensor, observed_constraints: Tensor) -> None:
        """Record the real safety observation, including unsafe outcomes.

        Unsafe outcomes are retained for model updates; a GP prediction is not
        a guarantee of actual safety.
        """
        if self._pending is None:
            raise RuntimeError("Call ask() before tell().")
        if candidate.shape != self._pending.shape or not torch.equal(candidate, self._pending):
            raise ValueError("candidate must match the pending ask() candidate.")
        if observed_constraints.shape != (1, len(self.specification.constraints)):
            raise ValueError("observed_constraints must have shape [1, m].")
        if not observed_constraints.is_floating_point() or not bool(
            torch.isfinite(observed_constraints).all()
        ):
            raise ValueError("observed_constraints must be finite floating values.")
        self._observed_X = torch.cat((self._observed_X, candidate.detach()), dim=0)
        self._observed_constraints = torch.cat(
            (self._observed_constraints, observed_constraints.detach()), dim=0
        )
        self._pending = None
