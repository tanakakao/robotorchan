"""Context-fixed acquisition optimization using the existing dispatcher."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from torch import Tensor

from robotorchan.optim.constraints.contracts import CandidateConstraints
from robotorchan.optim.dispatch import OptimizerName, optimize_acqf
from robotorchan.optim.domains.variable_space import MixedVariableSpace


class ContextualOptimizer:
    """Optimize decision coordinates while holding observed context fixed.

    Coordinates always refer to the full, original model input. The caller
    constructs and fits the model and acquisition function separately.
    """

    def __init__(
        self,
        *,
        decision_indices: Sequence[int],
        context_indices: Sequence[int],
        bounds: Tensor,
        optimizer: OptimizerName = "botorch",
        variable_space: MixedVariableSpace | None = None,
        constraints: CandidateConstraints | None = None,
    ) -> None:
        if bounds.ndim != 2 or bounds.shape[0] != 2 or bounds.shape[1] < 2:
            raise ValueError("bounds must have shape [2, d] with d >= 2.")
        if not bounds.is_floating_point() or not torch.isfinite(bounds).all():
            raise ValueError("bounds must be finite floating-point values.")
        if not torch.all(bounds[0] < bounds[1]):
            raise ValueError("Every lower bound must be below its upper bound.")
        decision = tuple(decision_indices)
        context = tuple(context_indices)
        dims = decision + context
        if not decision or not context or any(type(i) is not int for i in dims):
            raise ValueError("Decision and context indices must be nonempty integer sequences.")
        if len(set(dims)) != len(dims) or set(dims) != set(range(bounds.shape[1])):
            raise ValueError("Decision and context indices must partition all input dimensions.")
        if variable_space is not None and not torch.equal(variable_space.bounds, bounds):
            raise ValueError("variable_space bounds must match bounds.")
        self.decision_indices = decision
        self.context_indices = context
        self.bounds = bounds
        self.optimizer = optimizer
        self.variable_space = variable_space
        self.constraints = constraints

    def optimize(
        self,
        *,
        acq_function: AcquisitionFunction,
        context: Tensor,
        q: int = 1,
        **optimizer_kwargs: Any,
    ) -> tuple[Tensor, Tensor]:
        """Return (full-input candidates [q, d], acquisition value).

        The same context applies to every candidate in the q-batch.
        """
        if not isinstance(context, Tensor) or context.shape != (len(self.context_indices),):
            raise ValueError("context must be a tensor of shape [d_context].")
        if context.dtype != self.bounds.dtype or context.device != self.bounds.device:
            raise ValueError("context dtype and device must match bounds.")
        if not torch.isfinite(context).all():
            raise ValueError("context must contain only finite values.")
        if q < 1:
            raise ValueError("q must be positive.")
        if any(\n            key in optimizer_kwargs\n            for key in ("fixed_features", "variable_space", "constraints")\n        ):
            raise ValueError(\n                "Fixed features, variable space and constraints are owned by ContextualOptimizer."\n            )
        fixed = {index: context[pos].item() for pos, index in enumerate(self.context_indices)}
        for index, value in fixed.items():
            if value < self.bounds[0, index].item() or value > self.bounds[1, index].item():
                raise ValueError(f"Context value for dimension {index} lies outside bounds.")
        if self.variable_space is not None:
            self.variable_space.validate_fixed_features(fixed)
        candidates, value = optimize_acqf(
            acq_function=acq_function,
            bounds=self.bounds,
            q=q,
            optimizer=self.optimizer,
            fixed_features=fixed,
            variable_space=self.variable_space,
            constraints=self.constraints,
            **optimizer_kwargs,
        )
        if candidates.shape != (q, self.bounds.shape[1]):
            raise RuntimeError("Optimizer returned unexpected candidate shape.")
        # Do not silently repair a backend that fails to preserve fixed context.
        for index, fixed_value in fixed.items():
            expected = torch.as_tensor(\n                fixed_value, dtype=candidates.dtype, device=candidates.device\n            )
            if not torch.allclose(candidates[:, index], expected.expand(q), atol=1e-6, rtol=0):
                raise RuntimeError(f"Optimizer did not preserve context dimension {index}.")
        return candidates, value

    def extract_decisions(self, candidates: Tensor) -> Tensor:
        """Select decision coordinates without altering the full candidate tensor."""
        if candidates.shape[-1] != self.bounds.shape[1]:
            raise ValueError("Candidate dimension does not match bounds.")
        return candidates[..., list(self.decision_indices)]
