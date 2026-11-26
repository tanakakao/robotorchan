"""Enumerate evaluation masks and delegate candidate optimization to BoTorch."""

from __future__ import annotations

from collections.abc import Callable
from itertools import product
from typing import TYPE_CHECKING, Any

import torch
from botorch.acquisition.acquisition import AcquisitionFunction, OneShotAcquisitionFunction
from torch import Tensor

from robotorchan.optim.dispatch import optimize_acqf

if TYPE_CHECKING:
    from robotorchan.optim.constraints.contracts import CandidateConstraints
    from robotorchan.workflow.cost_budget import CostBudget


class _InputCostAcquisition(AcquisitionFunction):
    """Apply deterministic predicted cost to an ordinary masked acquisition."""

    def __init__(
        self,
        acquisition: AcquisitionFunction,
        cost_function: Callable[[Tensor, Tensor], Tensor],
        mask: Tensor,
    ) -> None:
        super().__init__(model=acquisition.model)
        self.acquisition = acquisition
        self.cost_function = cost_function
        self.register_buffer("mask", mask)

    def forward(self, X: Tensor) -> Tensor:
        value = self.acquisition(X)
        cost = self.cost_function(X, self.mask)
        if cost.shape != value.shape:
            raise ValueError("cost_function must return one cost per acquisition batch")
        if not torch.isfinite(cost).all() or (cost <= 0).any():
            raise ValueError("predicted costs must be finite and positive")
        return torch.where(value >= 0, value / cost, value * cost)


def optimize_acqf_decoupled(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    evaluation_masks: Tensor,
    q: int = 1,
    *,
    num_restarts: int = 10,
    raw_samples: int = 256,
    options: dict[str, Any] | None = None,
    batch_initial_conditions: Tensor | None = None,
    output_costs: Tensor | None = None,
    mask_costs: Tensor | None = None,
    budget: CostBudget | None = None,
    cost_function: Callable[[Tensor, Tensor], Tensor] | None = None,
    constraints: CandidateConstraints | None = None,
    fixed_features: dict[int, float | Tensor] | None = None,
    max_mask_assignments: int = 256,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return candidate, selected evaluation mask, and acquisition value.

    Enumerate ordered mask assignments for joint q-batches. One-shot
    auxiliary candidates are managed by BoTorch, never included in the mask.
    The original acquisition mask is restored even if optimization fails.
    Fixed costs rank nonnegative gains by gain / cost and negative gains by
    gain * cost. The returned value is the ranking score when costs are given.
    Budget filtering does not reserve or charge the ledger.
    Differentiable cost_function(X, mask) returns a positive cost per t-batch.
    Input-dependent budgets require explicit feasible initial conditions.
    """
    if q < 1 or max_mask_assignments < 1:
        raise ValueError("q and max_mask_assignments must be positive")
    if cost_function is not None and budget is not None:
        if q != 1 or constraints is not None or fixed_features:
            raise NotImplementedError("predicted budgets with batch or other constraints")
    if not hasattr(acq_function, "X_evaluation_mask"):
        raise TypeError("acquisition must support X_evaluation_mask")
    if (
        evaluation_masks.ndim != 2
        or evaluation_masks.shape[0] == 0
        or evaluation_masks.shape[1] != acq_function.model.num_outputs
        or evaluation_masks.dtype != torch.bool
        or not evaluation_masks.any(dim=-1).all()
    ):
        raise ValueError("evaluation_masks must contain nonempty boolean output masks")
    if evaluation_masks.device != bounds.device:
        raise ValueError("evaluation_masks and bounds must share a device")
    if getattr(acq_function, "X_pending", None) is not None:
        raise NotImplementedError("pending evaluations are not supported")
    if sum(cost is not None for cost in (output_costs, mask_costs, cost_function)) > 1:
        raise ValueError("provide only one cost specification")
    if cost_function is not None:
        if isinstance(acq_function, OneShotAcquisitionFunction):
            raise NotImplementedError("use native cost-aware utility for one-shot acquisitions")
        if budget is not None and batch_initial_conditions is None:
            raise ValueError("input-dependent budget requires batch_initial_conditions")
        if getattr(acq_function, "cost_aware_utility", None) is not None:
            raise ValueError("acquisition already has a cost-aware utility")
    if output_costs is not None:
        if output_costs.shape != (evaluation_masks.shape[1],):
            raise ValueError("output_costs must have one entry per output")
        if not torch.isfinite(output_costs).all() or (output_costs <= 0).any():
            raise ValueError("output costs must be finite and positive")
        mask_costs = evaluation_masks.to(bounds) @ output_costs.to(bounds)
    if mask_costs is not None:
        if mask_costs.shape != (evaluation_masks.shape[0],):
            raise ValueError("mask_costs must have one entry per allowed mask")
        if not torch.isfinite(mask_costs).all() or (mask_costs <= 0).any():
            raise ValueError("mask costs must be finite and positive")
        mask_costs = mask_costs.to(bounds)
        if getattr(acq_function, "cost_aware_utility", None) is not None:
            raise ValueError("acquisition already has a cost-aware utility")
    if budget is not None and mask_costs is None and cost_function is None:
        raise ValueError("budget filtering requires explicit costs")
    k = evaluation_masks.shape[0]
    if k**q > max_mask_assignments:
        raise ValueError("mask assignments exceed max_mask_assignments")
    original_mask = acq_function.X_evaluation_mask
    best = None
    try:
        for assignment in product(range(k), repeat=q):
            indices = list(assignment)
            cost = None if mask_costs is None else mask_costs[indices].sum()
            if budget is not None and cost is not None and not budget.can_afford(cost.item()):
                continue
            selected = evaluation_masks[indices].clone()
            acq_function.X_evaluation_mask = selected
            objective = acq_function
            if cost_function is not None:
                objective = _InputCostAcquisition(acq_function, cost_function, selected)
            initial = batch_initial_conditions
            restarts = num_restarts
            extra = {}
            if cost_function is not None and budget is not None:
                remaining = budget.remaining

                def budget_constraint(
                    x: Tensor, remaining: float = remaining, selected: Tensor = selected
                ) -> Tensor:
                    return remaining - cost_function(x.reshape(1, 1, -1), selected).reshape(())

                with torch.no_grad():
                    estimates = cost_function(initial, selected)
                    if estimates.shape != initial.shape[:-2]:
                        raise ValueError("cost_function must return one cost per initial batch")
                    if not torch.isfinite(estimates).all() or (estimates <= 0).any():
                        raise ValueError("predicted costs must be finite and positive")
                    initial = initial[estimates <= remaining]
                if initial.shape[0] == 0:
                    continue
                restarts = initial.shape[0]
                extra["nonlinear_inequality_constraints"] = ((budget_constraint, True),)
            candidate, value = optimize_acqf(
                acq_function=objective,
                bounds=bounds,
                q=q,
                num_restarts=restarts,
                raw_samples=raw_samples,
                options=options,
                batch_initial_conditions=initial,
                constraints=constraints,
                fixed_features=fixed_features,
                **extra,
            )
            if candidate.shape != (q, bounds.shape[-1]) or value.numel() != 1:
                raise RuntimeError("optimizer returned unexpected candidate or value shape")
            if not torch.isfinite(candidate).all() or not torch.isfinite(value).all():
                raise RuntimeError("optimizer returned nonfinite candidate or value")
            if cost_function is not None and budget is not None:
                with torch.no_grad():
                    estimate = cost_function(candidate.unsqueeze(0), selected).item()
                if not budget.can_afford(estimate):
                    # SLSQP can return a point just outside the exact budget.
                    # Retain the best strictly affordable supplied restart.
                    with torch.no_grad():
                        initial_values = objective(initial)
                        index = initial_values.reshape(-1).argmax()
                        candidate = initial[index]
                        value = initial_values.reshape(-1)[index]
                        estimate = cost_function(candidate.unsqueeze(0), selected).item()
                    if not budget.can_afford(estimate) or not torch.isfinite(value):
                        continue
            if cost is not None:
                value = torch.where(value >= 0, value / cost, value * cost)
                if not torch.isfinite(value).all():
                    raise RuntimeError("cost-adjusted acquisition value is nonfinite")
            if best is None or value.item() > best[2].item():
                best = (candidate.detach().clone(), selected, value.detach().clone())
    finally:
        acq_function.X_evaluation_mask = original_mask
    if best is None:
        raise ValueError("no affordable evaluation mask")
    return best
