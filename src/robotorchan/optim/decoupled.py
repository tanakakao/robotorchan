"""Enumerate evaluation masks and delegate candidate optimization to BoTorch."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from itertools import product
from random import Random
from typing import TYPE_CHECKING, Any, Literal

import torch
from botorch.acquisition.acquisition import AcquisitionFunction, OneShotAcquisitionFunction
from torch import Tensor

from robotorchan.optim.dispatch import OptimizerName, optimize_acqf
from robotorchan.optim.one_shot.mixed import optimize_mixed_one_shot_acqf

if TYPE_CHECKING:
    from robotorchan.optim.constraints.contracts import CandidateConstraints
    from robotorchan.optim.domains.variable_space import MixedVariableSpace
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
    variable_space: MixedVariableSpace | None = None,
    max_discrete_assignments: int = 256,
    strategy: Literal["joint", "sequential", "heuristic"] = "joint",
    optimizer: OptimizerName = "botorch",
    optimizer_options: dict[str, Any] | None = None,
    seed: int | None = None,
    mask_constraint: Callable[[Tensor], bool] | None = None,
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
    mask_constraint filters complete joint masks before candidate optimization.
    """
    if q < 1 or max_mask_assignments < 1 or max_discrete_assignments < 1:
        raise ValueError("q and assignment limits must be positive")
    if (
        cost_function is not None
        and budget is not None
        and (constraints is not None or fixed_features)
    ):
        raise NotImplementedError("predicted budgets with other candidate constraints")
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
    if strategy not in ("joint", "sequential", "heuristic"):
        raise ValueError("unknown decoupled strategy")
    assignment_count = k**q if strategy == "joint" else k
    if strategy != "heuristic" and assignment_count > max_mask_assignments:
        raise ValueError("mask assignments exceed max_mask_assignments")
    if isinstance(acq_function, OneShotAcquisitionFunction) and optimizer != "botorch":
        raise NotImplementedError("one-shot acquisitions require the BoTorch backend")
    if cost_function is not None and budget is not None and optimizer != "botorch":
        raise NotImplementedError("predicted hard budgets require the BoTorch backend")
    discrete = {}
    if variable_space is not None:
        if not torch.equal(variable_space.bounds, bounds):
            raise ValueError("variable_space bounds must match bounds")
        variable_space.validate_fixed_features(fixed_features)
        discrete = dict(variable_space.categorical_values or {})
        for dim in variable_space.integer_dims if optimizer == "botorch" else ():
            lower = int(bounds[0, dim].ceil().item())
            upper = int(bounds[1, dim].floor().item())
            discrete[dim] = tuple(range(lower, upper + 1))
    if discrete and optimizer == "botorch":
        if cost_function is not None and budget is not None:
            raise NotImplementedError("mixed predicted hard budgets are not supported yet")
        joint_mixed = q > 1 and strategy != "sequential"
        if (
            (joint_mixed or isinstance(acq_function, OneShotAcquisitionFunction))
            and optimizer_options is not None
        ):
            raise NotImplementedError(
                "mixed joint or one-shot constraints and custom starts or options"
            )
        count = 1
        for values in discrete.values():
            count *= len(values)
        if count > max_discrete_assignments:
            raise ValueError("discrete assignments exceed max_discrete_assignments")
    if strategy == "sequential" and q > 1:
        if (
            isinstance(acq_function, OneShotAcquisitionFunction)
            or constraints is not None
            or mask_constraint is not None
            or batch_initial_conditions is not None
            or cost_function is not None
            or getattr(acq_function, "cost_aware_utility", None) is not None
        ):
            raise NotImplementedError(
                "sequential requires ordinary acquisition without constraints, starts or costs"
            )
        return _optimize_sequential(
            acq_function,
            bounds,
            evaluation_masks,
            q,
            mask_costs=mask_costs,
            budget=budget,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            options=options,
            fixed_features=fixed_features,
            variable_space=variable_space,
            max_discrete_assignments=max_discrete_assignments,
            optimizer=optimizer,
            optimizer_options=optimizer_options,
            seed=seed,
        )
    original_mask = acq_function.X_evaluation_mask
    best = None
    try:
        assignments: Iterable[tuple[int, ...]] = product(range(k), repeat=q)
        if strategy == "heuristic":
            assignments = _heuristic_assignments(k, q, max_mask_assignments, seed, mask_costs)
        for assignment in assignments:
            indices = list(assignment)
            cost = None if mask_costs is None else mask_costs[indices].sum()
            if budget is not None and cost is not None and not budget.can_afford(cost.item()):
                continue
            selected = evaluation_masks[indices].clone()
            if mask_constraint is not None and not mask_constraint(selected):
                continue
            acq_function.X_evaluation_mask = selected
            objective = acq_function
            if cost_function is not None:
                objective = _InputCostAcquisition(acq_function, cost_function, selected)
            initial = batch_initial_conditions
            restarts = num_restarts
            extra = {}
            if optimizer_options is not None:
                extra["optimizer_options"] = optimizer_options
            if seed is not None and optimizer != "botorch":
                extra["seed"] = seed
            if optimizer != "botorch":
                extra["optimizer"] = optimizer
                extra["variable_space"] = variable_space
            if cost_function is not None and budget is not None:
                remaining = budget.remaining

                def budget_constraint(
                    x: Tensor, remaining: float = remaining, selected: Tensor = selected
                ) -> Tensor:
                    batch = x.reshape(1, selected.shape[0], -1)
                    return remaining - cost_function(batch, selected).reshape(())

                with torch.no_grad():
                    if initial.ndim != 3 or initial.shape[1:] != (q, bounds.shape[-1]):
                        raise ValueError("initial conditions must have shape [restarts, q, d]")
                    estimates = cost_function(initial, selected)
                    if estimates.shape != initial.shape[:-2]:
                        raise ValueError("cost_function must return one cost per initial batch")
                    if not torch.isfinite(estimates).all() or (estimates <= 0).any():
                        raise ValueError("predicted costs must be finite and positive")
                    initial = initial[estimates <= remaining]
                if initial.shape[0] == 0:
                    continue
                restarts = initial.shape[0]
                extra["nonlinear_inequality_constraints"] = ((budget_constraint, q == 1),)
            one_shot = isinstance(objective, OneShotAcquisitionFunction)
            if optimizer == "botorch" and (
                (discrete and (q > 1 or one_shot))
                or (one_shot and (constraints is not None or fixed_features))
            ):
                candidate, value = optimize_mixed_one_shot_acqf(
                    objective,
                    bounds,
                    categorical_features=discrete,
                    q=q,
                    num_restarts=restarts,
                    raw_samples=raw_samples,
                    max_assignments=max_discrete_assignments,
                    constraints=constraints,
                    batch_initial_conditions=initial,
                    options=options,
                    fixed_features=fixed_features,
                )
            else:
                if discrete and optimizer == "botorch":
                    if optimizer_options is not None:
                        raise ValueError("mixed BoTorch mask optimization owns fixed_features_list")
                    dims = tuple(discrete)
                    extra["variable_space"] = variable_space
                    extra["optimizer_options"] = {
                        "fixed_features_list": [
                            dict(zip(dims, values, strict=True))
                            for values in product(*(discrete[dim] for dim in dims))
                        ]
                    }
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
        raise ValueError("no affordable evaluation mask satisfying the requested constraints")
    return best


class _PrefixAcquisition(AcquisitionFunction):
    """Evaluate the full joint acquisition with a fixed selected prefix."""

    def __init__(self, acquisition: AcquisitionFunction, prefix: Tensor) -> None:
        super().__init__(model=acquisition.model)
        self.acquisition = acquisition
        self.register_buffer("prefix", prefix)

    def forward(self, X: Tensor) -> Tensor:
        prefix = self.prefix.expand(*X.shape[:-2], *self.prefix.shape)
        return self.acquisition(torch.cat((prefix, X), dim=-2))


def _optimize_sequential(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    evaluation_masks: Tensor,
    q: int,
    *,
    mask_costs: Tensor | None,
    budget: CostBudget | None,
    **optimizer_kwargs: Any,
) -> tuple[Tensor, Tensor, Tensor]:
    original = acq_function.X_evaluation_mask
    prefix = bounds.new_empty((0, bounds.shape[-1]))
    selected = evaluation_masks[:0]
    spent = 0.0
    try:
        for step in range(q):
            best = None
            for index, mask in enumerate(evaluation_masks):
                cost = 0.0 if mask_costs is None else mask_costs[index].item()
                total = spent + cost
                if budget is not None:
                    minimum = mask_costs.min().item()
                    if not budget.can_afford(total + (q - step - 1) * minimum):
                        continue
                trial_mask = torch.cat((selected, mask.unsqueeze(0)), dim=0)
                acq_function.X_evaluation_mask = trial_mask
                objective = _PrefixAcquisition(acq_function, prefix)
                # Reuse q=1 dispatch with a local mask; the prefix wrapper
                # evaluates the original acquisition with the full mask.
                objective.X_evaluation_mask = mask.unsqueeze(0)
                candidate, _, value = optimize_acqf_decoupled(
                    objective, bounds, mask.unsqueeze(0), **optimizer_kwargs
                )
                if mask_costs is not None:
                    value = torch.where(value >= 0, value / total, value * total)
                if best is None or value.item() > best[2].item():
                    best = (candidate, trial_mask, value, total)
            if best is None:
                raise ValueError("no affordable sequential evaluation mask")
            candidate, selected, value, spent = best
            prefix = torch.cat((prefix, candidate), dim=0)
        return prefix, selected, value
    finally:
        acq_function.X_evaluation_mask = original


def _heuristic_assignments(
    k: int, q: int, limit: int, seed: int | None, costs: Tensor | None
) -> Iterable[tuple[int, ...]]:
    """Bounded joint search over distinct mask assignments."""
    if k**q <= limit:
        return product(range(k), repeat=q)
    rng = Random(seed)
    cheapest = 0 if costs is None else int(costs.argmin().item())
    selected = {(cheapest,) * q: None}
    for index in range(k):
        if len(selected) == limit:
            break
        selected[(index,) * q] = None
    while len(selected) < limit:
        selected[tuple(rng.randrange(k) for _ in range(q))] = None
    return iter(selected)
