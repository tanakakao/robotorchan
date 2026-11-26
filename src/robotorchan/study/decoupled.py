"""Connect partial Study observations to existing decoupled acquisitions."""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from torch import Tensor

from robotorchan.acquisition import create_acquisition
from robotorchan.optim.decoupled import optimize_acqf_decoupled
from robotorchan.study.trial import TrialState
from robotorchan.workflow.cost_budget import CostBudget


def create_decoupled_proposer(
    *,
    parameter_names: Sequence[str],
    output_names: Sequence[str],
    evaluation_masks: Tensor,
    model_factory: Callable[..., Any],
    acquisition_factory: str | Callable[..., Any],
    acquisition_kwargs: Mapping[str, Any] | None = None,
    optimizer_kwargs: Mapping[str, Any] | None = None,
    allow_pending: bool = False,
    pending_fantasies: bool = False,
    output_costs: Mapping[str, float] | None = None,
    total_budget: float | None = None,
) -> Callable[..., Any]:
    """Build a proposer using output-specific training data and existing optimization.

    Factories receive named training pairs and a fitted model, respectively.
    The model factory owns fitting and initial data for outputs without observations.
    Pending evaluations are rejected by default. With pending_fantasies, native
    hypervolume KG incorporates outstanding output evaluations. Otherwise opt-in
    ignores pending information.
    """
    parameters = tuple(parameter_names)
    outputs = tuple(output_names)
    options = dict(optimizer_kwargs or {})
    if not parameters or len(set(parameters)) != len(parameters):
        raise ValueError("parameter_names must be nonempty and unique.")
    if not outputs or len(set(outputs)) != len(outputs):
        raise ValueError("output_names must be nonempty and unique.")
    if (
        evaluation_masks.ndim != 2
        or evaluation_masks.shape[1] != len(outputs)
        or evaluation_masks.dtype != torch.bool
        or not len(evaluation_masks)
        or not evaluation_masks.any(dim=-1).all()
    ):
        raise ValueError("evaluation_masks must contain nonempty boolean output selections.")
    masks = evaluation_masks.detach().clone()
    if (output_costs is None) != (total_budget is None):
        raise ValueError("Specify output_costs and total_budget together.")
    costs = dict(output_costs or {})
    if total_budget is not None:
        if set(costs) != set(outputs):
            raise ValueError("Costs must cover exactly the configured outputs.")
        if any(not torch.isfinite(torch.tensor(v)) or v <= 0 for v in costs.values()):
            raise ValueError("Output costs must be positive and finite.")
        CostBudget(total_budget)
        if {"output_costs", "mask_costs", "cost_function", "budget"} & options.keys():
            raise ValueError("Budget configuration conflicts with optimizer_kwargs.")

    def budget_from_history(history):
        ledger = CostBudget(total_budget)
        for trial in history:
            for name in set(trial.observations) | set(trial.actual_costs):
                ledger.record(
                    trial.actual_costs[name]
                    if name in trial.actual_costs
                    else costs[name]
                )
        for trial in history:
            if trial.state is TrialState.RUNNING:
                for name in trial.evaluation_mask:
                    if name not in trial.observations:
                        ledger.reserve(f"{trial.number}:{name}", estimate(trial.params, name))
        return ledger

    def check_retry_budget(history, remaining):
        if total_budget is not None and not budget_from_history(history).can_afford(
            sum(costs[name] for name in remaining)
        ):
            raise ValueError("Insufficient evaluation budget for retry.")

    def propose(history, bounds, *, q, fixed_features):
        pending = [t for t in history if t.state is TrialState.RUNNING]
        if not (allow_pending or pending_fantasies) and pending:
            raise ValueError("Pending evaluations require an explicit pending policy.")
        data = {}
        for name in outputs:
            observed = [trial for trial in history if name in trial.observations]
            X = bounds.new_tensor(
                [[trial.params[key] for key in parameters] for trial in observed]
            ).reshape(-1, len(parameters))
            Y = bounds.new_tensor([trial.observations[name] for trial in observed]).reshape(-1, 1)
            data[name] = (X, Y)
        call_options = dict(options)
        if total_budget is not None:
            ledger = budget_from_history(history)
            if ledger.exhausted:
                raise ValueError("Evaluation budget exhausted.")
            call_options["output_costs"] = bounds.new_tensor([costs[n] for n in outputs])
            call_options["budget"] = ledger
        model = model_factory(data)
        kwargs = dict(acquisition_kwargs or {})
        if isinstance(acquisition_factory, str):
            acq = create_acquisition(acquisition_factory, model=model, **kwargs)
        else:
            acq = acquisition_factory(model, **kwargs)
        if pending_fantasies and pending:
            pending_X = bounds.new_tensor(
                [[trial.params[key] for key in parameters] for trial in pending]
            )
            pending_mask = torch.tensor(
                [
                    [name in t.evaluation_mask and name not in t.observations for name in outputs]
                    for t in pending
                ],
                dtype=torch.bool,
                device=bounds.device,
            )
            acq.set_X_pending(pending_X, X_pending_evaluation_mask=pending_mask)
        return optimize_acqf_decoupled(
            acq,
            bounds,
            masks.to(device=bounds.device),
            q=q,
            fixed_features=fixed_features,
            **call_options,
        )

    propose.check_retry_budget = check_retry_budget
    return propose
