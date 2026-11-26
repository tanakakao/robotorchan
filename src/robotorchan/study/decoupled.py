"""Connect partial Study observations to existing decoupled acquisitions."""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from torch import Tensor

from robotorchan.optim.decoupled import optimize_acqf_decoupled
from robotorchan.study.trial import TrialState
from robotorchan.workflow.cost_budget import CostBudget


def create_decoupled_proposer(
    *,
    parameter_names: Sequence[str],
    output_names: Sequence[str],
    evaluation_masks: Tensor,
    model_factory: Callable[..., Any],
    acquisition_factory: Callable[..., Any],
    optimizer_kwargs: Mapping[str, Any] | None = None,
    allow_pending: bool = False,
    output_costs: Mapping[str, float] | None = None,
    total_budget: float | None = None,
) -> Callable[..., Any]:
    """Build a proposer using output-specific training data and existing optimization.

    Factories receive named training pairs and a fitted model, respectively.
    The model factory owns fitting and initial data for outputs without observations.
    Pending evaluations are rejected by default; opt-in ignores pending information.
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

    def propose(history, bounds, *, q, fixed_features):
        if not allow_pending and any(t.state is TrialState.RUNNING for t in history):
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
            ledger = CostBudget(total_budget)
            for trial in history:
                for name in trial.observations:
                    ledger.record(costs[name])
            for trial in history:
                if trial.state is TrialState.RUNNING:
                    for name in trial.evaluation_mask:
                        if name not in trial.observations:
                            ledger.reserve(f"{trial.number}:{name}", costs[name])
            if ledger.exhausted:
                raise ValueError("Evaluation budget exhausted.")
            call_options["output_costs"] = bounds.new_tensor([costs[n] for n in outputs])
            call_options["budget"] = ledger.remaining
        model = model_factory(data)
        acq = acquisition_factory(model)
        return optimize_acqf_decoupled(
            acq,
            bounds,
            masks.to(device=bounds.device),
            q=q,
            fixed_features=fixed_features,
            **call_options,
        )

    return propose
