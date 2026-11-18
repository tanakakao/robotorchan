"""Choose a single hard intervention using existing BoTorch LogEI and optimizers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite, log
from typing import Literal

import torch
from botorch.acquisition.analytic import LogExpectedImprovement

from robotorchan.causal.intervention import HardIntervention
from robotorchan.causal.surrogate import InterventionGPSurrogate
from robotorchan.optim.dispatch import OptimizerName, optimize_acqf


@dataclass(frozen=True, slots=True)
class CausalInterventionProposal:
    """Chosen hard intervention and its comparable log-space utility."""

    intervention: HardIntervention
    log_expected_improvement: float
    intervention_cost: float
    log_utility: float


def propose_causal_intervention(
    surrogates: Sequence[InterventionGPSurrogate],
    bounds: Mapping[str, tuple[float, float]],
    *,
    direction: Literal["maximize", "minimize"] = "maximize",
    intervention_costs: Mapping[frozenset[str], float] | None = None,
    optimizer: OptimizerName = "botorch",
    num_restarts: int = 5,
    raw_samples: int = 64,
) -> CausalInterventionProposal:
    """Select the intervention set and its q=1 continuous values.

    Each surrogate must already be fitted for a different intervention set,
    on the same graph instance and outcome. A *single* incumbent from all
    interventional training outcomes is shared across the groups, so LogEI
    scores are comparable. Constant per-set costs adjust the ranking via
    log(EI / cost), without changing within-set candidate optimization.

    No causal discovery, observational identification, mixed variables,
    q-batch selection, or candidate-dependent cost models are performed.
    """
    groups = tuple(surrogates)
    if not groups:
        raise ValueError("At least one intervention surrogate is required.")
    if direction not in {"maximize", "minimize"}:
        raise ValueError("direction must be 'maximize' or 'minimize'.")
    if num_restarts < 1 or raw_samples < 1:
        raise ValueError("num_restarts and raw_samples must be positive.")

    if any(not isinstance(group, InterventionGPSurrogate) for group in groups):
        raise TypeError("surrogates must contain InterventionGPSurrogate instances.")
    graph = groups[0].graph
    outcome = groups[0].outcome
    keys = [frozenset(group.intervention_nodes) for group in groups]
    if any(group.graph is not graph or group.outcome != outcome for group in groups):
        raise ValueError("All surrogates must share the same graph instance and outcome.")
    if len(set(keys)) != len(keys):
        raise ValueError("Each intervention set must appear only once.")
    if any(group.model.training for group in groups):
        raise RuntimeError("Fit each surrogate before proposing an intervention.")

    required = set().union(*keys)
    if set(bounds) != required:
        raise ValueError("bounds must specify exactly the candidate intervention variables.")
    for node, interval in bounds.items():
        if len(interval) != 2:
            raise ValueError(f"bounds for {node} must contain lower and upper values.")
        lower, upper = interval
        if not isfinite(float(lower)) or not isfinite(float(upper)) or lower >= upper:
            raise ValueError(f"bounds for {node} must be finite with lower < upper.")

    costs = {} if intervention_costs is None else dict(intervention_costs)
    if set(costs) - set(keys):
        raise ValueError("intervention_costs contains an unknown intervention set.")
    for group_key, value in costs.items():
        if not isinstance(group_key, frozenset):
            raise TypeError("intervention_costs keys must be frozensets of node names.")
        if isinstance(value, bool) or not isfinite(float(value)) or value <= 0:
            raise ValueError("intervention costs must be positive finite numbers.")

    outcomes = torch.cat([group.train_Y.reshape(-1) for group in groups])
    best_f = float(outcomes.max() if direction == "maximize" else outcomes.min())
    selected: CausalInterventionProposal | None = None

    for group, group_key in zip(groups, keys, strict=True):
        train_X = group.train_X
        group_bounds = torch.tensor(
            [bounds[node] for node in group.intervention_nodes],
            dtype=train_X.dtype,
            device=train_X.device,
        ).T.contiguous()
        acquisition = LogExpectedImprovement(
            model=group.model,
            best_f=best_f,
            maximize=direction == "maximize",
        )
        candidate, log_ei_tensor = optimize_acqf(
            acquisition,
            group_bounds,
            q=1,
            optimizer=optimizer,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )
        if candidate.shape != (1, len(group.intervention_nodes)):
            raise ValueError("Optimizer must return one candidate in intervention coordinates.")
        if not bool(torch.isfinite(candidate).all()):
            raise ValueError("Optimizer returned non-finite intervention values.")
        if bool((candidate < group_bounds[0] - 1e-6).any()) or bool(
            (candidate > group_bounds[1] + 1e-6).any()
        ):
            raise ValueError("Optimizer returned a candidate outside the intervention bounds.")
        if log_ei_tensor.numel() != 1:
            raise ValueError("Optimizer must return one scalar acquisition value.")
        log_ei = float(log_ei_tensor.item())
        if not isfinite(log_ei):
            raise ValueError("Optimizer returned a non-finite LogEI value.")
        cost = float(costs.get(group_key, 1.0))
        score = log_ei - log(cost)
        result = CausalInterventionProposal(
            intervention=HardIntervention(
                dict(zip(group.intervention_nodes, candidate[0].tolist(), strict=True))
            ),
            log_expected_improvement=log_ei,
            intervention_cost=cost,
            log_utility=score,
        )
        if selected is None or result.log_utility > selected.log_utility:
            selected = result

    if selected is None:  # pragma: no cover - nonempty groups above
        raise RuntimeError("No candidate was generated.")
    return selected
