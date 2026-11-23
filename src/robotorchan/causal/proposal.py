"""Choose a single hard intervention using existing BoTorch LogEI and optimizers."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite, log
from typing import Literal

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
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


@dataclass(frozen=True, slots=True)
class CausalAcquisitionProposal:
    """Best intervention for caller-supplied, comparable acquisition scores.

    acquisition_value is the native BoTorch acquisition value, whereas
    ranking_score is the caller-defined value used across intervention sets.
    """

    intervention: HardIntervention
    acquisition_value: float
    ranking_score: float


def optimize_causal_acqfs(
    acqfs: Mapping[tuple[str, ...], AcquisitionFunction],
    bounds: Mapping[str, tuple[float, float]],
    *,
    score: Callable[[tuple[str, ...], float], float],
    optimizer: OptimizerName = "botorch",
    num_restarts: int = 5,
    raw_samples: int = 64,
) -> CausalAcquisitionProposal:
    """Optimize q=1 native BoTorch acquisitions over alternative intervention sets.

    Each key is an *ordered* tuple of intervention nodes: its order defines
    the input columns of the corresponding acquisition function. The caller
    constructs each acquisition function directly, choosing its model,
    incumbent, objective and other hyperparameters with native BoTorch APIs.

    The explicit score(nodes, acquisition_value) callback must put the
    optimized native values on a common scale. For example, a cost-aware
    LogEI score is log_ei - log(cost). Do not apply that rule to UCB or
    PosteriorMean automatically. This helper does not infer causal effects
    from observational data or perform q>1 or mixed-variable optimization.
    """
    if not isinstance(acqfs, Mapping):
        raise TypeError("acqfs must map ordered node tuples to AcquisitionFunction instances.")
    if not acqfs:
        raise ValueError("At least one acquisition function is required.")
    if not callable(score):
        raise TypeError("score must be a callable for cross-set ranking.")
    if num_restarts < 1 or raw_samples < 1:
        raise ValueError("num_restarts and raw_samples must be positive.")

    seen: set[frozenset[str]] = set()
    required: set[str] = set()
    for nodes, acquisition in acqfs.items():
        if not isinstance(nodes, tuple) or not nodes:
            raise ValueError("Acquisition keys must be nonempty ordered node tuples.")
        if any(not isinstance(node, str) or not node for node in nodes):
            raise ValueError("Intervention node names must be nonempty strings.")
        if len(set(nodes)) != len(nodes):
            raise ValueError("Duplicate nodes in an intervention set.")
        if not isinstance(acquisition, AcquisitionFunction):
            raise TypeError("Each acquisition must be a BoTorch AcquisitionFunction.")
        group = frozenset(nodes)
        if group in seen:
            raise ValueError("Each intervention set must appear only once.")
        seen.add(group)
        required.update(nodes)

    if set(bounds) != required:
        raise ValueError("bounds must specify exactly the candidate intervention variables.")
    checked_bounds: dict[str, tuple[float, float]] = {}
    for node, interval in bounds.items():
        try:
            lower, upper = interval
            if isinstance(lower, bool) or isinstance(upper, bool):
                raise ValueError("Boolean bounds are not valid.")
            low, high = float(lower), float(upper)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"bounds for {node} must contain lower and upper values.") from exc
        if not isfinite(low) or not isfinite(high) or low >= high:
            raise ValueError(f"bounds for {node} must be finite with lower < upper.")
        checked_bounds[node] = (low, high)

    selected: CausalAcquisitionProposal | None = None
    for nodes, acquisition in acqfs.items():
        reference = next(acquisition.parameters(), None)
        if reference is None:
            reference = next(acquisition.buffers(), None)
        dtype = (
            reference.dtype if reference is not None and reference.is_floating_point()
            else torch.double
        )
        device = reference.device if reference is not None else torch.device("cpu")
        group_bounds = torch.tensor(
            [checked_bounds[node] for node in nodes],
            dtype=dtype,
            device=device,
        ).T.contiguous()

        candidate, native_value = optimize_acqf(
            acquisition,
            group_bounds,
            q=1,
            optimizer=optimizer,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )
        if candidate.shape != (1, len(nodes)):
            raise ValueError("Optimizer must return one candidate in intervention coordinates.")
        if not bool(torch.isfinite(candidate).all()):
            raise ValueError("Optimizer returned non-finite intervention values.")
        if bool((candidate < group_bounds[0] - 1e-6).any()) or bool(
            (candidate > group_bounds[1] + 1e-6).any()
        ):
            raise ValueError("Optimizer returned a candidate outside the intervention bounds.")
        if native_value.numel() != 1:
            raise ValueError("Optimizer must return one scalar acquisition value.")
        acquisition_value = float(native_value.item())
        if not isfinite(acquisition_value):
            raise ValueError("Optimizer returned a non-finite acquisition value.")
        raw_score = score(nodes, acquisition_value)
        if isinstance(raw_score, bool):
            raise ValueError("score must return a finite real number.")
        try:
            ranking_score = float(raw_score)
        except (TypeError, ValueError) as exc:
            raise ValueError("score must return a finite real number.") from exc
        if not isfinite(ranking_score):
            raise ValueError("score must return a finite real number.")

        result = CausalAcquisitionProposal(
            intervention=HardIntervention(dict(zip(nodes, candidate[0].tolist(), strict=True))),
            acquisition_value=acquisition_value,
            ranking_score=ranking_score,
        )
        if selected is None or result.ranking_score > selected.ranking_score:
            selected = result

    if selected is None:  # pragma: no cover - nonempty acqfs above
        raise RuntimeError("No candidate was generated.")
    return selected


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
    acquisitions = {
        group.intervention_nodes: LogExpectedImprovement(
            model=group.model,
            best_f=best_f,
            maximize=direction == "maximize",
        )
        for group in groups
    }
    chosen = optimize_causal_acqfs(
        acquisitions,
        bounds,
        score=lambda nodes, value: value - log(float(costs.get(frozenset(nodes), 1.0))),
        optimizer=optimizer,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
    )
    selected_cost = float(costs.get(frozenset(chosen.intervention.values), 1.0))
    return CausalInterventionProposal(
        intervention=chosen.intervention,
        log_expected_improvement=chosen.acquisition_value,
        intervention_cost=selected_cost,
        log_utility=chosen.ranking_score,
    )
