"""Sequential ask/tell workflow for known-graph, hard-intervention CBO."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from math import isfinite
from types import MappingProxyType
from typing import Literal

import torch

from robotorchan.causal.data import CausalObservation
from robotorchan.causal.gp_scm import GPStructuralCausalModel
from robotorchan.causal.gp_scm_proposal import propose_gp_scm_intervention
from robotorchan.causal.graph import CausalGraph
from robotorchan.causal.intervention import HardIntervention
from robotorchan.causal.proposal import (
    CausalAcquisitionProposal,
    CausalInterventionProposal,
    propose_causal_intervention,
)
from robotorchan.causal.surrogate import InterventionGPSurrogate
from robotorchan.optim.backends.sampling import sample_candidate_batches
from robotorchan.optim.dispatch import OptimizerName
from robotorchan.study.trial import TrialState


@dataclass(frozen=True, slots=True)
class CausalTrial:
    """A snapshot of one explicitly intervened experiment."""

    number: int
    intervention: HardIntervention
    state: TrialState
    observation: CausalObservation | None = None
    proposal: CausalInterventionProposal | CausalAcquisitionProposal | None = None


class CausalBOStudy:
    """Sequential intervention experiments without causal-structure discovery.

    The default mode retains independent intervention-set GPs and the
    existing per-set Sobol cold start. In gp_scm mode, provided observational
    records fit shared structural mechanisms and one initial Sobol intervention
    establishes the measured incumbent for MC expected improvement.
    Only one RUNNING trial is allowed at a time.
    """

    def __init__(
        self,
        graph: CausalGraph,
        outcome: str,
        intervention_sets: Sequence[Sequence[str]],
        bounds: Mapping[str, tuple[float, float]],
        *,
        direction: Literal["maximize", "minimize"] = "maximize",
        intervention_costs: Mapping[frozenset[str], float] | None = None,
        optimizer: OptimizerName = "botorch",
        num_restarts: int = 5,
        raw_samples: int = 64,
        n_initial_trials: int = 3,
        seed: int | None = None,
        dtype: torch.dtype = torch.double,
        device: torch.device | str | None = None,
        model_type: Literal["intervention_gp", "gp_scm"] = "intervention_gp",
        observational_data: Iterable[CausalObservation] | None = None,
        mc_samples: int = 128,
        mc_seed: int = 0,
    ) -> None:
        if not isinstance(graph, CausalGraph):
            raise TypeError("graph must be a CausalGraph.")
        if outcome not in graph.nodes:
            raise ValueError("outcome must belong to the graph.")
        if direction not in {"maximize", "minimize"}:
            raise ValueError("direction must be 'maximize' or 'minimize'.")
        if n_initial_trials < 2:
            raise ValueError("n_initial_trials must be at least 2 for the GP.")
        if num_restarts < 1 or raw_samples < 1:
            raise ValueError("num_restarts and raw_samples must be positive.")
        if model_type not in ("intervention_gp", "gp_scm"):
            raise ValueError("model_type must be 'intervention_gp' or 'gp_scm'.")
        if mc_samples < 1 or not isinstance(mc_seed, int) or mc_seed < 0:
            raise ValueError("mc_samples must be positive and mc_seed nonnegative.")
        prior = tuple(() if observational_data is None else observational_data)
        if model_type == "intervention_gp" and prior:
            raise ValueError("observational_data is only used with model_type='gp_scm'.")
        for record in prior:
            if not isinstance(record, CausalObservation):
                raise TypeError("observational_data must contain CausalObservation records.")
            if record.intervention is not None:
                raise ValueError("observational_data must not contain do-interventions.")
            if set(record.values) - set(graph.nodes):
                raise ValueError("observational_data contains unknown graph nodes.")

        sets: list[frozenset[str]] = []
        for nodes in intervention_sets:
            if isinstance(nodes, (str, bytes)):
                raise TypeError("Each intervention set must be a sequence of node names.")
            names = tuple(nodes)
            if not names or len(set(names)) != len(names):
                raise ValueError("Intervention sets must be nonempty without duplicates.")
            key = frozenset(names)
            if outcome in key or not key.issubset(graph.nodes):
                raise ValueError("Intervention nodes must be known non-outcome graph nodes.")
            if key in sets:
                raise ValueError("Intervention sets must be distinct.")
            sets.append(key)
        if not sets:
            raise ValueError("At least one intervention set is required.")

        required = set().union(*sets)
        if set(bounds) != required:
            raise ValueError("bounds must contain exactly the intervention variables.")
        checked_bounds: dict[str, tuple[float, float]] = {}
        for node, limits in bounds.items():
            if len(limits) != 2:
                raise ValueError(f"Bounds for {node} must have lower and upper values.")
            lower, upper = (float(limits[0]), float(limits[1]))
            if not isfinite(lower) or not isfinite(upper) or lower >= upper:
                raise ValueError(f"Bounds for {node} must satisfy finite lower < upper.")
            checked_bounds[node] = (lower, upper)

        costs = {} if intervention_costs is None else dict(intervention_costs)
        if any(not isinstance(key, frozenset) or key not in sets for key in costs):
            raise ValueError("intervention_costs keys must identify configured sets.")
        if any(
            isinstance(cost, bool) or not isfinite(float(cost)) or cost <= 0
            for cost in costs.values()
        ):
            raise ValueError("intervention costs must be finite positive numbers.")

        self.graph = graph
        self.outcome = outcome
        self.intervention_sets = tuple(sets)
        self.bounds = MappingProxyType(checked_bounds)
        self.direction = direction
        self.intervention_costs = MappingProxyType(
            {key: float(value) for key, value in costs.items()}
        )
        self.optimizer = optimizer
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples
        self.n_initial_trials = n_initial_trials
        self.seed = seed
        self.dtype = dtype
        self.device = device
        self.model_type = model_type
        self.observational_data = prior
        self.mc_samples = mc_samples
        self.mc_seed = mc_seed
        self._trials: dict[int, CausalTrial] = {}
        self._surrogates: tuple[InterventionGPSurrogate, ...] | None = None
        self._gp_scm: GPStructuralCausalModel | None = None

    @property
    def trials(self) -> tuple[CausalTrial, ...]:
        """All RUNNING, COMPLETE and FAIL trial snapshots in issuance order."""
        return tuple(self._trials.values())

    @property
    def observations(self) -> tuple[CausalObservation, ...]:
        """Completed hard-intervention records only; failures are excluded."""
        return tuple(
            trial.observation
            for trial in self.trials
            if trial.state is TrialState.COMPLETE and trial.observation is not None
        )

    def get_trial(self, number: int) -> CausalTrial:
        """Return a registered trial snapshot by its monotonic number."""
        return self._trials[number]

    def fit(self) -> tuple[InterventionGPSurrogate, ...] | GPStructuralCausalModel:
        """Fit and cache either independent GPs or the shared GP-SCM.

        In gp_scm mode, mechanism-specific record requirements replace
        the legacy per-intervention-set cold-start requirement.
        Completed do records and observational_data keep their identities.
        """
        if self.model_type == "gp_scm":
            if self._gp_scm is None:
                model = GPStructuralCausalModel(
                    self.graph,
                    (*self.observational_data, *self.observations),
                    dtype=self.dtype,
                    device=self.device,
                )
                model.fit()
                self._gp_scm = model
            return self._gp_scm

        if self._surrogates is None:
            records = self.observations
            for key in self.intervention_sets:
                count = sum(
                    frozenset(record.intervention.values) == key
                    for record in records
                    if record.intervention is not None
                )
                if count < self.n_initial_trials:
                    raise RuntimeError("Complete initial interventions in all sets before fitting.")

            fitted = []
            for key in self.intervention_sets:
                names = tuple(node for node in self.graph.topological_order if node in key)
                surrogate = InterventionGPSurrogate(
                    self.graph,
                    self.outcome,
                    names,
                    records,
                    dtype=self.dtype,
                    device=self.device,
                )
                surrogate.fit()
                fitted.append(surrogate)
            self._surrogates = tuple(fitted)
        return self._surrogates

    def ask(self) -> CausalTrial:
        """Register one RUNNING intervention, using Sobol before the GP phase."""
        if any(trial.state is TrialState.RUNNING for trial in self._trials.values()):
            raise RuntimeError("Tell the running trial before calling ask() again.")

        counts = {
            key: sum(
                trial.state is TrialState.COMPLETE and frozenset(trial.intervention.values) == key
                for trial in self._trials.values()
            )
            for key in self.intervention_sets
        }
        if self.model_type == "gp_scm":
            # One actual intervention is required to establish EI's best_f.
            # The GP-SCM itself may fit from observational_data alone.
            uninitialized = [self.intervention_sets[0]] if not self.observations else []
        else:
            uninitialized = [
                key for key in self.intervention_sets if counts[key] < self.n_initial_trials
            ]
        proposal: CausalInterventionProposal | CausalAcquisitionProposal | None = None
        if uninitialized:
            chosen = min(uninitialized, key=lambda key: counts[key])
            nodes = tuple(node for node in self.graph.topological_order if node in chosen)
            limits = torch.tensor(
                [self.bounds[node] for node in nodes],
                dtype=self.dtype,
                device=self.device,
            ).T.contiguous()
            attempt = sum(
                frozenset(trial.intervention.values) == chosen for trial in self._trials.values()
            )
            candidate = sample_candidate_batches(
                limits,
                num_samples=attempt + 1,
                q=1,
                method="sobol",
                seed=self.seed,
            )[-1, 0]
            intervention = HardIntervention(dict(zip(nodes, candidate.tolist(), strict=True)))
        else:
            if self.model_type == "gp_scm":
                prior_outcomes = [record.values[self.outcome] for record in self.observations]
                incumbent = (
                    max(prior_outcomes) if self.direction == "maximize" else min(prior_outcomes)
                )
                model = self.fit()
                assert isinstance(model, GPStructuralCausalModel)
                groups = [
                    tuple(node for node in self.graph.topological_order if node in group)
                    for group in self.intervention_sets
                ]
                proposal = propose_gp_scm_intervention(
                    model,
                    groups,
                    self.bounds,
                    outcome=self.outcome,
                    best_f=incumbent,
                    direction=self.direction,
                    intervention_costs=self.intervention_costs,
                    optimizer=self.optimizer,
                    num_restarts=self.num_restarts,
                    raw_samples=self.raw_samples,
                    mc_samples=self.mc_samples,
                    mc_seed=self.mc_seed,
                )
            else:
                model = self.fit()
                assert isinstance(model, tuple)
                proposal = propose_causal_intervention(
                    model,
                    self.bounds,
                    direction=self.direction,
                    intervention_costs=self.intervention_costs,
                    optimizer=self.optimizer,
                    num_restarts=self.num_restarts,
                    raw_samples=self.raw_samples,
                )
            intervention = proposal.intervention

        number = len(self._trials)
        trial = CausalTrial(
            number=number,
            intervention=intervention,
            state=TrialState.RUNNING,
            proposal=proposal,
        )
        self._trials[number] = trial
        return trial

    def tell(
        self,
        trial: CausalTrial | int,
        values: Mapping[str, float] | float | None = None,
        *,
        state: TrialState = TrialState.COMPLETE,
    ) -> CausalTrial:
        """Complete or fail one issued trial; reject repeats and mismatches.

        A scalar is the measured outcome. A mapping may include other
        measured nodes, but must contain the outcome and exactly match any
        intervention values supplied. Observational-only data is not inferred.
        """
        number = trial.number if isinstance(trial, CausalTrial) else trial
        current = self.get_trial(number)
        if isinstance(trial, CausalTrial) and trial.intervention != current.intervention:
            raise ValueError("Trial intervention differs from the registered trial.")
        if current.state is not TrialState.RUNNING:
            raise ValueError("Only a RUNNING trial may be completed or failed.")

        if state is TrialState.FAIL:
            if values is not None:
                raise ValueError("Failed trials must not include observations.")
            updated = replace(current, state=TrialState.FAIL)
        elif state is TrialState.COMPLETE:
            if values is None:
                raise ValueError("Completed trials require measured outcome values.")
            if isinstance(values, Mapping):
                measurements = {**current.intervention.values, **values}
            else:
                measurements = {**current.intervention.values, self.outcome: values}
            if self.outcome not in measurements:
                raise ValueError("Measured values must include the outcome.")
            if set(measurements) - set(self.graph.nodes):
                raise ValueError("Measured values contain nodes outside the causal graph.")
            observation = CausalObservation(measurements, current.intervention)
            updated = replace(
                current,
                state=TrialState.COMPLETE,
                observation=observation,
            )
        else:
            raise ValueError("tell() only accepts COMPLETE or FAIL.")
        self._trials[number] = updated
        if updated.state is TrialState.COMPLETE:
            self._surrogates = None
            self._gp_scm = None
        return updated
