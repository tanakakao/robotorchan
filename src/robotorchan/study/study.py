"""In-memory Study core, independent of model training and optimizers."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace
from math import isfinite
from typing import Any

import torch
from botorch.acquisition.logei import qLogExpectedImprovement

from robotorchan.models.standard.single_task import SingleTaskGP
from robotorchan.optim.base import SearchStrategy
from robotorchan.optim.constraints.contracts import CandidateConstraints
from robotorchan.optim.domains.variable_space import MixedVariableSpace
from robotorchan.study.feasibility import is_feasible
from robotorchan.study.generation import CandidateGenerator
from robotorchan.study.search_space import SearchSpace
from robotorchan.study.trial import Trial, TrialState
from robotorchan.workflow.components import SelectionStatus
from robotorchan.workflow.problem import ProblemSpec
from robotorchan.workflow.recommendation import WorkflowRecommendation, check_compatibility


class Study:
    """Manage experiment trials with stable, monotonically increasing IDs."""

    def __init__(
        self,
        search_space: SearchSpace,
        *,
        direction: str = "maximize",
        model: str | Callable[..., Any] = SingleTaskGP,
        model_kwargs: Mapping[str, Any] | None = None,
        acquisition: str | Callable[..., Any] = qLogExpectedImprovement,
        acquisition_kwargs: Mapping[str, Any] | None = None,
        optimizer: str | SearchStrategy = "botorch",
        optimizer_kwargs: Mapping[str, Any] | None = None,
        n_initial_trials: int = 3,
        seed: int | None = None,
        variable_space: MixedVariableSpace | None = None,
        constraints: CandidateConstraints | None = None,
        context_names: tuple[str, ...] = (),
        output_names: tuple[str, ...] = (),
        decoupled_proposer: Callable[..., Any] | None = None,
    ) -> None:
        self.search_space = search_space
        names = tuple(search_space.bounds)
        if (
            len(set(context_names)) != len(context_names)
            or any(name not in names for name in context_names)
            or len(context_names) == len(names)
        ):
            raise ValueError("context_names must be unique known dimensions with decisions left.")
        self.context_names = tuple(context_names)
        if len(set(output_names)) != len(output_names) or any(
            not isinstance(name, str) or not name for name in output_names
        ):
            raise ValueError("output_names must contain unique nonempty names.")
        if bool(output_names) != (decoupled_proposer is not None):
            raise ValueError("Decoupled Study requires output_names and a proposer.")
        if decoupled_proposer is not None and not callable(decoupled_proposer):
            raise TypeError("decoupled_proposer must be callable.")
        self.output_names = tuple(output_names)
        self.decoupled_proposer = decoupled_proposer
        self._generator = CandidateGenerator(
            search_space,
            direction=direction,
            model=model,
            model_kwargs=model_kwargs,
            acquisition=acquisition,
            acquisition_kwargs=acquisition_kwargs,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            n_initial_trials=n_initial_trials,
            seed=seed,
            variable_space=variable_space,
            constraints=constraints,
        )
        self._trials: dict[int, Trial] = {}
        self._next_number = 0

    @property
    def trials(self) -> tuple[Trial, ...]:
        """Return immutable trial snapshots in registration order."""
        return tuple(self._trials.values())

    def trials_dataframe(self, *, states: tuple[TrialState, ...] | None = None) -> Any:
        """Export trial snapshots as a pandas DataFrame.

        Columns are number, state, value, followed by named search-space
        parameters in declaration order. Non-complete values are missing.
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "DataFrame export requires pandas. Install robotorchan[dataframe]."
            ) from exc

        if states is not None and (
            not isinstance(states, tuple)
            or any(not isinstance(state, TrialState) for state in states)
        ):
            raise TypeError("states must be a tuple of TrialState values.")
        names = tuple(self.search_space.bounds)
        columns = ["number", "state", "value", *names]
        if self.output_names:
            columns += [
                "evaluation_mask",
                *[f"observation:{n}" for n in self.output_names],
                *[f"actual_cost:{n}" for n in self.output_names],
            ]
        rows = [
            {
                "number": trial.number,
                "state": trial.state.value,
                "value": trial.value,
                **trial.params,
                **(
                    {
                        "evaluation_mask": trial.evaluation_mask,
                        **{f"observation:{n}": v for n, v in trial.observations.items()},
                        **{f"actual_cost:{n}": v for n, v in trial.actual_costs.items()},
                    }
                    if self.output_names
                    else {}
                ),
            }
            for trial in self.trials
            if states is None or trial.state in states
        ]
        return pd.DataFrame.from_records(rows, columns=columns)

    @property
    def best_trial(self) -> Trial:
        """Return the best completed trial, or raise if none exists."""
        if self.output_names:
            raise ValueError("Decoupled Study has no implicit scalar best trial.")
        completed = [trial for trial in self.trials if trial.state is TrialState.COMPLETE]
        if not completed:
            raise ValueError("No completed trials are available.")
        if self._generator.direction == "maximize":
            return max(completed, key=lambda trial: trial.value)
        return min(completed, key=lambda trial: trial.value)

    @property
    def best_value(self) -> float:
        """Return the best completed objective value."""
        return self.best_trial.value

    @property
    def best_params(self) -> Mapping[str, float]:
        """Return the best completed trial parameters."""
        return self.best_trial.params

    def optimize(
        self,
        objective: Callable[[Trial], float],
        n_trials: int,
        *,
        callbacks: list[Callable[[Study, Trial], None]] | None = None,
        catch: tuple[type[Exception], ...] = (),
    ) -> None:
        """Run sequential ask/evaluate/tell with optional failure handling.

        Objective receives a Trial and returns one finite scalar. Exceptions
        matching catch mark the trial failed; other exceptions are re-raised
        after marking the trial failed. Callback errors propagate.
        """
        if self.output_names:
            raise NotImplementedError("Use ask/tell with named observations for decoupled Study.")
        if not callable(objective):
            raise TypeError("objective must be callable.")
        if isinstance(n_trials, bool) or not isinstance(n_trials, int) or n_trials < 0:
            raise ValueError("n_trials must be a non-negative integer.")
        if not isinstance(catch, tuple) or any(
            not isinstance(exc, type) or not issubclass(exc, Exception) for exc in catch
        ):
            raise TypeError("catch must be a tuple of Exception subclasses.")
        callbacks = list(callbacks or [])
        if not all(callable(callback) for callback in callbacks):
            raise TypeError("callbacks must contain callables.")
        for _ in range(n_trials):
            trial = self.ask()
            try:
                value = objective(trial)
                finished = self.tell(trial, value)
            except Exception as exc:
                if self.get_trial(trial.number).state is TrialState.RUNNING:
                    finished = self.tell(trial, state=TrialState.FAIL)
                if not isinstance(exc, catch):
                    raise
            for callback in callbacks:
                callback(self, finished)

    def get_trial(self, number: int) -> Trial:
        """Return a snapshot by ID; raise KeyError for unknown IDs."""
        return self._trials[number]

    def ask(self, *, context: Mapping[str, float] | None = None) -> Trial:
        """Propose a trial for the observed context, if configured."""
        return self.ask_batch(1, context=context)[0]

    def ask_batch(self, q: int, *, context: Mapping[str, float] | None = None) -> tuple[Trial, ...]:
        """Propose q joint candidates sharing one observed context."""
        if self.context_names:
            if context is None or set(context) != set(self.context_names):
                raise ValueError("context must specify exactly the configured context_names.")
        elif context is not None:
            raise ValueError("This Study has no configured context_names.")
        names = tuple(self.search_space.bounds)
        fixed = (
            {names.index(name): context[name] for name in self.context_names}
            if context is not None
            else None
        )
        if self.decoupled_proposer is not None:
            return self._ask_decoupled(q, fixed)
        completed = [trial for trial in self.trials if trial.state is TrialState.COMPLETE]
        candidates = self._generator.propose_batch(completed, q=q, fixed_features=fixed)
        return tuple(self.start_trial(self._register_trial(params).number) for params in candidates)

    def _ask_decoupled(self, q: int, fixed: Mapping[int, float] | None) -> tuple[Trial, ...]:
        if type(q) is not int or q < 1:
            raise ValueError("q must be a positive integer.")
        bounds = torch.tensor(list(self.search_space.bounds.values()), dtype=torch.double).T
        points, mask, _ = self.decoupled_proposer(self.trials, bounds, q=q, fixed_features=fixed)
        if points.shape != (q, bounds.shape[-1]) or not torch.isfinite(points).all():
            raise ValueError("Proposer must return finite [q, d] candidates.")
        if (
            mask.shape != (q, len(self.output_names))
            or mask.dtype != torch.bool
            or not mask.any(dim=-1).all()
        ):
            raise ValueError("Proposer must return nonempty boolean [q, m] masks.")
        if not ((points >= bounds[0]) & (points <= bounds[1])).all():
            raise ValueError("Proposer returned out-of-bounds candidates.")
        for dim, value in (fixed or {}).items():
            if not (points[:, dim] == value).all():
                raise ValueError("Proposer violated fixed context.")
        if not is_feasible(
            points, self._generator.constraints, variable_space=self._generator.variable_space
        ):
            raise ValueError("Proposer returned infeasible candidates.")
        names = tuple(self.search_space.bounds)
        variable_space = self._generator.variable_space
        if variable_space is not None:
            for dim in variable_space.integer_dims:
                if not (points[:, dim] == points[:, dim].round()).all():
                    raise ValueError("Trial integer parameter must be integral.")
            for dim, values in variable_space.categorical_values.items():
                if any(value not in values for value in points[:, dim].tolist()):
                    raise ValueError("Trial categorical parameter is not allowed.")
        return tuple(
            self.start_trial(
                self._register_trial(
                    dict(zip(names, row.tolist(), strict=True)),
                    evaluation_mask=tuple(
                        name
                        for name, selected in zip(self.output_names, flags, strict=True)
                        if selected
                    ),
                ).number
            )
            for row, flags in zip(points, mask.tolist(), strict=True)
        )

    def retry_failed(self, number: int) -> Trial:
        """Register a new trial for only the failed trial's unobserved outputs."""
        current = self.get_trial(number)
        if current.state is not TrialState.FAIL or not current.evaluation_mask:
            raise ValueError("Retry requires a failed decoupled trial.")
        remaining = tuple(n for n in current.evaluation_mask if n not in current.observations)
        if not remaining:
            raise ValueError("No unobserved outputs remain.")
        check_budget = getattr(self.decoupled_proposer, "check_retry_budget", None)
        if check_budget is not None:
            check_budget(self.trials, remaining)
        return self.start_trial(
            self._register_trial(current.params, evaluation_mask=remaining).number
        )

    def ask_contexts(self, contexts: list[Mapping[str, float]]) -> tuple[Trial, ...]:
        """Propose one candidate per observed context using independent q=1 searches.

        This is not joint q-batch acquisition optimization. CandidateConstraints
        must be intra-point because different contexts are optimized separately.
        """
        if self.output_names:
            raise NotImplementedError("Use ask_batch with one context for decoupled Study.")
        if not self.context_names:
            raise ValueError("ask_contexts requires configured context_names.")
        if not isinstance(contexts, list) or not contexts:
            raise ValueError("contexts must be a nonempty list of context mappings.")
        for context in contexts:
            if not isinstance(context, Mapping) or set(context) != set(self.context_names):
                raise ValueError("Each context must specify exactly the configured context_names.")
        constraints = self._generator.constraints
        if constraints is not None:
            linear = (*constraints.inequality_constraints, *constraints.equality_constraints)
            if any(indices.ndim == 2 for indices, _, _ in linear) or any(
                not intrapoint for _, intrapoint in constraints.nonlinear_inequality_constraints
            ):
                raise NotImplementedError(
                    "ask_contexts does not support inter-point CandidateConstraints."
                )
        names = tuple(self.search_space.bounds)
        completed = [trial for trial in self.trials if trial.state is TrialState.COMPLETE]
        candidates = [
            self._generator.propose_batch(
                completed,
                q=1,
                fixed_features={names.index(name): context[name] for name in self.context_names},
            )[0]
            for context in contexts
        ]
        return tuple(self.start_trial(self._register_trial(params).number) for params in candidates)

    def tell(
        self,
        trial: Trial | int,
        value: float | None = None,
        *,
        state: TrialState = TrialState.COMPLETE,
        observations: Mapping[str, float] | None = None,
        actual_costs: Mapping[str, float] | None = None,
    ) -> Trial:
        """Complete or fail a running trial after external evaluation."""
        number = trial.number if isinstance(trial, Trial) else trial
        current = self.get_trial(number)
        if isinstance(trial, Trial) and trial.params != current.params:
            raise ValueError("Trial parameters do not match the current Study trial.")
        if current.evaluation_mask:
            if value is not None:
                raise ValueError("Decoupled trials require named observations.")
            if current.state is not TrialState.RUNNING:
                raise ValueError("Only running trials accept observations.")
            if state not in (TrialState.COMPLETE, TrialState.FAIL):
                raise ValueError("Invalid tell state.")
            updates = dict(observations or {})
            if set(updates) & set(current.observations):
                raise ValueError("An output has already been observed.")
            merged = {**current.observations, **updates}
            if not updates and state is not TrialState.FAIL:
                raise ValueError("Provide at least one observation.")
            target = state
            if state is TrialState.COMPLETE and set(merged) != set(current.evaluation_mask):
                target = TrialState.RUNNING
            costs = dict(actual_costs or {})
            if set(costs) & set(current.actual_costs):
                raise ValueError("An actual cost has already been recorded.")
            updated = replace(
                current,
                observations=merged,
                actual_costs={**current.actual_costs, **costs},
                state=target,
            )
            self._trials[number] = updated
            return updated
        if observations is not None or actual_costs is not None:
            raise ValueError("Scalar trials do not accept named observations.")
        if state is TrialState.COMPLETE:
            if value is None:
                raise ValueError("Completed trials require an observation.")
            return self.complete_trial(number, value)
        if state is TrialState.FAIL:
            if value is not None:
                raise ValueError("Failed trials must not contain an observation.")
            return self.fail_trial(number)
        raise ValueError("tell() only accepts COMPLETE or FAIL as terminal states.")

    def add_trial(
        self, params: Mapping[str, float], *, evaluation_mask: tuple[str, ...] = ()
    ) -> Trial:
        """Register a feasible externally supplied candidate in WAITING state."""
        return self._register_trial(params, check_constraints=True, evaluation_mask=evaluation_mask)

    def _register_trial(
        self,
        params: Mapping[str, float],
        *,
        check_constraints: bool = False,
        evaluation_mask: tuple[str, ...] = (),
    ) -> Trial:
        self.search_space.validate(params)
        if self._generator.variable_space is not None:
            names = tuple(self.search_space.bounds)
            candidate = torch.tensor(
                [params[name] for name in names], dtype=torch.double
            ).unsqueeze(0)
            variable_space = self._generator.variable_space
            for dim in variable_space.integer_dims:
                if candidate[0, dim] != candidate[0, dim].round():
                    raise ValueError("Trial integer parameter must be integral.")
            for dim, values in variable_space.categorical_values.items():
                if candidate[0, dim].item() not in values:
                    raise ValueError("Trial categorical parameter is not allowed.")
        else:
            candidate = torch.tensor(
                [params[name] for name in self.search_space.bounds],
                dtype=torch.double,
            ).unsqueeze(0)
        if check_constraints and not is_feasible(
            candidate,
            self._generator.constraints,
            variable_space=self._generator.variable_space,
        ):
            raise ValueError("Trial parameters violate CandidateConstraints.")
        if bool(evaluation_mask) != bool(self.output_names) or not set(evaluation_mask).issubset(
            self.output_names
        ):
            raise ValueError("Evaluation mask must select configured outputs.")
        trial = Trial(number=self._next_number, params=params, evaluation_mask=evaluation_mask)
        self._trials[trial.number] = trial
        self._next_number += 1
        return trial

    def start_trial(self, number: int) -> Trial:
        """Move a waiting trial to RUNNING."""
        return self._transition(number, TrialState.WAITING, TrialState.RUNNING)

    def complete_trial(self, number: int, value: float) -> Trial:
        """Record a single finite objective observation exactly once."""
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
            raise ValueError("Observation must be a finite number.")
        return self._transition(number, TrialState.RUNNING, TrialState.COMPLETE, value=float(value))

    def fail_trial(self, number: int) -> Trial:
        """Mark a running trial as failed."""
        return self._transition(number, TrialState.RUNNING, TrialState.FAIL)

    def _transition(
        self,
        number: int,
        expected: TrialState,
        target: TrialState,
        *,
        value: float | None = None,
    ) -> Trial:
        current = self.get_trial(number)
        if current.state is not expected:
            raise ValueError(
                f"Trial {number} must be {expected.value} to transition to {target.value}."
            )
        updated = replace(current, state=target, value=value)
        self._trials[number] = updated
        return updated


def create_study(
    *,
    search_space: SearchSpace | Mapping[str, tuple[float, float]],
    direction: str = "maximize",
    model: str | Callable[..., Any] | None = None,
    model_kwargs: Mapping[str, Any] | None = None,
    acquisition: str | Callable[..., Any] | None = None,
    acquisition_kwargs: Mapping[str, Any] | None = None,
    optimizer: str | SearchStrategy | None = None,
    optimizer_kwargs: Mapping[str, Any] | None = None,
    n_initial_trials: int = 3,
    seed: int | None = None,
    variable_space: MixedVariableSpace | None = None,
    constraints: CandidateConstraints | None = None,
    context_names: tuple[str, ...] = (),
    output_names: tuple[str, ...] = (),
    decoupled_proposer: Callable[..., Any] | None = None,
    recommendation: WorkflowRecommendation | None = None,
    initial_trials: Iterable[Trial] = (),
) -> Study:
    """Create a scalar Study, optionally restoring ordered historical trial snapshots."""
    if recommendation is not None:
        if output_names:
            raise ValueError("Scalar recommendations do not configure decoupled Study.")
        if any(value is not None for value in (model, acquisition, optimizer)):
            raise ValueError("recommendation cannot accompany explicit component selections.")
        if recommendation.compatibility.status not in (
            SelectionStatus.COMPATIBLE,
            SelectionStatus.CONDITIONAL,
        ):
            raise ValueError("recommendation must be compatible or conditional.")
        if (
            recommendation.optimizer_name is not None
            and recommendation.search_strategy_name is not None
        ):
            raise ValueError("Study requires one optimizer or search strategy, not both.")
        if recommendation.optimizer_name is None and recommendation.search_strategy_name is None:
            raise ValueError("recommendation must specify an optimizer or search strategy.")
        spec = ProblemSpec(
            purpose="bayesian_optimization",
            input_type="mixed" if variable_space is not None else "continuous",
            candidate_constrained=constraints is not None and constraints.has_constraints,
        )
        decision = check_compatibility(
            spec,
            model=recommendation.model_name,
            acquisition=recommendation.acquisition_name,
            optimizer=recommendation.optimizer_name,
            search_strategy=recommendation.search_strategy_name,
        )
        if decision.status not in (SelectionStatus.COMPATIBLE, SelectionStatus.CONDITIONAL):
            raise ValueError(
                f"recommendation is incompatible with scalar Study: {decision.reasons}"
            )
        model = recommendation.model_name
        acquisition = recommendation.acquisition_name
        optimizer = recommendation.search_strategy_name or recommendation.optimizer_name
    if not isinstance(search_space, SearchSpace):
        search_space = SearchSpace(search_space)
    study = Study(
        search_space,
        direction=direction,
        model=SingleTaskGP if model is None else model,
        model_kwargs=model_kwargs,
        acquisition=qLogExpectedImprovement if acquisition is None else acquisition,
        acquisition_kwargs=acquisition_kwargs,
        optimizer="botorch" if optimizer is None else optimizer,
        optimizer_kwargs=optimizer_kwargs,
        n_initial_trials=n_initial_trials,
        seed=seed,
        variable_space=variable_space,
        constraints=constraints,
        context_names=context_names,
        output_names=output_names,
        decoupled_proposer=decoupled_proposer,
    )
    # Restore observations, not optimizer or random-number-generator state.
    for trial in initial_trials:
        if not isinstance(trial, Trial):
            raise TypeError("initial_trials must contain Trial snapshots.")
        if type(trial.number) is not int or trial.number < study._next_number:
            raise ValueError("initial_trials must have strictly increasing integer numbers.")
        if not isinstance(trial.state, TrialState):
            raise ValueError("initial_trials must have valid TrialState values.")
        if (
            not trial.evaluation_mask
            and trial.state is TrialState.COMPLETE
            and (isinstance(trial.value, bool) or not isinstance(trial.value, (int, float)))
        ):
            raise ValueError("Completed trials require a numeric observation.")
        # Historical joint batches cannot be reconstructed from individual trials.
        study._next_number = trial.number
        study._register_trial(trial.params, evaluation_mask=trial.evaluation_mask)
        study._trials[trial.number] = trial
    return study
