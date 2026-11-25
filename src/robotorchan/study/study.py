"""In-memory Study core, independent of model training and optimizers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
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
        rows = [
            {
                "number": trial.number,
                "state": trial.state.value,
                "value": trial.value,
                **trial.params,
            }
            for trial in self.trials
            if states is None or trial.state in states
        ]
        return pd.DataFrame.from_records(rows, columns=columns)

    @property
    def best_trial(self) -> Trial:
        """Return the best completed trial, or raise if none exists."""
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
        completed = [trial for trial in self.trials if trial.state is TrialState.COMPLETE]
        candidates = self._generator.propose_batch(completed, q=q, fixed_features=fixed)
        return tuple(self.start_trial(self._register_trial(params).number) for params in candidates)

    def ask_contexts(self, contexts: list[Mapping[str, float]]) -> tuple[Trial, ...]:
        """Propose one candidate per observed context using independent q=1 searches.

        This is not joint q-batch acquisition optimization. CandidateConstraints
        must be intra-point because different contexts are optimized separately.
        """
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
    ) -> Trial:
        """Complete or fail a running trial after external evaluation."""
        number = trial.number if isinstance(trial, Trial) else trial
        current = self.get_trial(number)
        if isinstance(trial, Trial) and trial.params != current.params:
            raise ValueError("Trial parameters do not match the current Study trial.")
        if state is TrialState.COMPLETE:
            if value is None:
                raise ValueError("Completed trials require an observation.")
            return self.complete_trial(number, value)
        if state is TrialState.FAIL:
            if value is not None:
                raise ValueError("Failed trials must not contain an observation.")
            return self.fail_trial(number)
        raise ValueError("tell() only accepts COMPLETE or FAIL as terminal states.")

    def add_trial(self, params: Mapping[str, float]) -> Trial:
        """Register a feasible externally supplied candidate in WAITING state."""
        return self._register_trial(params, check_constraints=True)

    def _register_trial(
        self, params: Mapping[str, float], *, check_constraints: bool = False
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
        trial = Trial(number=self._next_number, params=params)
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
    recommendation: WorkflowRecommendation | None = None,
) -> Study:
    """Create a scalar Study from explicit components or a selected recommendation."""
    if recommendation is not None:
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
        if (
            recommendation.optimizer_name is None
            and recommendation.search_strategy_name is None
        ):
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
    return Study(
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
    )
