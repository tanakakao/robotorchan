"""In-memory Study core, independent of model training and optimizers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from math import isfinite

from robotorchan.study.search_space import SearchSpace
from robotorchan.study.trial import Trial, TrialState


class Study:
    """Manage experiment trials with stable, monotonically increasing IDs."""

    def __init__(self, search_space: SearchSpace) -> None:
        self.search_space = search_space
        self._trials: dict[int, Trial] = {}
        self._next_number = 0

    @property
    def trials(self) -> tuple[Trial, ...]:
        """Return immutable trial snapshots in registration order."""
        return tuple(self._trials.values())

    def get_trial(self, number: int) -> Trial:
        """Return a snapshot by ID; raise KeyError for unknown IDs."""
        return self._trials[number]

    def add_trial(self, params: Mapping[str, float]) -> Trial:
        """Register a candidate without generating it (Phase 3)."""
        self.search_space.validate(params)
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
        return self._transition(
            number, TrialState.RUNNING, TrialState.COMPLETE, value=float(value)
        )

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


def create_study(*, search_space: SearchSpace | Mapping[str, tuple[float, float]]) -> Study:
    """Create an in-memory study with fixed continuous bounds."""
    if not isinstance(search_space, SearchSpace):
        search_space = SearchSpace(search_space)
    return Study(search_space)
