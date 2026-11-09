"""Evaluation-cost budget accounting for sequential and pending experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite


def _cost(value: float, *, positive: bool = False) -> float:
    value = float(value)
    if not isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError("Cost must be finite and nonnegative (positive when required).")
    return value


@dataclass
class CostBudget:
    """Track actual spending and predicted-cost reservations.

    Reservation IDs must be unique among pending evaluations. Settlement
    releases the reservation and charges actual cost. Overspending is recorded.
    This in-memory ledger is not concurrency-safe.
    """

    total: float
    spent: float = 0.0
    _pending: dict[str, float] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.total = _cost(self.total, positive=True)
        self.spent = _cost(self.spent)
        if self.spent > self.total:
            raise ValueError("Initial spending exceeds total budget.")

    @property
    def reserved(self) -> float:
        return sum(self._pending.values())

    @property
    def remaining(self) -> float:
        return max(0.0, self.total - self.spent - self.reserved)

    @property
    def exhausted(self) -> bool:
        return self.remaining == 0

    def can_afford(self, predicted_cost: float) -> bool:
        return _cost(predicted_cost) <= self.remaining

    def reserve(self, evaluation_id: str, predicted_cost: float) -> None:
        if not isinstance(evaluation_id, str) or not evaluation_id:
            raise ValueError("Evaluation ID must be a nonempty string.")
        if evaluation_id in self._pending:
            raise ValueError("Evaluation ID is already pending.")
        estimate = _cost(predicted_cost)
        if not self.can_afford(estimate):
            raise ValueError("Insufficient evaluation budget.")
        self._pending[evaluation_id] = estimate

    def cancel(self, evaluation_id: str) -> float:
        """Release a pending reservation without charging an actual cost."""
        return self._pending.pop(evaluation_id)

    def settle(self, evaluation_id: str, actual_cost: float) -> None:
        """Charge completed evaluation and release its predicted reservation."""
        actual = _cost(actual_cost)
        if evaluation_id not in self._pending:
            raise KeyError(evaluation_id)
        self._pending.pop(evaluation_id)
        self.spent += actual

    def record(self, actual_cost: float) -> None:
        """Charge an evaluation without a reservation, e.g. initial design."""
        self.spent += _cost(actual_cost)
