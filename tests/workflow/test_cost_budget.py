"""Focused tests for evaluation-cost budget accounting."""

import pytest

from robotorchan.workflow import CostBudget


def test_reservations_prevent_double_spending_and_settle_actual_cost() -> None:
    budget = CostBudget(total=10, spent=2)
    budget.reserve("a", 5)
    assert budget.remaining == 3
    assert not budget.can_afford(4)
    with pytest.raises(ValueError, match="Insufficient"):
        budget.reserve("b", 4)
    budget.reserve("b", 3)
    assert budget.exhausted
    budget.settle("a", 6)
    assert budget.spent == 8
    assert budget.reserved == 3
    assert budget.remaining == 0
    assert budget.cancel("b") == 3
    assert budget.remaining == 2


def test_actual_overrun_is_recorded() -> None:
    budget = CostBudget(total=10)
    budget.reserve("a", 4)
    budget.settle("a", 12)
    assert budget.spent == 12
    assert budget.remaining == 0


def test_invalid_costs_and_unknown_reservations() -> None:
    with pytest.raises(ValueError):
        CostBudget(total=float("nan"))
    budget = CostBudget(total=5)
    for invalid in (-1, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            budget.reserve("bad", invalid)
    budget.reserve("a", 2)
    with pytest.raises(ValueError, match="already pending"):
        budget.reserve("a", 1)
    with pytest.raises(ValueError):
        budget.settle("a", -1)
    assert budget.reserved == 2
    with pytest.raises(KeyError):
        budget.settle("unknown", 1)
    with pytest.raises(KeyError):
        budget.cancel("unknown")
    budget.settle("a", 1)
    budget.record(2)
    assert budget.spent == 3
