"""Focused smoke test for the sequential MF-KG workflow example."""

import importlib.util
from pathlib import Path

import pytest


@pytest.mark.slow
def test_mfkg_cost_budget_loop() -> None:
    example = Path(__file__).resolve().parents[2] / "examples" / "mfkg_cost_budget_loop.py"
    spec = importlib.util.spec_from_file_location("mfkg_cost_budget_loop", example)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    budget = module.run(iterations=1, budget_total=10.0)
    assert budget.spent > 0
    assert budget.reserved == 0
    assert budget.remaining < budget.total
