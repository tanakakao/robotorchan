"""One-iteration integration smoke test for cost-aware multi-fidelity TuRBO."""

import importlib.util
from pathlib import Path


def test_turbo_multifidelity_cost_budget_loop() -> None:
    example = (
        Path(__file__).resolve().parents[2] / "examples" / "turbo_multifidelity_cost_budget.py"
    )
    spec = importlib.util.spec_from_file_location("turbo_multifidelity_cost_budget", example)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    budget = module.run(iterations=1, budget_total=10.0)
    assert budget.spent > 0
    assert budget.reserved == 0
    assert budget.remaining < budget.total
