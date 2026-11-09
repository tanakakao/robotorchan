"""Cost-budget benchmark output contract, without repeated GP optimization."""

import importlib.util
from pathlib import Path


def test_random_baseline_respects_budget_and_records_target_diagnostic() -> None:
    benchmark = (
        Path(__file__).resolve().parents[2] / "benchmarks" / "multifidelity_cost_budget.py"
    )
    spec = importlib.util.spec_from_file_location("multifidelity_cost_budget", benchmark)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    rows = module.experiment("random", seed=0, total_budget=3.0)
    assert rows
    assert all(0 <= row["spent"] <= 3.0 for row in rows)
    assert all(0.5 <= row["fidelity"] <= 1.0 for row in rows)
    assert all(row["best_target_oracle"] >= -1.0 for row in rows)
