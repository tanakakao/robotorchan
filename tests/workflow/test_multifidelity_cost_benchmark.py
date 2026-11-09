"""Cost-budget benchmark output contract, without repeated GP optimization."""

from benchmarks.multifidelity_cost_budget import experiment


def test_random_baseline_respects_budget_and_records_target_diagnostic() -> None:
    rows = experiment("random", seed=0, total_budget=3.0)
    assert rows
    assert all(0 <= row["spent"] <= 3.0 for row in rows)
    assert all(0.5 <= row["fidelity"] <= 1.0 for row in rows)
    assert all(row["best_target_oracle"] >= -1.0 for row in rows)
