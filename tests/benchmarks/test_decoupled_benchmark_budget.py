import pytest

from robotorchan.benchmarks.decoupled import run_decoupled_benchmark
from robotorchan.benchmarks.multiobjective_problems import branin_currin


@pytest.mark.parametrize("decoupled", [False, True])
@pytest.mark.parametrize("policy", ["epig", "random", "sobol"])
def test_equal_budget_accounting(decoupled, policy):
    records = run_decoupled_benchmark(
        branin_currin(),
        budget=12,
        n_initial=2,
        num_samples=4,
        decoupled=decoupled,
        policy=policy,
        fit_model=False,
    )
    assert records[0]["cost"] == 8
    assert len(records) >= 2
    for row in records:
        assert row["cost"] <= 12
        assert sum(n * c for n, c in zip(row["observations"], (1, 3), strict=True)) == row["cost"]
        assert row["hypervolume"] >= 0
    assert records[-1]["cost"] == 12
