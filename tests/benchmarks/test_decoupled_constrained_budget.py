import pytest

from robotorchan.benchmarks.constrained_problems import constrained_annulus, constrained_quadratic
from robotorchan.benchmarks.decoupled_constrained import run_constrained_decoupled_benchmark


@pytest.mark.parametrize("decoupled", [False, True])
@pytest.mark.parametrize("factory", [constrained_quadratic, constrained_annulus])
def test_constrained_budget(factory, decoupled):
    problem = factory()
    costs = (3,) + (1,) * problem.n_constraints
    budget = 3 * sum(costs)
    rows = run_constrained_decoupled_benchmark(
        problem,
        output_costs=costs,
        infeasible_utility=-5.0,
        budget=budget,
        n_initial=2,
        num_samples=4,
        decoupled=decoupled,
        fit_model=False,
    )
    assert rows[0]["cost"] == 2 * sum(costs)
    assert rows[-1]["cost"] == budget
    for row in rows:
        assert row["cost"] == sum(n * c for n, c in zip(row["observations"], costs, strict=True))
        assert row["cost"] <= budget
        if row["feasible"]:
            assert row["simple_regret"] >= 0
        else:
            assert row["simple_regret"] is None
