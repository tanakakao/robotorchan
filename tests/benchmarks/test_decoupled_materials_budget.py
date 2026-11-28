import pytest

from robotorchan.benchmarks.decoupled_materials import run_materials_decoupled_benchmark


@pytest.mark.parametrize("decoupled", [False, True])
def test_materials_budget(decoupled):
    rows = run_materials_decoupled_benchmark(
        budget=36,
        n_initial=3,
        num_samples=2,
        epochs=2,
        decoupled=decoupled,
        fit_regression=False,
    )
    assert rows[0]["cost"] == 27
    assert rows[-1]["cost"] == 36
    for row in rows:
        assert row["cost"] == sum(
            n * c for n, c in zip(row["observations"], (3, 5, 1), strict=True)
        )
        assert 0 <= row["brier_score"] <= 2
        assert -1 <= row["utility"] <= 1
        if not row["feasible"]:
            assert row["utility"] == -1
        if not decoupled:
            assert len(set(row["observations"])) == 1
