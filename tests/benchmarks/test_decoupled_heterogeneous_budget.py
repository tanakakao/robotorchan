import pytest

from robotorchan.benchmarks.decoupled_heterogeneous import run_heterogeneous_decoupled_benchmark


@pytest.mark.parametrize("kind", ["binary", "multiclass", "ordinal"])
@pytest.mark.parametrize("decoupled", [False, True])
def test_heterogeneous_budget(kind, decoupled):
    rows = run_heterogeneous_decoupled_benchmark(
        kind=kind,
        budget=16,
        n_initial=3,
        num_samples=2,
        epochs=2,
        decoupled=decoupled,
        fit_regression=False,
    )
    assert rows[0]["cost"] == 12
    assert rows[-1]["cost"] == 16
    for row in rows:
        assert row["cost"] == 3 * row["observations"][0] + row["observations"][1]
        assert 0 <= row["brier_score"] <= 2
        assert -1 <= row["utility"] <= 1
