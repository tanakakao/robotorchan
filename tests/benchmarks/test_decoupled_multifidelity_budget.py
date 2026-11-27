import pytest

from robotorchan.benchmarks.decoupled_multifidelity import run_multifidelity_decoupled_benchmark


@pytest.mark.parametrize("decoupled", [False, True])
@pytest.mark.parametrize("multifidelity", [False, True])
def test_multifidelity_budget(decoupled, multifidelity):
    rows = run_multifidelity_decoupled_benchmark(
        budget=36,
        n_initial=2,
        num_samples=4,
        decoupled=decoupled,
        multifidelity=multifidelity,
        fit_model=False,
    )
    assert rows[0]["cost"] == 24
    assert len(rows) >= 2
    for row in rows:
        assert row["cost"] <= 36
        assert row["simple_regret"] >= 0
        assert sum(row["fidelity_counts"]) == sum(row["observations"])
        if not multifidelity:
            assert row["fidelity_counts"][0] == 0
        if not decoupled:
            assert row["observations"][0] == row["observations"][1]
    if multifidelity and decoupled:
        assert rows[-1]["cost"] == 36
