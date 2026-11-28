import pytest
import torch
from botorch.utils.multi_objective.hypervolume import Hypervolume

from robotorchan.benchmarks.decoupled import _finite_pair_hypervolume, run_decoupled_benchmark
from robotorchan.benchmarks.multiobjective_problems import branin_currin


@pytest.mark.parametrize("decoupled", [False, True])
@pytest.mark.parametrize("policy", ["epig", "kg", "hvkg", "random", "sobol"])
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


def test_hvkg_baseline_uses_two_recommendations():
    mean = torch.tensor([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]], dtype=torch.double)
    hv = Hypervolume(mean.new_zeros(2))
    assert hv.compute(mean) == pytest.approx(6.0)
    assert _finite_pair_hypervolume(mean, hv) == pytest.approx(5.0)
    assert _finite_pair_hypervolume(mean[:1], hv) == pytest.approx(3.0)
