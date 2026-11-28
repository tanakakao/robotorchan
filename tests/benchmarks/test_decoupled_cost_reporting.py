import pytest
import torch

from robotorchan.benchmarks.decoupled_reporting import align_cost_curves, summarize_cost_curves


def test_cost_alignment_and_paired_bootstrap():
    records = []
    for seed in (0, 1):
        for method, points in (("a", [(2, 1), (4, 3)]), ("b", [(2, 1), (3, 2)])):
            records.extend(
                {"seed": seed, "method": method, "cost": cost, "utility": value + seed}
                for cost, value in points
            )
    curves = align_cost_curves(records, "utility", [2, 3, 4, 5])
    torch.testing.assert_close(curves["a"][0], torch.tensor([1, 1, 3, 3], dtype=torch.double))
    rows = summarize_cost_curves(records, "utility", [2, 3, 4, 5], baseline="b", n_resamples=20)
    selected = [row for row in rows if row["method"] == "a"]
    assert [row["difference"] for row in selected] == [0, -1, 1, 1]
    assert all(row["n_seeds"] == 2 for row in rows)
    with pytest.raises(ValueError, match="precedes"):
        align_cost_curves(records, "utility", [1, 2])
    with pytest.raises(ValueError, match="Missing"):
        align_cost_curves(records, "simple_regret", [2, 3])
    with pytest.raises(ValueError, match="Duplicate"):
        align_cost_curves(records + records[:1], "utility", [2, 3])
