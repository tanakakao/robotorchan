"""Focused reproducibility and budget tests for combinatorial benchmark."""

import importlib.util
from pathlib import Path


def _load_benchmark():
    path = Path(__file__).resolve().parents[2] / "examples" / "combinatorial_bo_benchmark.py"
    spec = importlib.util.spec_from_file_location("combinatorial_bo_benchmark", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_seed_aligned_budget_and_unique_proposals(monkeypatch):
    benchmark = _load_benchmark()
    methods = (
        "random",
        "sobol",
        "numeric_gp",
        "mixed_gp",
        "mixed_ga",
        "hamming",
        "discrete_tr",
        "bocs",
        "combo",
    )
    assert methods == benchmark.METHODS

    def next_unseen(method, X, Y, domain, seed, iteration):
        del method, Y, seed, iteration
        unseen = ~(domain[:, None, :] == X[None, :, :]).all(dim=-1).any(dim=-1)
        return domain[unseen][:1]

    monkeypatch.setattr(benchmark, "propose", next_unseen)
    rows = benchmark.run_benchmark(seeds=2, steps=2, dimensions=3, initial=2)
    assert len(rows) == 2 * 2 * len(methods)
    for seed in range(2):
        for method in methods:
            series = [row for row in rows if row["seed"] == seed and row["method"] == method]
            assert [row["evaluations"] for row in series] == [3, 4]
            assert series[1]["best_value"] >= series[0]["best_value"]


def test_random_and_sobol_propose_unseen_points():
    benchmark = _load_benchmark()
    domain = benchmark.binary_domain(3)
    X = domain[:2]
    Y = benchmark.oracle(X)
    for method in ("random", "sobol"):
        proposal = benchmark.propose(method, X, Y, domain, seed=1, iteration=0)
        assert proposal.shape == (1, 3)
        assert not bool((proposal == X).all(dim=-1).any())


def test_mixed_gp_and_mixed_ga_real_proposals():
    """Exercise both Phase 9 additions without replacing their optimizers."""
    benchmark = _load_benchmark()
    domain = benchmark.binary_domain(3)
    X = domain[[0, 3, 6]]
    Y = benchmark.oracle(X)
    for method in ("mixed_gp", "mixed_ga"):
        candidate = benchmark.propose(method, X, Y, domain, seed=0, iteration=0)
        assert candidate.shape == (1, 3)
        assert bool((domain == candidate).all(dim=-1).any())
        assert not bool((X == candidate).all(dim=-1).any())
