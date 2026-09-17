"""RNG stream contract for the continuous acquisition benchmark."""

import importlib.util
import sys
from pathlib import Path

MODULE_NAME = "high_dimensional_acqf_optimization_seed_contract"
MODULE_PATH = Path(__file__).parents[2] / "benchmarks" / "high_dimensional_acqf_optimization.py"
SPEC = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = benchmark
SPEC.loader.exec_module(benchmark)


def test_problem_seed_and_search_seeds_are_distinct() -> None:
    problem_seed = 23
    _, bounds, _ = benchmark.make_problem(5, n_train=4, seed=problem_seed)
    strategies = benchmark.strategy_factories(
        bounds,
        embedding_dim=3,
        random_samples=8,
        num_restarts=1,
        raw_samples=4,
        seed=problem_seed,
    )
    random_strategy = strategies["RandomSearch"]
    rembo_strategy = strategies["REMBO"]
    hesbo_strategy = strategies["HeSBO"]

    seeds = {
        problem_seed,
        benchmark._search_seed(problem_seed),
        benchmark._embedding_seed(problem_seed),
        benchmark._hesbo_seed(problem_seed),
    }
    assert len(seeds) == 4
    assert random_strategy.seed == benchmark._search_seed(problem_seed)
    assert rembo_strategy.embedding.shape == (5, 3)
    assert hesbo_strategy.embedding.shape == (5, 3)
