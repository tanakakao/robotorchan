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


def test_problem_seed_and_random_search_seed_are_distinct() -> None:
    problem_seed = 23
    _, bounds, _ = benchmark.make_problem(5, n_train=4, seed=problem_seed)
    strategy = benchmark.strategy_factories(
        bounds,
        random_samples=8,
        num_restarts=1,
        raw_samples=4,
        seed=problem_seed,
    )["RandomSearch"]

    assert strategy.seed == benchmark._search_seed(problem_seed)
    assert strategy.seed != problem_seed
