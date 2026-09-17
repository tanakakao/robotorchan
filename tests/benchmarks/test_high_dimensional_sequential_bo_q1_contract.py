"""q=1 contract for the analytic LogEI sequential benchmark."""

import importlib.util
import sys
from pathlib import Path

MODULE_NAME = "high_dimensional_sequential_bo_q1_contract"
MODULE_PATH = Path(__file__).parents[2] / "benchmarks" / "high_dimensional_sequential_bo.py"
SPEC = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = benchmark
SPEC.loader.exec_module(benchmark)


def test_sequential_benchmark_returns_one_candidate_per_iteration() -> None:
    rows = benchmark.run_strategy(
        "RandomSearch",
        6,
        n_train=4,
        n_iterations=2,
        latent_dim=2,
        random_samples=8,
        num_restarts=1,
        raw_samples=4,
        seed=17,
    )

    assert len(rows) == 2
    assert [row.iteration for row in rows] == [1, 2]
