import math

import pytest

pytest.importorskip("sklearn")

import importlib.util
from pathlib import Path

_BENCHMARK_PATH = Path(__file__).parents[2] / "benchmarks" / "non_gp_surrogates.py"
_SPEC = importlib.util.spec_from_file_location("non_gp_surrogates_benchmark", _BENCHMARK_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
make_data = _MODULE.make_data
run_benchmark = _MODULE.run_benchmark


def test_non_gp_benchmark_data_is_reproducible() -> None:
    first = make_data(n_train=12, n_test=8, seed=11)
    second = make_data(n_train=12, n_test=8, seed=11)

    for left, right in zip(first, second, strict=True):
        assert left.equal(right)


def test_non_gp_benchmark_reports_finite_metrics() -> None:
    results = run_benchmark(seed=3)

    assert {result.model for result in results} == {
        "RandomForest",
        "ExtraTrees",
        "GradientBoosting",
        "HistGradientBoosting",
    }
    for result in results:
        assert math.isfinite(result.rmse)
        assert result.rmse >= 0.0
        assert math.isfinite(result.mean_posterior_std)
        assert result.mean_posterior_std >= 0.0
        assert result.fit_seconds >= 0.0
        assert result.posterior_seconds >= 0.0
