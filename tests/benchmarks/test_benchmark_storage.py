"""Benchmark result storage round-trip and schema validation tests."""

import json

import pytest
import torch

from robotorchan.benchmarks.config import BenchmarkExperimentConfig
from robotorchan.benchmarks.runner import random_candidates, run_benchmark
from robotorchan.benchmarks.storage import load_benchmark_results, save_benchmark_results


def test_storage_roundtrip(tmp_path) -> None:
    config = BenchmarkExperimentConfig(
        problem="branin",
        strategy="random",
        seeds=(2, 4),
        initial_points=3,
        evaluation_budget=5,
        q=2,
    )
    trajectories = run_benchmark(config, random_candidates)
    path = tmp_path / "nested" / "results.json"
    save_benchmark_results(path, config, trajectories)
    loaded_config, loaded = load_benchmark_results(path)
    assert loaded_config == config
    assert len(loaded) == len(trajectories)
    for original, restored in zip(trajectories, loaded, strict=True):
        assert original.seed == restored.seed
        assert original.initial_points == restored.initial_points
        for name in ("X", "Y_observed", "Y_truth", "constraints", "costs"):
            torch.testing.assert_close(getattr(original, name), getattr(restored, name))


def test_storage_rejects_incorrect_seed(tmp_path) -> None:
    config = BenchmarkExperimentConfig(problem="branin", strategy="random", seeds=(1,))
    with pytest.raises(ValueError, match="seeds"):
        save_benchmark_results(tmp_path / "results.json", config, ())


@pytest.mark.parametrize(
    "mutation",
    ["version", "seed", "shape", "nan"],
)
def test_storage_rejects_corrupt_records(tmp_path, mutation: str) -> None:
    config = BenchmarkExperimentConfig(
        problem="branin",
        strategy="random",
        seeds=(1,),
        initial_points=3,
        evaluation_budget=2,
    )
    runs = run_benchmark(config, random_candidates)
    path = tmp_path / "results.json"
    save_benchmark_results(path, config, runs)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if mutation == "version":
        payload["schema_version"] = 999
    elif mutation == "seed":
        payload["trajectories"][0]["seed"] = 8
    elif mutation == "shape":
        payload["trajectories"][0]["X"]["shape"] = [100, 2]
    else:
        payload["trajectories"][0]["X"]["values"][0][0] = float("nan")
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_benchmark_results(path)
