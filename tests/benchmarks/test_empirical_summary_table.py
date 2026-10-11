"""Tests for comparison summary tables and CSV export."""

from __future__ import annotations

import csv

from robotorchan.benchmarks.comparison_report import build_comparison_report
from robotorchan.benchmarks.config import BenchmarkExperimentConfig
from robotorchan.benchmarks.registry import BenchmarkProblemRegistry
from robotorchan.benchmarks.runner import random_candidates, run_benchmark
from robotorchan.benchmarks.standard_problems import register_standard_problems
from robotorchan.benchmarks.summary_table import (
    save_comparison_table_csv,
    summarize_comparison_table,
)


def test_summary_table_preserves_final_statistics(tmp_path) -> None:
    registry = BenchmarkProblemRegistry()
    register_standard_problems(registry)
    experiments = {}
    for name in ("random", "sobol"):
        config = BenchmarkExperimentConfig(
            problem="branin",
            strategy=name,
            seeds=(0, 1, 2),
            initial_points=4,
            evaluation_budget=5,
            q=3,
        )
        runs = run_benchmark(
            config,
            random_candidates,
            registry=registry,
            independent_rng_streams=True,
            restrict_candidate_problem=True,
        )
        experiments[name] = (config, runs)
    report = build_comparison_report(
        registry.create("branin"),
        experiments,
        metric="simple_regret",
        reference="random",
        bootstrap_samples=100,
    )
    rows = summarize_comparison_table(report)
    assert len(rows) == 2
    assert [row.strategy for row in rows] == ["random", "sobol"]
    reference, alternative = rows
    assert reference.lower_is_better
    assert reference.paired_mean_difference is None
    assert reference.adjusted_p_value is None
    assert alternative.n_seeds == 3
    assert alternative.paired_mean_difference is not None
    assert alternative.adjusted_p_value is not None
    assert alternative.final_mean == report.strategies["sobol"]["mean"][-1]

    path = tmp_path / "summary.csv"
    save_comparison_table_csv(rows, path)
    with path.open(encoding="utf-8", newline="") as stream:
        records = list(csv.DictReader(stream))
    assert len(records) == 2
    assert records[1]["strategy"] == "sobol"
    assert records[1]["paired_mean_difference"] != ""
    assert records[0]["paired_mean_difference"] == ""
