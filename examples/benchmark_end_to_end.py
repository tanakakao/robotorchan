"""Runnable end-to-end benchmark example with seed-aligned metrics and storage."""

from __future__ import annotations

from pathlib import Path

from robotorchan.benchmarks import (
    BenchmarkProblemRegistry,
    get_execution_profile,
    load_benchmark_results,
    make_metric_report,
    random_candidates,
    register_standard_problems,
    run_benchmark,
    save_benchmark_results,
    simple_regret_curve,
)


def main(output_dir: Path) -> None:
    """Run Branin random search and persist its trajectories and metric report."""
    registry = BenchmarkProblemRegistry()
    register_standard_problems(registry)
    config = get_execution_profile("smoke").make_config("branin", "random")
    trajectories = run_benchmark(config, random_candidates, registry=registry)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "branin_random_results.json"
    save_benchmark_results(path, config, trajectories)
    loaded_config, loaded = load_benchmark_results(path)
    assert loaded_config == config

    problem = registry.create(config.problem)
    curves = {run.seed: simple_regret_curve(problem, run) for run in loaded}
    report = make_metric_report(
        strategy=config.strategy,
        metric="simple_regret",
        curves=curves,
    )
    (output_dir / "branin_random_report.json").write_text(
        report.to_json() + "\n", encoding="utf-8"
    )
    (output_dir / "branin_random_report.csv").write_text(
        report.to_csv(), encoding="utf-8"
    )
    print(f"Saved {len(loaded)} trajectory and metric report to {output_dir}")


if __name__ == "__main__":
    main(Path("benchmark_outputs"))
