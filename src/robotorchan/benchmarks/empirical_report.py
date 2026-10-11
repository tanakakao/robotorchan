"""Build auditable empirical summaries from completed trajectory artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from robotorchan.benchmarks.metrics import hypervolume_curve, simple_regret_curve
from robotorchan.benchmarks.multiobjective_problems import branin_currin
from robotorchan.benchmarks.persistence import load_trajectory
from robotorchan.benchmarks.standard_problems import branin, hartmann6

_PROBLEMS = {"branin": branin, "hartmann6": hartmann6, "branin_currin": branin_currin}
_ALLOWED = {
    "branin": ("random", "sobol", "qEI", "qNEI"),
    "hartmann6": ("random", "sobol", "qEI", "qNEI"),
    "branin_currin": ("random", "sobol", "qEHVI", "qNEHVI"),
}


def summarize_empirical_directory(directory: str | Path) -> dict[str, object]:
    """Refuse incomplete cohorts; derive truth metrics from every paired seed."""
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("status") != "runtime_validated":
        raise ValueError("Only runtime-validated cohorts can be summarized.")
    problem_name = manifest.get("problem")
    if problem_name not in _PROBLEMS:
        raise ValueError("Unsupported empirical problem.")
    strategies = manifest.get("strategies")
    if strategies != list(_ALLOWED[problem_name]):
        raise ValueError("Unexpected or incomplete comparison strategies.")
    if (
        manifest.get("seeds") != list(range(10))
        or manifest.get("initial_points") != 20
        or manifest.get("evaluation_budget") != 50
        or manifest.get("q") not in (1, 3)
        or manifest.get("dtype") != "float64"
        or manifest.get("device") != "cpu"
        or not manifest.get("git_sha")
    ):
        raise ValueError("Manifest violates the frozen empirical protocol.")
    if manifest.get("pairing", {}).get("design") is not True:
        raise ValueError("Initial designs were not paired.")
    if manifest.get("pairing", {}).get("truth") is not True:
        raise ValueError("Initial truths were not paired.")
    noise = manifest.get("observation_noise")
    if (
        isinstance(noise, dict)
        and noise.get("policy") == "paired"
        and manifest["pairing"].get("observation") is not True
    ):
        raise ValueError("Paired observation noise did not match.")
    problem = _PROBLEMS[problem_name]()
    q = manifest["q"]
    scores: dict[str, list[float]] = {}
    times: dict[str, list[float]] = {}
    for strategy in strategies:
        scores[strategy] = []
        times[strategy] = []
        for seed in range(10):
            run = load_trajectory(root / f"{strategy}_seed{seed}.json")
            if (
                run.seed != seed
                or run.initial_points != 20
                or run.evaluation_count != 50
                or run.X.shape != (70, problem.dimension)
                or run.Y_truth.shape != (70, problem.n_objectives)
                or not run.completed_batches
                or run.completed_batches[-1] != 50
                or len(run.completed_batches) != (50 + q - 1) // q
                or len(run.candidate_seconds) != len(run.completed_batches)
                or len(run.evaluation_seconds) != len(run.completed_batches)
                or run.initial_evaluation_seconds is None
            ):
                raise ValueError(f"Incomplete trajectory: {strategy} seed {seed}.")
            if problem.n_objectives == 1:
                curve = simple_regret_curve(problem, run)
            else:
                curve = hypervolume_curve(problem, run)
            if not torch.isfinite(curve).all():
                raise ValueError("Nonfinite optimization metric.")
            scores[strategy].append(float(curve[-1].item()))
            times[strategy].append(
                float(
                    run.initial_evaluation_seconds
                    + sum(run.candidate_seconds)
                    + sum(run.evaluation_seconds)
                )
            )
    comparisons = []
    for baseline in ("random", "sobol"):
        for strategy in strategies:
            if strategy == baseline:
                continue
            differences = torch.tensor(scores[strategy], dtype=torch.double)
            differences -= torch.tensor(scores[baseline], dtype=torch.double)
            if problem.n_objectives == 1:
                differences = -differences
            generator = torch.Generator().manual_seed(2026)
            indices = torch.randint(10, (10000, 10), generator=generator)
            bootstrap = differences[indices].mean(dim=1)
            bounds = torch.quantile(bootstrap, torch.tensor([0.025, 0.975]))
            comparisons.append(
                {
                    "strategy": strategy,
                    "baseline": baseline,
                    "mean_paired_improvement": float(differences.mean().item()),
                    "bootstrap_ci95": bounds.tolist(),
                    "n_paired": 10,
                }
            )
    report: dict[str, object] = {
        "status": "complete",
        "problem": problem_name,
        "q": q,
        "git_sha": manifest["git_sha"],
        "noise": noise,
        "metric": "simple_regret" if problem.n_objectives == 1 else "hypervolume",
        "final_metric_by_seed": scores,
        "elapsed_seconds_by_seed": times,
        "comparisons": comparisons,
        "seed_count": 10,
    }
    return report


def save_empirical_report(directory: str | Path, destination: str | Path) -> None:
    """Write a verified report without overwriting existing evidence."""
    output = Path(destination)
    if output.exists():
        raise FileExistsError("Refusing to overwrite an existing empirical report.")
    report = summarize_empirical_directory(directory)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
