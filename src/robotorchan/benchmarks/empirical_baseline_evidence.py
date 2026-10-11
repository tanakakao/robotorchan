"""Run frozen Random/Sobol baseline cohorts without GP acquisition overhead."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

import torch

from robotorchan.benchmarks.comparative_baselines import run_comparative_baseline
from robotorchan.benchmarks.comparative_matrix import ComparativeExperimentCell
from robotorchan.benchmarks.config import BenchmarkExperimentConfig
from robotorchan.benchmarks.initial_pairing import validate_initial_pairing
from robotorchan.benchmarks.metrics import simple_regret_curve
from robotorchan.benchmarks.persistence import load_trajectory, save_trajectory
from robotorchan.benchmarks.standard_problems import branin, hartmann6

_PROBLEMS = {"branin": branin, "hartmann6": hartmann6}
_STRATEGIES = ("random", "sobol")


def run_baseline_cohort(destination: Path, *, problem: str, q: int, git_sha: str) -> None:
    """Persist ten paired seeds and refuse to certify incomplete runs."""
    if problem not in _PROBLEMS or q not in (1, 3):
        raise ValueError("Unsupported problem or batch size.")
    if destination.exists():
        raise FileExistsError("Baseline evidence destination already exists.")
    destination.mkdir(parents=True)
    manifest = {
        "problem": problem,
        "q": q,
        "strategies": list(_STRATEGIES),
        "seeds": list(range(10)),
        "initial_points": 20,
        "evaluation_budget": 50,
        "dtype": "float64",
        "device": "cpu",
        "git_sha": git_sha,
        "status": "running",
    }

    def save_manifest() -> None:
        (destination / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

    save_manifest()
    rows = []
    trajectories = {}
    try:
        benchmark_problem = _PROBLEMS[problem]()
        for strategy in _STRATEGIES:
            config = BenchmarkExperimentConfig(
                problem=problem,
                strategy=strategy,
                seeds=tuple(range(10)),
                initial_points=20,
                evaluation_budget=50,
                q=q,
            )
            cell = ComparativeExperimentCell(tier="empirical_standard", config=config)
            runs = run_comparative_baseline(cell)
            if len(runs) != 10 or {run.seed for run in runs} != set(range(10)):
                raise ValueError("Missing or duplicate baseline seeds.")
            trajectories[strategy] = runs
            for run in runs:
                if (
                    run.initial_points != 20
                    or run.evaluation_count != 50
                    or run.completed_batches != tuple(range(q, 51, q)) + (
                        (50,) if 50 % q else ()
                    )
                ):
                    raise ValueError("Incorrect baseline budget or batch boundaries.")
                path = destination / f"{strategy}_seed{run.seed}.json"
                save_trajectory(run, path)
                restored = load_trajectory(path)
                torch.testing.assert_close(restored.X, run.X)
                torch.testing.assert_close(restored.Y_truth, run.Y_truth)
                curve = simple_regret_curve(benchmark_problem, restored)
                if curve.numel() != 70 or not torch.isfinite(curve).all():
                    raise ValueError("Invalid baseline regret curve.")
                rows.append(
                    {
                        "strategy": strategy,
                        "seed": run.seed,
                        "final_regret": float(curve[-1]),
                        "candidate_seconds": sum(restored.candidate_seconds),
                        "evaluation_seconds": sum(restored.evaluation_seconds),
                        "initial_evaluation_seconds": restored.initial_evaluation_seconds,
                    }
                )
        pairing = validate_initial_pairing(trajectories)
        if not pairing.design_paired or not pairing.truth_paired:
            raise ValueError("Baseline initial designs or truths are not paired.")
        with (destination / "baseline.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        manifest["status"] = "runtime_validated"
        manifest["pairing"] = {"design": True, "truth": True}
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        save_manifest()


def main() -> None:
    """Execute each frozen baseline stratum in an isolated evidence directory."""
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True)
    if dirty:
        raise RuntimeError("Run empirical evidence from a clean git checkout.")
    for problem in _PROBLEMS:
        for q in (1, 3):
            run_baseline_cohort(
                args.destination / f"{problem}_q{q}", problem=problem, q=q, git_sha=sha
            )


if __name__ == "__main__":
    main()
