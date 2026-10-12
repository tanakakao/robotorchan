"""Evaluate sequential, batch and async random baselines on a shared budget."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import torch

from robotorchan.benchmarks.config import BenchmarkExperimentConfig
from robotorchan.benchmarks.parallel_comparison import run_parallel_comparison
from robotorchan.benchmarks.standard_problems import branin, hartmann6

_PROBLEMS = {"branin": branin, "hartmann6": hartmann6}
_ARMS = ("sequential", "batch", "asynchronous")


def run_batch_async_evidence(
    destination: str | Path,
    *,
    problem: str,
    git_sha: str,
    batch_size: int = 3,
    max_concurrency: int = 3,
) -> dict[str, object]:
    """Persist paired 10-seed truth regret and separate simulated async makespan."""
    if problem not in _PROBLEMS or not git_sha:
        raise ValueError("Unsupported batch/async empirical problem or source revision.")
    if batch_size != 3 or max_concurrency != 3:
        raise ValueError("Frozen empirical batch and concurrency are both three.")
    root = Path(destination)
    if root.exists():
        raise FileExistsError("Refusing to overwrite batch/async evidence.")
    root.mkdir(parents=True)
    manifest: dict[str, object] = {
        "problem": problem,
        "strategies": list(_ARMS),
        "seeds": list(range(10)),
        "initial_points": 20,
        "evaluation_budget": 50,
        "batch_size": batch_size,
        "max_concurrency": max_concurrency,
        "dtype": "float64",
        "device": "cpu",
        "git_sha": git_sha,
        "duration_model": "constant_one_simulated_unit",
        "status": "running",
    }
    manifest_path = root / "manifest.json"

    def persist() -> None:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    persist()
    try:
        config = BenchmarkExperimentConfig(
            problem=problem,
            strategy="random",
            seeds=tuple(range(10)),
            initial_points=20,
            evaluation_budget=50,
            q=3,
        )
        comparison = run_parallel_comparison(
            config, batch_size=batch_size, max_concurrency=max_concurrency
        )
        definition = _PROBLEMS[problem]()
        rows = []
        paired = {}
        for arm in _ARMS:
            runs = getattr(comparison, arm)
            if len(runs) != 10 or {run.seed for run in runs} != set(range(10)):
                raise ValueError("Missing or duplicate batch/async seeds.")
            for run in runs:
                if run.X.shape != (70, definition.dimension):
                    raise ValueError("Incorrect batch/async evaluation budget.")
                if run.X.dtype != torch.float64 or run.Y_truth.dtype != torch.float64:
                    raise ValueError("Batch/async empirical evidence requires float64.")
                if run.seed in paired:
                    torch.testing.assert_close(run.X[:20], paired[run.seed], rtol=0, atol=0)
                else:
                    paired[run.seed] = run.X[:20]
                if arm == "asynchronous":
                    if len(run.evaluations) != 50:
                        raise ValueError("Incorrect async event count.")
                    if run.max_concurrency > max_concurrency:
                        raise ValueError("Async worker capacity exceeded.")
                    makespan = run.simulated_makespan
                else:
                    expected_q = 1 if arm == "sequential" else batch_size
                    expected = tuple(range(expected_q, 51, expected_q))
                    if 50 % expected_q:
                        expected += (50,)
                    if run.completed_batches != expected:
                        raise ValueError("Incorrect synchronous batch boundaries.")
                    makespan = None
                regret = float(definition.simple_regret(run.X))
                if not math.isfinite(regret):
                    raise ValueError("Nonfinite batch/async regret.")
                rows.append(
                    {
                        "arm": arm,
                        "seed": run.seed,
                        "final_truth_regret": regret,
                        "simulated_makespan": makespan,
                    }
                )
        with (root / "seed_results.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        manifest["status"] = "runtime_validated"
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        persist()
    return manifest
