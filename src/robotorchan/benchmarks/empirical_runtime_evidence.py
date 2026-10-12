"""Measured candidate and evaluation runtime evidence for paired baselines."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import torch

from robotorchan.benchmarks.comparative_baselines import run_comparative_baseline
from robotorchan.benchmarks.comparative_matrix import ComparativeExperimentCell
from robotorchan.benchmarks.config import BenchmarkExperimentConfig

_STRATEGIES = ("random", "sobol")
_PROBLEMS = ("branin", "hartmann6")


def runtime_breakdown(run: object) -> dict[str, float | int]:
    """Validate timing arrays and separate candidate from evaluation time."""
    candidate = run.candidate_seconds
    evaluation = run.evaluation_seconds
    initial = run.initial_evaluation_seconds
    if len(candidate) != len(run.completed_batches) or len(evaluation) != len(candidate):
        raise ValueError("Missing per-batch timing observations.")
    if not candidate or initial is None:
        raise ValueError("Missing initial or candidate runtime observations.")
    times = (*candidate, *evaluation, initial)
    if any(not math.isfinite(value) or value < 0 for value in times):
        raise ValueError("Invalid nonnegative wall-clock timings.")
    return {
        "candidate_seconds": float(sum(candidate)),
        "evaluation_seconds": float(sum(evaluation)),
        "initial_evaluation_seconds": float(initial),
        "total_measured_seconds": float(sum(candidate) + sum(evaluation) + initial),
        "candidate_batches": len(candidate),
        "evaluation_count": run.evaluation_count,
    }


def run_runtime_evidence(
    destination: str | Path, *, problem: str, q: int, git_sha: str
) -> dict[str, object]:
    """Persist wall-clock time breakdown for 10 paired Random/Sobol seeds."""
    if problem not in _PROBLEMS or q not in (1, 3) or not git_sha:
        raise ValueError("Unsupported runtime evidence stratum.")
    root = Path(destination)
    if root.exists():
        raise FileExistsError("Refusing to overwrite runtime evidence.")
    root.mkdir(parents=True)
    manifest: dict[str, object] = {
        "problem": problem,
        "strategies": list(_STRATEGIES),
        "seeds": list(range(10)),
        "initial_points": 20,
        "evaluation_budget": 50,
        "q": q,
        "dtype": "float64",
        "device": "cpu",
        "git_sha": git_sha,
        "runtime_kind": "perf_counter_wall_clock_not_simulated_makespan_or_evaluation_cost",
        "excludes": ["initial_design_generation", "report_serialization", "process_startup"],
        "status": "running",
    }
    path = root / "manifest.json"

    def persist() -> None:
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    persist()
    rows = []
    paired_initial = {}
    try:
        for strategy in _STRATEGIES:
            config = BenchmarkExperimentConfig(
                problem=problem,
                strategy=strategy,
                seeds=tuple(range(10)),
                initial_points=20,
                evaluation_budget=50,
                q=q,
            )
            runs = run_comparative_baseline(
                ComparativeExperimentCell(tier="empirical_standard", config=config)
            )
            if len(runs) != 10 or {run.seed for run in runs} != set(range(10)):
                raise ValueError("Missing or duplicate runtime seeds.")
            for run in runs:
                if run.X.dtype != torch.float64 or run.evaluation_count != 50:
                    raise ValueError("Incorrect runtime evidence dtype or budget.")
                expected = tuple(range(q, 51, q)) + ((50,) if 50 % q else ())
                if run.completed_batches != expected:
                    raise ValueError("Incomplete runtime evidence batches.")
                if run.seed in paired_initial:
                    torch.testing.assert_close(run.X[:20], paired_initial[run.seed], rtol=0, atol=0)
                else:
                    paired_initial[run.seed] = run.X[:20]
                rows.append({"strategy": strategy, "seed": run.seed, **runtime_breakdown(run)})
        with (root / "runtime_by_seed.csv").open("w", newline="", encoding="utf-8") as handle:
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
