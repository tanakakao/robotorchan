"""Paired multi-fidelity cost evidence without conflating fidelity and target quality."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import torch

from robotorchan.benchmarks.config import BenchmarkExperimentConfig
from robotorchan.benchmarks.multifidelity_problems import (
    multifidelity_oscillatory,
    multifidelity_quadratic,
    register_multifidelity_problems,
)
from robotorchan.benchmarks.registry import BenchmarkProblemRegistry
from robotorchan.benchmarks.runner import random_candidates, run_benchmark

_PROBLEMS = {
    "multifidelity_oscillatory": multifidelity_oscillatory,
    "multifidelity_quadratic": multifidelity_quadratic,
}


def run_cost_evidence(
    destination: str | Path, *, problem: str, q: int, git_sha: str
) -> dict[str, object]:
    """Record 10-seed target-fidelity regret and cumulative simulated evaluation cost."""
    if problem not in _PROBLEMS or q not in (1, 3) or not git_sha:
        raise ValueError("Unsupported cost evidence stratum.")
    root = Path(destination)
    if root.exists():
        raise FileExistsError("Refusing to overwrite cost evidence.")
    root.mkdir(parents=True)
    registry = BenchmarkProblemRegistry()
    register_multifidelity_problems(registry)
    definition = _PROBLEMS[problem]()
    manifest: dict[str, object] = {
        "problem": problem,
        "strategy": "random",
        "seeds": list(range(10)),
        "initial_points": 20,
        "evaluation_budget": 50,
        "q": q,
        "dtype": "float64",
        "device": "cpu",
        "git_sha": git_sha,
        "cost_type": "problem_defined_evaluation_cost_not_wall_clock",
        "quality_metric": "best_high_fidelity_target_at_sampled_design_coordinates",
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
            q=q,
        )
        runs = run_benchmark(
            config,
            random_candidates,
            registry=registry,
            restrict_candidate_problem=True,
            independent_rng_streams=True,
            isolate_strategy_rng=True,
        )
        if len(runs) != 10 or {run.seed for run in runs} != set(range(10)):
            raise ValueError("Missing or duplicated cost-evidence seeds.")
        rows = []
        for run in runs:
            if run.X.shape != (70, 2) or run.evaluation_count != 50:
                raise ValueError("Incorrect cost evidence budget.")
            if run.X.dtype != torch.float64 or run.costs.dtype != torch.float64:
                raise ValueError("Cost evidence must use float64.")
            if run.costs.shape != (70, 1) or (run.costs < 0).any():
                raise ValueError("Invalid evaluation costs.")
            expected = tuple(range(q, 51, q)) + ((50,) if 50 % q else ())
            if run.completed_batches != expected:
                raise ValueError("Incomplete cost-evidence batches.")
            definition._validate_X(run.X)
            target_X = run.X.clone()
            target_X[:, -1] = 1.0
            target_truth = definition.evaluate_truth(target_X).squeeze(-1)
            incumbent = torch.cummin(target_truth, dim=0).values
            cumulative_cost = run.cumulative_cost
            if not torch.isfinite(cumulative_cost).all():
                raise ValueError("Nonfinite cumulative evaluation cost.")
            if (cumulative_cost[1:] < cumulative_cost[:-1]).any():
                raise ValueError("Cumulative cost must be nondecreasing.")
            for index in range(run.X.shape[0]):
                rows.append(
                    {
                        "seed": run.seed,
                        "evaluation": index + 1,
                        "post_initial": max(0, index + 1 - 20),
                        "cumulative_cost": float(cumulative_cost[index]),
                        "best_target_fidelity_value": float(incumbent[index]),
                        "target_simple_regret": (
                            float((incumbent[index] - definition.optimal_value[0]).clamp_min(0))
                            if definition.optimal_value is not None
                            else ""
                        ),
                    }
                )
            if not all(math.isfinite(float(row["cumulative_cost"])) for row in rows):
                raise ValueError("Nonfinite cost curve.")
        with (root / "cost_curves.csv").open("w", newline="", encoding="utf-8") as handle:
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
