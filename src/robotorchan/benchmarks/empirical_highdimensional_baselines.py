"""Paired high-dimensional empirical baselines for controlled effective dimensions."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import torch

from robotorchan.benchmarks.comparative_baselines import run_comparative_baseline
from robotorchan.benchmarks.comparative_matrix import ComparativeExperimentCell
from robotorchan.benchmarks.config import BenchmarkExperimentConfig
from robotorchan.benchmarks.high_dimensional_problems import (
    interaction_chain_30,
    rotated_subspace_40,
    sparse_sphere_50,
)
from robotorchan.benchmarks.metrics import simple_regret_curve
from robotorchan.benchmarks.persistence import save_trajectory

_PROBLEMS = {
    "interaction_chain_30": interaction_chain_30,
    "rotated_subspace_40": rotated_subspace_40,
    "sparse_sphere_50": sparse_sphere_50,
}
_STRATEGIES = ("random", "sobol")


def run_highdimensional_baselines(
    destination: str | Path, *, problem: str, q: int, git_sha: str
) -> dict[str, object]:
    """Persist ten paired seeds of truth regret and candidate-generation timing."""
    if problem not in _PROBLEMS or q not in (1, 3) or not git_sha:
        raise ValueError("Unsupported high-dimensional empirical stratum.")
    root = Path(destination)
    if root.exists():
        raise FileExistsError("Refusing to overwrite high-dimensional evidence.")
    root.mkdir(parents=True)
    definition = _PROBLEMS[problem]()
    manifest: dict[str, object] = {
        "problem": problem,
        "dimension": definition.dimension,
        "strategies": list(_STRATEGIES),
        "seeds": list(range(10)),
        "initial_points": 20,
        "evaluation_budget": 50,
        "q": q,
        "dtype": "float64",
        "device": "cpu",
        "git_sha": git_sha,
        "status": "running",
    }
    manifest_path = root / "manifest.json"

    def persist() -> None:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    persist()
    rows = []
    initial_by_seed = {}
    try:
        for strategy in _STRATEGIES:
            cell = ComparativeExperimentCell(
                tier="empirical_standard",
                config=BenchmarkExperimentConfig(
                    problem=problem,
                    strategy=strategy,
                    seeds=tuple(range(10)),
                    initial_points=20,
                    evaluation_budget=50,
                    q=q,
                ),
            )
            runs = run_comparative_baseline(cell)
            if len(runs) != 10 or {run.seed for run in runs} != set(range(10)):
                raise ValueError("Missing or duplicated high-dimensional seeds.")
            for run in runs:
                if run.X.shape != (70, definition.dimension) or run.evaluation_count != 50:
                    raise ValueError("Incorrect high-dimensional trajectory shape or budget.")
                if run.X.dtype != torch.float64 or run.Y_truth.dtype != torch.float64:
                    raise ValueError("High-dimensional empirical evidence requires float64.")
                expected_batches = tuple(range(q, 51, q)) + ((50,) if 50 % q else ())
                if run.completed_batches != expected_batches:
                    raise ValueError("Incomplete high-dimensional batch sequence.")
                definition._validate_X(run.X)
                if run.seed in initial_by_seed:
                    torch.testing.assert_close(
                        run.X[:20], initial_by_seed[run.seed], rtol=0, atol=0
                    )
                else:
                    initial_by_seed[run.seed] = run.X[:20]
                regret = float(simple_regret_curve(definition, run)[-1])
                if not torch.isfinite(torch.tensor(regret)):
                    raise ValueError("Nonfinite high-dimensional regret.")
                save_trajectory(run, root / f"{strategy}_seed{run.seed}.json")
                rows.append(
                    {
                        "strategy": strategy,
                        "seed": run.seed,
                        "final_regret": regret,
                        "candidate_seconds": sum(run.candidate_seconds),
                        "evaluation_seconds": sum(run.evaluation_seconds),
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
