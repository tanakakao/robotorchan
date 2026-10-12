"""Execute frozen paired constrained baselines with feasibility-aware reporting."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import torch

from robotorchan.benchmarks.comparative_baselines import run_comparative_baseline
from robotorchan.benchmarks.comparative_matrix import ComparativeExperimentCell
from robotorchan.benchmarks.config import BenchmarkExperimentConfig
from robotorchan.benchmarks.constrained_problems import constrained_quadratic
from robotorchan.benchmarks.metrics import feasibility_rate, simple_regret_curve
from robotorchan.benchmarks.persistence import save_trajectory
from robotorchan.benchmarks.runner import BenchmarkTrajectory

STRATEGIES = ("random", "sobol")


def constrained_baseline_metrics(
    trajectory: BenchmarkTrajectory,
) -> dict[str, float | bool | int | None]:
    """Keep zero-feasible runs explicit instead of reporting infinite regret."""
    if trajectory.constraints.shape != (70, 1):
        raise ValueError("Expected one constraint margin per evaluation.")
    feasible = (trajectory.constraints >= 0).all(dim=-1)
    count = int(feasible.sum().item())
    regret = simple_regret_curve(constrained_quadratic(), trajectory)
    final_regret = float(regret[-1]) if count else None
    return {
        "feasible_count": count,
        "feasible_rate": float(feasibility_rate(trajectory.constraints)),
        "zero_feasible": count == 0,
        "final_feasible_regret": final_regret,
    }


def run_constrained_baselines(
    destination: str | Path, *, q: int, git_sha: str
) -> dict[str, object]:
    """Run Random/Sobol paired seeds for constrained quadratic, retaining all runs."""
    if q not in (1, 3) or not git_sha:
        raise ValueError("Expected q=1/3 and an explicit source revision.")
    root = Path(destination)
    if root.exists():
        raise FileExistsError("Refusing to overwrite constrained evidence.")
    root.mkdir(parents=True)
    manifest: dict[str, object] = {
        "problem": "constrained_quadratic",
        "strategies": list(STRATEGIES),
        "seeds": list(range(10)),
        "initial_points": 20,
        "evaluation_budget": 50,
        "q": q,
        "dtype": "float64",
        "device": "cpu",
        "git_sha": git_sha,
        "status": "running",
    }
    path = root / "manifest.json"

    def persist() -> None:
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    persist()
    rows = []
    initial_by_seed = {}
    try:
        for strategy in STRATEGIES:
            config = BenchmarkExperimentConfig(
                problem="constrained_quadratic",
                strategy=strategy,
                seeds=tuple(range(10)),
                initial_points=20,
                evaluation_budget=50,
                q=q,
            )
            cell = ComparativeExperimentCell(tier="empirical_standard", config=config)
            runs = run_comparative_baseline(cell)
            if len(runs) != 10 or {run.seed for run in runs} != set(range(10)):
                raise ValueError("Missing or duplicated constrained seeds.")
            for run in runs:
                if run.initial_points != 20 or run.evaluation_count != 50:
                    raise ValueError("Incorrect constrained evaluation budget.")
                if run.Y_truth.shape != (70, 1) or run.X.dtype != torch.float64:
                    raise ValueError("Incorrect constrained trajectory shape or dtype.")
                if not run.completed_batches or run.completed_batches[-1] != 50:
                    raise ValueError("Incomplete constrained batches.")
                if run.seed in initial_by_seed:
                    torch.testing.assert_close(
                        run.X[:20], initial_by_seed[run.seed], rtol=0, atol=0
                    )
                else:
                    initial_by_seed[run.seed] = run.X[:20]
                save_trajectory(run, root / f"{strategy}_seed{run.seed}.json")
                rows.append(
                    {"strategy": strategy, "seed": run.seed, **constrained_baseline_metrics(run)}
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
