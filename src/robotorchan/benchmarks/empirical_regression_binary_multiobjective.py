"""Paired two-objective regression and binary feasibility empirical evidence."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import torch

from robotorchan.benchmarks.comparative_baselines import run_comparative_baseline
from robotorchan.benchmarks.comparative_matrix import ComparativeExperimentCell
from robotorchan.benchmarks.config import BenchmarkExperimentConfig
from robotorchan.benchmarks.constrained_evidence import (
    observed_pass_labels,
    validate_constrained_trajectory,
)
from robotorchan.benchmarks.heterogeneous_problems import strength_conductivity_pass
from robotorchan.benchmarks.metrics import hypervolume_2d

_STRATEGIES = ("random", "sobol")


def run_regression_binary_multiobjective_evidence(
    destination: str | Path, *, q: int, git_sha: str
) -> dict[str, object]:
    """Record paired 10-seed feasible hypervolume and canonical Pass labels."""
    if q not in (1, 3) or not git_sha:
        raise ValueError("Frozen multiobjective evidence requires q=1/3 and git_sha.")
    root = Path(destination)
    if root.exists():
        raise FileExistsError("Refusing to overwrite heterogeneous evidence.")
    root.mkdir(parents=True)
    problem = strength_conductivity_pass()
    manifest: dict[str, object] = {
        "problem": problem.name,
        "strategies": list(_STRATEGIES),
        "seeds": list(range(10)),
        "initial_points": 20,
        "evaluation_budget": 50,
        "q": q,
        "dtype": "float64",
        "device": "cpu",
        "git_sha": git_sha,
        "objective": "two_regression_maximize",
        "constraint": "canonical_binary_pass_label",
        "constraint_margins": "evaluation_only",
        "metric": "feasible_truth_hypervolume",
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
                problem=problem.name,
                strategy=strategy,
                seeds=tuple(range(10)),
                initial_points=20,
                evaluation_budget=50,
                q=q,
            )
            cell = ComparativeExperimentCell(tier="empirical_standard", config=config)
            runs = run_comparative_baseline(cell)
            if len(runs) != 10 or {run.seed for run in runs} != set(range(10)):
                raise ValueError("Missing or duplicate heterogeneous seeds.")
            for run in runs:
                if run.X.shape != (70, 2) or run.Y_truth.shape != (70, 2):
                    raise ValueError("Incorrect multiobjective trajectory dimensions.")
                if run.X.dtype != torch.float64 or run.evaluation_count != 50:
                    raise ValueError("Incorrect multiobjective empirical budget or dtype.")
                expected = tuple(range(q, 51, q)) + ((50,) if 50 % q else ())
                if run.completed_batches != expected:
                    raise ValueError("Incomplete multiobjective evaluation batches.")
                if run.seed in paired_initial:
                    torch.testing.assert_close(run.X[:20], paired_initial[run.seed], rtol=0, atol=0)
                else:
                    paired_initial[run.seed] = run.X[:20]
                validation = validate_constrained_trajectory(problem.name, strategy, run)
                labels = observed_pass_labels(problem.name, run.X).squeeze(-1)
                feasible = labels.bool()
                reference = problem.reference_point.to(dtype=run.Y_truth.dtype)
                for index in range(70):
                    selected = run.Y_truth[: index + 1][feasible[: index + 1]]
                    hv = float(hypervolume_2d(selected, reference))
                    rows.append(
                        {
                            "strategy": strategy,
                            "seed": run.seed,
                            "evaluation": index + 1,
                            "post_initial": max(0, index + 1 - 20),
                            "observed_pass_label": int(labels[index]),
                            "feasible_count": int(feasible[: index + 1].sum()),
                            "zero_feasible": not bool(feasible[: index + 1].any()),
                            "feasible_truth_hypervolume": hv,
                        }
                    )
                if validation["feasible_count"] != int(feasible.sum()):
                    raise ValueError("Binary labels disagree with evaluation feasibility.")
        with (root / "feasible_hypervolume_curves.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
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
