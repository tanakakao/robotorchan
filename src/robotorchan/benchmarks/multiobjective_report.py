"""Validate and summarize measured Branin-Currin multiobjective trajectories."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import torch

from robotorchan.benchmarks.metrics import hypervolume_curve
from robotorchan.benchmarks.multiobjective_problems import branin_currin
from robotorchan.benchmarks.persistence import load_trajectory

STRATEGIES = ("random", "sobol", "qEHVI", "qNEHVI")
_COLUMNS = ("strategy", "seed", "final_hypervolume", "candidate_seconds", "evaluation_seconds")


def summarize_multiobjective_evidence(directory: str | Path) -> list[dict[str, object]]:
    """Reject incomplete or unpaired runs before writing hypervolume summaries."""
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "runtime_validated"
        or manifest.get("problem") != "branin_currin"
        or manifest.get("q") not in (1, 3)
        or manifest.get("strategies") != list(STRATEGIES)
        or manifest.get("seeds") != list(range(10))
        or manifest.get("initial_points") != 20
        or manifest.get("evaluation_budget") != 50
        or manifest.get("dtype") != "float64"
        or manifest.get("device") != "cpu"
        or manifest.get("dirty") is not False
    ):
        raise ValueError("Incomplete or nonconforming multiobjective evidence.")
    problem = branin_currin()
    if manifest.get("reference_point") != problem.reference_point.tolist():
        raise ValueError("Hypervolume reference point differs from frozen problem.")
    if manifest.get("objective_directions") != list(problem.directions):
        raise ValueError("Objective directions differ from frozen problem.")
    expected = {f"{strategy}_seed{seed}.json" for strategy in STRATEGIES for seed in range(10)}
    if {p.name for p in root.glob("*_seed*.json")} != expected:
        raise ValueError("Missing or unexpected multiobjective trajectories.")
    rows = []
    initial_by_seed = {}
    for strategy in STRATEGIES:
        for seed in range(10):
            trajectory = load_trajectory(root / f"{strategy}_seed{seed}.json")
            if trajectory.seed != seed or trajectory.initial_points != 20:
                raise ValueError("Incorrect trajectory identity.")
            if trajectory.evaluation_count != 50 or trajectory.Y_truth.shape != (70, 2):
                raise ValueError("Incorrect multiobjective trajectory shape or budget.")
            if trajectory.Y_truth.dtype != torch.float64:
                raise ValueError("Multiobjective evidence must use float64.")
            batches = trajectory.completed_batches
            expected_batches = tuple(range(manifest["q"], 51, manifest["q"]))
            if 50 % manifest["q"]:
                expected_batches += (50,)
            if batches != expected_batches:
                raise ValueError("Incorrect candidate batch boundaries.")
            initial = trajectory.X[:20]
            if seed in initial_by_seed:
                torch.testing.assert_close(initial, initial_by_seed[seed], rtol=0, atol=0)
            else:
                initial_by_seed[seed] = initial
            value = float(hypervolume_curve(problem, trajectory)[-1])
            if not math.isfinite(value):
                raise ValueError("Nonfinite measured hypervolume.")
            rows.append(
                {
                    "strategy": strategy,
                    "seed": seed,
                    "final_hypervolume": value,
                    "candidate_seconds": sum(trajectory.candidate_seconds),
                    "evaluation_seconds": sum(trajectory.evaluation_seconds),
                }
            )
    summary = []
    for strategy in STRATEGIES:
        group = [row for row in rows if row["strategy"] == strategy]
        summary.append(
            {
                "strategy": strategy,
                "n_seeds": len(group),
                "mean_final_hypervolume": sum(row["final_hypervolume"] for row in group) / 10,
                "mean_candidate_seconds": sum(row["candidate_seconds"] for row in group) / 10,
                "mean_evaluation_seconds": sum(row["evaluation_seconds"] for row in group) / 10,
            }
        )
    for filename, records, columns in (
        ("seed_results.csv", rows, _COLUMNS),
        ("summary.csv", summary, tuple(summary[0])),
    ):
        with (root / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(records)
    return summary
