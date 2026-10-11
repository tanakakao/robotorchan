"""Certify and summarize persisted single-objective empirical comparison evidence."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import torch

from robotorchan.benchmarks.empirical_noiseless import PROBLEMS
from robotorchan.benchmarks.metrics import simple_regret_curve
from robotorchan.benchmarks.persistence import load_trajectory
from robotorchan.benchmarks.standard_problems import branin, hartmann6

_PROBLEM_FACTORIES = {"branin": branin, "hartmann6": hartmann6}
_COLUMNS = ("strategy", "seed", "final_regret", "candidate_seconds", "evaluation_seconds")


def summarize_singleobjective_evidence(directory: str | Path) -> list[dict[str, object]]:
    """Require all ten paired seeds and write seed-level and aggregate CSV evidence."""
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "runtime_validated":
        raise ValueError("Cannot report incomplete empirical evidence.")
    if manifest.get("problem") not in PROBLEMS or manifest.get("q") not in (1, 3):
        raise ValueError("Unknown empirical stratum.")
    if (
        manifest.get("seeds") != list(range(10))
        or manifest.get("initial_points") != 20
        or manifest.get("evaluation_budget") != 50
        or manifest.get("dtype") != "float64"
        or manifest.get("device") != "cpu"
        or manifest.get("dirty") is not False
    ):
        raise ValueError("Evidence does not match the frozen protocol.")
    strategies = manifest.get("strategies")
    if (
        not isinstance(strategies, list)
        or len(strategies) != len(set(strategies))
        or not {"random", "sobol"}.issubset(strategies)
    ):
        raise ValueError("Missing mandatory paired baselines.")
    expected = {f"{strategy}_seed{seed}.json" for strategy in strategies for seed in range(10)}
    actual = {p.name for p in root.glob("*_seed*.json")}
    if actual != expected:
        raise ValueError("Missing or unexpected empirical trajectories.")
    problem = _PROBLEM_FACTORIES[manifest["problem"]]()
    rows = []
    initial_by_seed = {}
    for strategy in strategies:
        for seed in range(10):
            trajectory = load_trajectory(root / f"{strategy}_seed{seed}.json")
            if trajectory.seed != seed or trajectory.initial_points != 20:
                raise ValueError("Incorrect seed or initial budget.")
            if trajectory.evaluation_count != 50 or trajectory.Y_truth.dtype != torch.float64:
                raise ValueError("Incorrect evaluation count or dtype.")
            if not trajectory.completed_batches or trajectory.completed_batches[-1] != 50:
                raise ValueError("Incomplete batch sequence.")
            initial = trajectory.X[:20]
            if seed in initial_by_seed:
                torch.testing.assert_close(initial, initial_by_seed[seed], rtol=0, atol=0)
            else:
                initial_by_seed[seed] = initial
            regret = float(simple_regret_curve(problem, trajectory)[-1])
            if not math.isfinite(regret):
                raise ValueError("Nonfinite final regret.")
            rows.append(
                {
                    "strategy": strategy,
                    "seed": seed,
                    "final_regret": regret,
                    "candidate_seconds": sum(trajectory.candidate_seconds),
                    "evaluation_seconds": sum(trajectory.evaluation_seconds),
                }
            )
    summaries = []
    for strategy in strategies:
        group = [row for row in rows if row["strategy"] == strategy]
        summaries.append(
            {
                "strategy": strategy,
                "n_seeds": len(group),
                "mean_final_regret": sum(row["final_regret"] for row in group) / 10,
                "mean_candidate_seconds": sum(row["candidate_seconds"] for row in group) / 10,
                "mean_evaluation_seconds": sum(row["evaluation_seconds"] for row in group) / 10,
            }
        )
    for filename, records, columns in (
        ("seed_results.csv", rows, _COLUMNS),
        ("summary.csv", summaries, tuple(summaries[0])),
    ):
        with (root / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(records)
    return summaries
