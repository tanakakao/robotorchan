"""Export per-evaluation empirical curves and per-seed timing as portable CSV."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import torch

from robotorchan.benchmarks.empirical_report import summarize_empirical_directory
from robotorchan.benchmarks.metrics import hypervolume_curve, simple_regret_curve
from robotorchan.benchmarks.multiobjective_problems import branin_currin
from robotorchan.benchmarks.persistence import load_trajectory
from robotorchan.benchmarks.standard_problems import branin, hartmann6

_PROBLEMS = {"branin": branin, "hartmann6": hartmann6, "branin_currin": branin_currin}


def export_empirical_curves(
    directory: str | Path, destination: str | Path
) -> dict[str, object]:
    """Require a complete validated cohort before emitting any performance curve."""
    output = Path(destination)
    if output.exists():
        raise FileExistsError("Refusing to overwrite existing curve artifacts.")
    summary = summarize_empirical_directory(directory)
    root = Path(directory)
    problem = _PROBLEMS[summary["problem"]]()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    strategies = manifest["strategies"]
    rows: list[dict[str, object]] = []
    timing_rows: list[dict[str, object]] = []
    for strategy in strategies:
        for seed in range(10):
            run = load_trajectory(root / f"{strategy}_seed{seed}.json")
            curve = (
                simple_regret_curve(problem, run)
                if problem.n_objectives == 1
                else hypervolume_curve(problem, run)
            )
            if curve.numel() != 70 or not torch.isfinite(curve).all():
                raise ValueError("Invalid truth-based performance curve.")
            for index, value in enumerate(curve.tolist()):
                rows.append(
                    {
                        "problem": summary["problem"],
                        "q": summary["q"],
                        "strategy": strategy,
                        "seed": seed,
                        "evaluations": max(0, index + 1 - run.initial_points),
                        "total_points": index + 1,
                        "metric": summary["metric"],
                        "value": value,
                    }
                )
            timing_rows.append(
                {
                    "problem": summary["problem"],
                    "q": summary["q"],
                    "strategy": strategy,
                    "seed": seed,
                    "candidate_seconds": sum(run.candidate_seconds),
                    "evaluation_seconds": sum(run.evaluation_seconds),
                    "initial_evaluation_seconds": run.initial_evaluation_seconds,
                    "total_seconds": summary["elapsed_seconds_by_seed"][strategy][seed],
                }
            )
    output.mkdir(parents=True)
    try:
        for filename, data, fields in (
            (
                "curves.csv",
                rows,
                (
                    "problem",
                    "q",
                    "strategy",
                    "seed",
                    "evaluations",
                    "total_points",
                    "metric",
                    "value",
                ),
            ),
            (
                "timing.csv",
                timing_rows,
                (
                    "problem",
                    "q",
                    "strategy",
                    "seed",
                    "candidate_seconds",
                    "evaluation_seconds",
                    "initial_evaluation_seconds",
                    "total_seconds",
                ),
            ),
        ):
            with (output / filename).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(data)
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    except Exception:
        for path in output.iterdir():
            path.unlink()
        output.rmdir()
        raise
    return {"status": "exported", "curve_rows": len(rows), "timing_rows": len(timing_rows)}
