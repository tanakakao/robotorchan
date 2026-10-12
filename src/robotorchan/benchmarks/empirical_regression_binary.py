"""Paired empirical evidence for regression objectives with observed binary labels."""

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
from robotorchan.benchmarks.regression_binary_problems import strength_pass

_STRATEGIES = ("random", "sobol")


def run_regression_binary_evidence(
    destination: str | Path, *, q: int, git_sha: str
) -> dict[str, object]:
    """Persist 10 paired seeds and label-aware feasible incumbent trajectories."""
    if q not in (1, 3) or not git_sha:
        raise ValueError("Frozen heterogeneous evidence requires q=1/3 and git_sha.")
    root = Path(destination)
    if root.exists():
        raise FileExistsError("Refusing to overwrite regression/binary evidence.")
    root.mkdir(parents=True)
    manifest: dict[str, object] = {
        "problem": "strength_pass",
        "strategies": list(_STRATEGIES),
        "seeds": list(range(10)),
        "initial_points": 20,
        "evaluation_budget": 50,
        "q": q,
        "dtype": "float64",
        "device": "cpu",
        "git_sha": git_sha,
        "classification_source": "observed_canonical_0_1_labels",
        "constraint_margins": "evaluation_only_not_training_features",
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
                problem="strength_pass",
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
                if run.X.shape != (70, 2) or run.Y_truth.shape != (70, 1):
                    raise ValueError("Incorrect regression/binary evidence dimensions.")
                if run.X.dtype != torch.float64 or run.evaluation_count != 50:
                    raise ValueError("Incorrect empirical dtype or budget.")
                expected = tuple(range(q, 51, q)) + ((50,) if 50 % q else ())
                if run.completed_batches != expected:
                    raise ValueError("Incomplete heterogeneous evaluation batches.")
                if run.seed in paired_initial:
                    torch.testing.assert_close(run.X[:20], paired_initial[run.seed], rtol=0, atol=0)
                else:
                    paired_initial[run.seed] = run.X[:20]
                validation = validate_constrained_trajectory("strength_pass", strategy, run)
                labels = observed_pass_labels("strength_pass", run.X).squeeze(-1)
                feasible = labels.bool()
                best = None
                for index in range(70):
                    if feasible[index]:
                        value = float(run.Y_truth[index, 0])
                        best = value if best is None else max(best, value)
                    rows.append(
                        {
                            "strategy": strategy,
                            "seed": run.seed,
                            "evaluation": index + 1,
                            "post_initial": max(0, index + 1 - 20),
                            "observed_pass_label": int(labels[index]),
                            "feasible_count": int(feasible[: index + 1].sum()),
                            "best_feasible_strength": best if best is not None else "",
                            "feasible_simple_regret": (
                                max(0.0, float(strength_pass().optimal_value[0]) - best)
                                if best is not None
                                else ""
                            ),
                            "zero_feasible": best is None,
                        }
                    )
                if validation["feasible_count"] != int(feasible.sum()):
                    raise ValueError("Label and feasibility summaries disagree.")
        with (root / "label_feasibility_curves.csv").open(
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
