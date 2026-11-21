"""Archive paired multi-seed binary GP active-learning evidence.

Example:
    python examples/benchmarks/binary_gp_active_learning_multiseed.py \\
        --output artifacts/binary-gp-al --git-sha YOUR_COMMIT_SHA

The default full protocol is computationally expensive (10 seeds, 50 updates,
three GP refits per step). Use --seeds 1 --budget 1 --epochs 2 for smoke.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import traceback
from pathlib import Path

from binary_gp_active_learning_smoke import run

STRATEGIES = ("random", "sobol", "predictive_entropy")
METRICS = ("accuracy", "brier")


def write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def execute(
    destination: str | Path,
    *,
    git_sha: str,
    seeds: int = 10,
    budget: int = 50,
    epochs: int = 8,
) -> dict[str, object]:
    """Run all paired seeds, retain failures, and aggregate only complete pairs."""
    if not git_sha or seeds < 1 or budget < 1 or budget > 256 or epochs < 1:
        raise ValueError("Require git_sha, positive seeds/epochs and 1 <= budget <= 256.")
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, object] = {
        "status": "running",
        "git_sha": git_sha,
        "seeds": list(range(seeds)),
        "strategies": list(STRATEGIES),
        "initial_points": 20,
        "candidate_pool": 256,
        "reference_points": 1024,
        "reference_seed": 93827,
        "boundary": "wave",
        "budget": budget,
        "epochs_per_refit": epochs,
        "learning_rate": 0.03,
        "model": "BinarySingleTaskGPClassifier",
        "acquisition": "PredictiveEntropy",
        "dtype": "float64",
        "device": "cpu",
        "protocol": "paired_initial_and_candidate_pool; GP_refit_every_step",
        "note": "No claim of statistical superiority; inspect individual seeds and failures.",
    }
    manifest_path = root / "manifest.json"

    def persist() -> None:
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    persist()
    curves: list[dict[str, object]] = []
    differences: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for seed in range(seeds):
        try:
            rows = run(seed=seed, budget=budget, epochs=epochs)
            lookup = {(str(row["strategy"]), int(row["labels_added"])): row for row in rows}
            expected = {(strategy, step) for strategy in STRATEGIES for step in range(budget + 1)}
            if set(lookup) != expected:
                raise ValueError("Incomplete or duplicate strategy/step records.")
            # Commit each seed only after all three strategies complete successfully.
            curves.extend(rows)
            for step in range(budget + 1):
                for metric in METRICS:
                    baseline = float(lookup["random", step][metric])
                    for strategy in ("sobol", "predictive_entropy"):
                        differences.append(
                            {
                                "seed": seed,
                                "labels_added": step,
                                "strategy": strategy,
                                "metric": metric,
                                "difference_vs_random": float(lookup[strategy, step][metric])
                                - baseline,
                            }
                        )
        except Exception as error:
            failures.append(
                {
                    "seed": seed,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
            )
        manifest["completed_seeds"] = sorted({int(row["seed"]) for row in curves})
        manifest["failed_seeds"] = [int(row["seed"]) for row in failures]
        persist()
    write_csv(
        root / "curves.csv",
        curves,
        ["strategy", "seed", "labels_added", "accuracy", "brier"],
    )
    write_csv(
        root / "paired_differences.csv",
        differences,
        ["seed", "labels_added", "strategy", "metric", "difference_vs_random"],
    )
    summaries: list[dict[str, object]] = []
    for strategy in STRATEGIES:
        for step in range(budget + 1):
            for metric in METRICS:
                values = [
                    float(row[metric])
                    for row in curves
                    if row["strategy"] == strategy and row["labels_added"] == step
                ]
                if values:
                    summaries.append(
                        {
                            "strategy": strategy,
                            "labels_added": step,
                            "metric": metric,
                            "n": len(values),
                            "mean": statistics.mean(values),
                            "std": statistics.stdev(values) if len(values) > 1 else "",
                        }
                    )
    write_csv(
        root / "summary.csv",
        summaries,
        ["strategy", "labels_added", "metric", "n", "mean", "std"],
    )
    (root / "failures.json").write_text(
        json.dumps(failures, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    manifest["status"] = "complete" if not failures else "partial_failure"
    persist()
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--git-sha", required=True)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--budget", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()
    manifest = execute(
        args.output,
        git_sha=args.git_sha,
        seeds=args.seeds,
        budget=args.budget,
        epochs=args.epochs,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
