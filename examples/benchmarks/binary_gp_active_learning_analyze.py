"""Summarize paired binary GP AL results without statistical superiority claims.

Usage:
    python examples/benchmarks/binary_gp_active_learning_analyze.py artifacts/gp-al-run
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

STRATEGIES = ("random", "sobol", "predictive_entropy")
METRICS = ("accuracy", "brier")


def analyze(root: str | Path) -> list[dict[str, object]]:
    """Return per-seed paired endpoint and normalized learning-curve differences."""
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise ValueError("Only complete experiments are eligible for paired analysis.")
    seeds = manifest["seeds"]
    budget = manifest["budget"]
    if not seeds or budget < 1 or manifest.get("failed_seeds"):
        raise ValueError("Invalid or incomplete experiment manifest.")
    with (root / "curves.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    lookup: dict[tuple[int, str, int], dict[str, float]] = {}
    for row in rows:
        key = (int(row["seed"]), row["strategy"], int(row["labels_added"]))
        if key in lookup:
            raise ValueError(f"Duplicate curve entry: {key}")
        lookup[key] = {metric: float(row[metric]) for metric in METRICS}
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in lookup[key].values()):
            raise ValueError(f"Invalid metric at {key}")
    expected = {
        (seed, strategy, step)
        for seed in seeds
        for strategy in STRATEGIES
        for step in range(budget + 1)
    }
    if set(lookup) != expected or set(manifest.get("completed_seeds", [])) != set(seeds):
        raise ValueError("Curves are not complete paired observations.")
    results: list[dict[str, object]] = []
    for metric in METRICS:
        for strategy in ("sobol", "predictive_entropy"):
            endpoint_diffs = []
            area_diffs = []
            for seed in seeds:
                differences = [
                    lookup[seed, strategy, step][metric] - lookup[seed, "random", step][metric]
                    for step in range(budget + 1)
                ]
                endpoint_diffs.append(differences[-1])
                # Trapezoidal area divided by observation budget; retains step zero.
                area_diffs.append(
                    (sum(differences[1:-1]) + (differences[0] + differences[-1]) / 2) / budget
                )
            for statistic, values in (("endpoint", endpoint_diffs), ("normalized_auc", area_diffs)):
                results.append(
                    {
                        "strategy": strategy,
                        "baseline": "random",
                        "metric": metric,
                        "statistic": statistic,
                        "n": len(values),
                        "mean_difference": statistics.mean(values),
                        "sd_difference": statistics.stdev(values) if len(values) > 1 else "",
                    }
                )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir")
    args = parser.parse_args()
    results = analyze(args.experiment_dir)
    for row in results:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
