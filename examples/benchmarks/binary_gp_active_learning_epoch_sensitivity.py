"""Compare paired binary GP active-learning runs with different fit epochs.

Both archives must use the same seeds, candidate/reference protocol, and budget.
The output is descriptive: no significance or generalization claims.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

METRICS = ("accuracy", "brier")
STRATEGIES = ("random", "sobol", "predictive_entropy")
MATCH_FIELDS = (
    "seeds",
    "strategies",
    "initial_points",
    "candidate_pool",
    "reference_points",
    "reference_seed",
    "boundary",
    "budget",
    "learning_rate",
    "model",
    "acquisition",
    "dtype",
    "device",
    "protocol",
)


def load(root: Path) -> tuple[dict, dict]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete" or manifest.get("failed_seeds"):
        raise ValueError(f"Incomplete experiment: {root}")
    with (root / "curves.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    values = {}
    for row in rows:
        key = (int(row["seed"]), row["strategy"], int(row["labels_added"]))
        if key in values:
            raise ValueError(f"Duplicate record: {key}")
        values[key] = {metric: float(row[metric]) for metric in METRICS}
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values[key].values()):
            raise ValueError(f"Invalid metric: {key}")
    expected = {
        (seed, strategy, step)
        for seed in manifest["seeds"]
        for strategy in STRATEGIES
        for step in range(manifest["budget"] + 1)
    }
    if set(values) != expected or set(manifest.get("completed_seeds", [])) != set(manifest["seeds"]):
        raise ValueError(f"Unpaired or incomplete curves: {root}")
    return manifest, values


def compare(lower: str | Path, higher: str | Path) -> list[dict[str, object]]:
    """Summarize per-seed changes from lower to higher training epochs."""
    first, a = load(Path(lower))
    second, b = load(Path(higher))
    if any(first.get(field) != second.get(field) for field in MATCH_FIELDS):
        raise ValueError("The experimental protocols differ beyond training epochs.")
    if first["epochs_per_refit"] >= second["epochs_per_refit"]:
        raise ValueError("Expected strictly increasing fit epochs.")
    result = []
    for strategy in STRATEGIES:
        for metric in METRICS:
            changes = [
                b[seed, strategy, first["budget"]][metric]
                - a[seed, strategy, first["budget"]][metric]
                for seed in first["seeds"]
            ]
            result.append(
                {
                    "strategy": strategy,
                    "metric": metric,
                    "epochs_before": first["epochs_per_refit"],
                    "epochs_after": second["epochs_per_refit"],
                    "n": len(changes),
                    "mean_change": statistics.mean(changes),
                    "sd_change": statistics.stdev(changes) if len(changes) > 1 else None,
                }
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("lower_epoch_archive")
    parser.add_argument("higher_epoch_archive")
    args = parser.parse_args()
    for row in compare(args.lower_epoch_archive, args.higher_epoch_archive):
        print(json.dumps(row))


if __name__ == "__main__":
    main()
