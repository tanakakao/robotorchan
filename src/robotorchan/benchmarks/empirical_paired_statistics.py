"""Paired seed-level bootstrap inference for empirical optimization comparisons."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

_SEEDS = frozenset(range(10))


def paired_bootstrap_comparison(
    baseline: Sequence[Mapping[str, object]],
    challenger: Sequence[Mapping[str, object]],
    *,
    metric: str,
    direction: str,
    bootstrap_samples: int = 10000,
    bootstrap_seed: int = 2026,
) -> dict[str, object]:
    """Estimate paired challenger improvements conditional on successful seed pairs.

    Positive improvement means challenger is better. Resample *paired differences*,
    never independent method scores. A confidence interval is descriptive and does
    not correct for multiple comparisons or imply statistical significance.
    """
    if direction not in ("minimize", "maximize") or not metric:
        raise ValueError("Metric and optimization direction are required.")
    if bootstrap_samples < 2 or bootstrap_seed < 0:
        raise ValueError("Invalid bootstrap configuration.")

    def index(records: Sequence[Mapping[str, object]]) -> dict[int, Mapping[str, object]]:
        result = {}
        for record in records:
            seed = record.get("seed")
            if type(seed) is not int or seed not in _SEEDS or seed in result:
                raise ValueError("Unexpected or duplicate seed.")
            status = record.get("status")
            if status not in ("success", "failed", "not_run"):
                raise ValueError("Unknown execution status.")
            value = record.get(metric)
            if status == "success":
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError("Successful seed requires a numeric metric.")
                if not math.isfinite(float(value)):
                    raise ValueError("Successful seed requires a finite metric.")
            elif value is not None:
                raise ValueError("Unsuccessful seed cannot supply a performance metric.")
            result[seed] = record
        if set(result) != _SEEDS:
            raise ValueError("Both arms must account for all ten seeds.")
        return result

    left = index(baseline)
    right = index(challenger)
    paired = []
    excluded = []
    for seed in sorted(_SEEDS):
        a = left[seed]
        b = right[seed]
        if a["status"] == "success" and b["status"] == "success":
            difference = float(a[metric]) - float(b[metric])
            if direction == "maximize":
                difference = -difference
            paired.append({"seed": seed, "improvement": difference})
        else:
            excluded.append(
                {
                    "seed": seed,
                    "baseline_status": a["status"],
                    "challenger_status": b["status"],
                }
            )
    if not paired:
        return {
            "metric": metric,
            "direction": direction,
            "paired_success_count": 0,
            "excluded_pairs": excluded,
            "mean_improvement": None,
            "ci95_lower": None,
            "ci95_upper": None,
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
            "paired_differences": paired,
            "inference_status": "insufficient_pairs",
        }
    differences = torch.tensor([item["improvement"] for item in paired], dtype=torch.float64)
    generator = torch.Generator().manual_seed(bootstrap_seed)
    indices = torch.randint(
        len(differences), (bootstrap_samples, len(differences)), generator=generator
    )
    means = differences[indices].mean(dim=1)
    bounds = torch.quantile(means, torch.tensor([0.025, 0.975], dtype=torch.float64))
    return {
        "metric": metric,
        "direction": direction,
        "paired_success_count": len(paired),
        "excluded_pairs": excluded,
        "mean_improvement": float(differences.mean()),
        "ci95_lower": float(bounds[0]),
        "ci95_upper": float(bounds[1]),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "paired_differences": paired,
        "inference_status": "descriptive_paired_bootstrap" if len(paired) >= 2 else "one_pair_only",
    }


def write_paired_bootstrap_evidence(
    destination: str | Path,
    *,
    baseline: Sequence[Mapping[str, object]],
    challenger: Sequence[Mapping[str, object]],
    metric: str,
    direction: str,
    git_sha: str,
) -> dict[str, object]:
    """Persist deterministic paired statistics and per-seed inclusion records."""
    if not git_sha:
        raise ValueError("Source revision is required.")
    result = paired_bootstrap_comparison(
        baseline, challenger, metric=metric, direction=direction
    )
    root = Path(destination)
    if root.exists():
        raise FileExistsError("Refusing to overwrite statistical evidence.")
    root.mkdir(parents=True)
    with (root / "paired_differences.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=("seed", "improvement"))
        writer.writeheader()
        writer.writerows(result["paired_differences"])
    report = {"git_sha": git_sha, **result}
    (root / "paired_bootstrap.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report
