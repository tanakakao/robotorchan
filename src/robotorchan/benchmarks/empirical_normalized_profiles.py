"""Comparable within-stratum normalized optimization performance profiles."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path


def normalized_performance_profiles(
    records: Sequence[Mapping[str, object]],
    *,
    direction: str,
    metric: str,
) -> list[dict[str, object]]:
    """Normalize complete method/seed comparisons only within identical strata.

    The stratum must include problem, budget, q and metric. Normalize successful
    scores to [0, 1] using the successful observed min/max within each stratum
    and seed; zero range gives 1 for all tied successes. Failed or not-run seeds
    receive no score and remain explicit. This is descriptive, not a universal
    quality score, and is sensitive to the set of compared methods.
    """
    if direction not in ("minimize", "maximize") or not metric:
        raise ValueError("Metric and direction are required.")
    indexed = {}
    groups = defaultdict(list)
    for record in records:
        problem = record.get("problem")
        strategy = record.get("strategy")
        seed = record.get("seed")
        budget = record.get("budget")
        q = record.get("q")
        status = record.get("status")
        if (
            not isinstance(problem, str)
            or not problem
            or not isinstance(strategy, str)
            or not strategy
            or type(seed) is not int
            or seed not in range(10)
            or type(budget) is not int
            or budget < 1
            or type(q) is not int
            or q < 1
            or status not in ("success", "failed", "not_run")
        ):
            raise ValueError("Invalid normalized profile record.")
        value = record.get(metric)
        if status == "success":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("Successful score must be numeric.")
            if not math.isfinite(float(value)):
                raise ValueError("Successful score must be finite.")
        elif value is not None:
            raise ValueError("Unsuccessful score must be absent.")
        key = (problem, budget, q, seed, strategy)
        if key in indexed:
            raise ValueError("Duplicate strategy/seed/stratum record.")
        indexed[key] = record
        groups[(problem, budget, q, seed)].append(record)
    if not groups:
        raise ValueError("At least one profile record is required.")
    strata = defaultdict(lambda: defaultdict(set))
    for problem, budget, q, seed, strategy in indexed:
        strata[(problem, budget, q)][strategy].add(seed)
    for strategies in strata.values():
        if len(strategies) < 2 or any(seeds != set(range(10)) for seeds in strategies.values()):
            raise ValueError("Every stratum requires at least two complete ten-seed arms.")
    output = []
    for key in sorted(groups):
        successful = [
            float(record[metric]) for record in groups[key] if record["status"] == "success"
        ]
        low = min(successful) if successful else None
        high = max(successful) if successful else None
        for record in sorted(groups[key], key=lambda item: str(item["strategy"])):
            score = None
            if record["status"] == "success":
                value = float(record[metric])
                if high == low:
                    score = 1.0
                elif direction == "minimize":
                    score = (high - value) / (high - low)
                else:
                    score = (value - low) / (high - low)
            output.append(
                {
                    "problem": key[0],
                    "budget": key[1],
                    "q": key[2],
                    "seed": key[3],
                    "strategy": record["strategy"],
                    "status": record["status"],
                    "normalized_score": score,
                    "comparable_successful_arms": len(successful),
                }
            )
    return output


def write_normalized_profiles(
    destination: str | Path,
    *,
    records: Sequence[Mapping[str, object]],
    direction: str,
    metric: str,
    git_sha: str,
) -> dict[str, object]:
    """Persist normalized profiles and explicitly scoped comparison metadata."""
    if not git_sha:
        raise ValueError("Source revision is required.")
    profiles = normalized_performance_profiles(records, direction=direction, metric=metric)
    root = Path(destination)
    if root.exists():
        raise FileExistsError("Refusing to overwrite performance evidence.")
    root.mkdir(parents=True)
    with (root / "normalized_profiles.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(profiles[0]))
        writer.writeheader()
        writer.writerows(profiles)
    report = {
        "git_sha": git_sha,
        "metric": metric,
        "direction": direction,
        "normalization": "per_problem_budget_q_seed_observed_successful_range",
        "missing_policy": "no_imputation",
        "status_counts": {
            status: sum(row["status"] == status for row in profiles)
            for status in ("success", "failed", "not_run")
        },
        "row_count": len(profiles),
    }
    (root / "profile_manifest.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report
