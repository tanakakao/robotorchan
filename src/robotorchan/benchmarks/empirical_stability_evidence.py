"""Failure-aware, seed-complete empirical robustness summaries."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

_EXPECTED_SEEDS = tuple(range(10))
_STATUSES = frozenset(("success", "failed", "not_run"))


def summarize_seed_stability(
    records: Sequence[Mapping[str, object]], *, metric: str, direction: str
) -> dict[str, object]:
    """Summarize all ten seeds without dropping failures or unexecuted seeds.

    Failed and not-run seeds are excluded from conditional performance statistics,
    but remain explicit in status counts. No missing outcome is imputed.
    """
    if direction not in ("minimize", "maximize") or not metric:
        raise ValueError("Metric and direction must be explicit.")
    by_seed: dict[int, Mapping[str, object]] = {}
    for record in records:
        seed = record.get("seed")
        if type(seed) is not int or seed not in _EXPECTED_SEEDS or seed in by_seed:
            raise ValueError("Unexpected or duplicate seed.")
        status = record.get("status")
        if status not in _STATUSES:
            raise ValueError("Unknown seed execution status.")
        value = record.get(metric)
        if status == "success":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("Successful seed requires a numeric metric.")
            if not math.isfinite(float(value)):
                raise ValueError("Successful seed requires a finite metric.")
        elif value is not None:
            raise ValueError("Failed and not-run seeds must not have a metric.")
        by_seed[seed] = record
    if set(by_seed) != set(_EXPECTED_SEEDS):
        raise ValueError("All ten seeds must be accounted for.")
    counts = Counter(record["status"] for record in by_seed.values())
    values = sorted(
        float(record[metric]) for record in by_seed.values() if record["status"] == "success"
    )
    n = len(values)
    mean = sum(values) / n if n else None
    std = math.sqrt(sum((value - mean) ** 2 for value in values) / (n - 1)) if n > 1 else None

    def quantile(p: float) -> float | None:
        if not values:
            return None
        position = (len(values) - 1) * p
        lower = math.floor(position)
        upper = math.ceil(position)
        return values[lower] + (position - lower) * (values[upper] - values[lower])

    return {
        "expected_seeds": 10,
        "success_count": counts["success"],
        "failed_count": counts["failed"],
        "not_run_count": counts["not_run"],
        "observed_failure_rate": (
            counts["failed"] / (counts["success"] + counts["failed"])
            if counts["success"] + counts["failed"]
            else None
        ),
        "completion_rate": (counts["success"] + counts["failed"]) / 10,
        "metric": metric,
        "direction": direction,
        "successful_mean": mean,
        "successful_sample_std": std,
        "successful_median": quantile(0.5),
        "successful_q25": quantile(0.25),
        "successful_q75": quantile(0.75),
        "conditional_on_success": True,
    }


def write_stability_evidence(
    destination: str | Path,
    *,
    records: Sequence[Mapping[str, object]],
    metric: str,
    direction: str,
    git_sha: str,
) -> dict[str, object]:
    """Persist seed-level provenance and status-aware summary without overwrite."""
    if not git_sha:
        raise ValueError("Source revision is required.")
    summary = summarize_seed_stability(records, metric=metric, direction=direction)
    root = Path(destination)
    if root.exists():
        raise FileExistsError("Refusing to overwrite stability evidence.")
    root.mkdir(parents=True)
    with (root / "seed_status.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("seed", "status", metric, "failure_type"), extrasaction="ignore"
        )
        writer.writeheader()
        for record in sorted(records, key=lambda item: int(item["seed"])):
            writer.writerow(record)
    report = {"git_sha": git_sha, **summary}
    (root / "stability_summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report
