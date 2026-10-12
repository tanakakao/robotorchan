"""Strict seed-aligned numerical rerun verification for empirical evidence."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path


def compare_empirical_reruns(
    original: Sequence[Mapping[str, object]],
    rerun: Sequence[Mapping[str, object]],
    *,
    metric: str,
    atol: float = 1e-8,
    rtol: float = 1e-6,
) -> dict[str, object]:
    """Compare complete ten-seed reruns with explicit execution status agreement.

    Numerical comparisons use abs(a-b) <= atol + rtol*abs(a). Status mismatches
    fail verification, and failed/not-run seeds never count as numerical matches.
    """
    if not metric or not math.isfinite(atol) or not math.isfinite(rtol):
        raise ValueError("Metric and finite tolerances are required.")
    if atol < 0 or rtol < 0:
        raise ValueError("Tolerances must be nonnegative.")

    def index(records: Sequence[Mapping[str, object]]) -> dict[int, Mapping[str, object]]:
        indexed = {}
        for record in records:
            seed = record.get("seed")
            status = record.get("status")
            if type(seed) is not int or seed not in range(10) or seed in indexed:
                raise ValueError("Invalid or duplicate seed.")
            if status not in ("success", "failed", "not_run"):
                raise ValueError("Unknown execution status.")
            value = record.get(metric)
            if status == "success":
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError("Successful seed requires a numeric metric.")
                if not math.isfinite(float(value)):
                    raise ValueError("Successful seed requires a finite metric.")
            elif value is not None:
                raise ValueError("Unsuccessful seed cannot have a metric.")
            indexed[seed] = record
        if set(indexed) != set(range(10)):
            raise ValueError("Both reruns must account for all ten seeds.")
        return indexed

    left = index(original)
    right = index(rerun)
    rows = []
    for seed in range(10):
        a = left[seed]
        b = right[seed]
        status_match = a["status"] == b["status"]
        numerical_match = None
        absolute_error = None
        tolerance = None
        if a["status"] == "success" and b["status"] == "success":
            original_value = float(a[metric])
            rerun_value = float(b[metric])
            absolute_error = abs(original_value - rerun_value)
            tolerance = atol + rtol * abs(original_value)
            numerical_match = absolute_error <= tolerance
        rows.append(
            {
                "seed": seed,
                "original_status": a["status"],
                "rerun_status": b["status"],
                "status_match": status_match,
                "numerical_match": numerical_match,
                "absolute_error": absolute_error,
                "tolerance": tolerance,
            }
        )
    successful_pairs = [row for row in rows if row["numerical_match"] is not None]
    return {
        "metric": metric,
        "atol": atol,
        "rtol": rtol,
        "expected_seeds": 10,
        "successful_pairs": len(successful_pairs),
        "numerical_matches": sum(row["numerical_match"] is True for row in rows),
        "status_mismatches": sum(not row["status_match"] for row in rows),
        "numerical_mismatches": sum(row["numerical_match"] is False for row in rows),
        "all_successful_seeds_reproduced": (
            len(successful_pairs) == 10
            and all(row["numerical_match"] is True for row in rows)
        ),
        "rows": rows,
    }


def write_rerun_verification(
    destination: str | Path,
    *,
    original: Sequence[Mapping[str, object]],
    rerun: Sequence[Mapping[str, object]],
    metric: str,
    original_git_sha: str,
    rerun_git_sha: str,
    atol: float = 1e-8,
    rtol: float = 1e-6,
) -> dict[str, object]:
    """Persist deterministic rerun comparisons and revision provenance."""
    if not original_git_sha or not rerun_git_sha:
        raise ValueError("Both source revisions are required.")
    result = compare_empirical_reruns(
        original, rerun, metric=metric, atol=atol, rtol=rtol
    )
    root = Path(destination)
    if root.exists():
        raise FileExistsError("Refusing to overwrite rerun verification evidence.")
    root.mkdir(parents=True)
    with (root / "rerun_by_seed.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(result["rows"][0]))
        writer.writeheader()
        writer.writerows(result["rows"])
    report = {
        "original_git_sha": original_git_sha,
        "rerun_git_sha": rerun_git_sha,
        **{key: value for key, value in result.items() if key != "rows"},
    }
    (root / "rerun_verification.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report
