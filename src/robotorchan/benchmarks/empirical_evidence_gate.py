"""Fail-closed acceptance gate for final empirical optimization evidence."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

_REQUIRED_FAMILIES = frozenset(
    (
        "single_objective",
        "multi_objective",
        "constrained",
        "mixed_variable",
        "high_dimensional",
        "batch_async",
        "heterogeneous",
        "active_learning",
    )
)


def audit_empirical_evidence(
    records: Sequence[Mapping[str, object]],
    *,
    ci_success: bool,
    provenance_verified: bool,
    rerun_verified: bool,
) -> dict[str, object]:
    """Reject incomplete empirical evidence without inferring measurements.

    Each family must have a real executed ten-seed comparison with a named
    baseline and challenger. The gate does not prove superiority, only that
    evidence prerequisites were declared and independently verified.
    """
    if not all(
        isinstance(flag, bool) for flag in (ci_success, provenance_verified, rerun_verified)
    ):
        raise ValueError("Verification flags must be booleans.")
    by_family = {}
    for record in records:
        family = record.get("family")
        if family not in _REQUIRED_FAMILIES or family in by_family:
            raise ValueError("Unknown or duplicate evidence family.")
        by_family[family] = record
    failures = []
    for family in sorted(_REQUIRED_FAMILIES):
        record = by_family.get(family)
        if record is None:
            failures.append(f"{family}:missing")
            continue
        if record.get("execution") != "measured":
            failures.append(f"{family}:not_measured")
        if record.get("completed_seeds") != 10:
            failures.append(f"{family}:incomplete_seeds")
        baseline = record.get("baseline")
        challenger = record.get("challenger")
        if (
            not isinstance(baseline, str)
            or not baseline
            or not isinstance(challenger, str)
            or not challenger
            or baseline == challenger
        ):
            failures.append(f"{family}:invalid_comparison")
        if record.get("artifact_verified") is not True:
            failures.append(f"{family}:artifact_unverified")
    if not ci_success:
        failures.append("ci:unverified")
    if not provenance_verified:
        failures.append("provenance:unverified")
    if not rerun_verified:
        failures.append("rerun:unverified")
    return {
        "accepted": not failures,
        "families_required": sorted(_REQUIRED_FAMILIES),
        "families_supplied": sorted(by_family),
        "ci_success": ci_success,
        "provenance_verified": provenance_verified,
        "rerun_verified": rerun_verified,
        "failures": failures,
        "scope": "evidence_prerequisites_not_algorithm_superiority",
    }


def write_empirical_evidence_audit(
    destination: str | Path,
    *,
    records: Sequence[Mapping[str, object]],
    ci_success: bool,
    provenance_verified: bool,
    rerun_verified: bool,
    git_sha: str,
) -> dict[str, object]:
    """Write a non-overwriting machine-readable final evidence gate report."""
    if not git_sha:
        raise ValueError("Source revision is required.")
    report = audit_empirical_evidence(
        records,
        ci_success=ci_success,
        provenance_verified=provenance_verified,
        rerun_verified=rerun_verified,
    )
    report = {"git_sha": git_sha, **report}
    path = Path(destination)
    if path.exists():
        raise FileExistsError("Refusing to overwrite empirical evidence audit.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report
