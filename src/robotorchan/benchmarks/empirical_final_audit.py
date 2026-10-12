"""Final empirical benchmark audit: do not confuse CI with scientific evidence."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

_GOALS = (
    "optimization_performance",
    "algorithm_comparison",
    "problem_adaptability",
    "efficiency_and_stability",
    "reproducibility",
)


def assess_final_empirical_goals(
    evidence: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Assess five frozen objectives using explicitly verified measured evidence.

    A goal passes only with a nonempty artifact reference and affirmative
    measured, verified and reproduced flags. Unknown goals are rejected.
    """
    unknown = set(evidence) - set(_GOALS)
    if unknown:
        raise ValueError(f"Unknown empirical goals: {sorted(unknown)}")
    details = {}
    for goal in _GOALS:
        item = evidence.get(goal, {})
        if not isinstance(item, Mapping):
            raise ValueError(f"{goal} evidence must be a mapping.")
        artifact = item.get("artifact")
        measured = item.get("measured") is True
        verified = item.get("verified") is True
        reproduced = item.get("reproduced") is True
        has_artifact = isinstance(artifact, str) and bool(artifact.strip())
        complete = measured and verified and reproduced and has_artifact
        details[goal] = {
            "complete": complete,
            "artifact": artifact if has_artifact else None,
            "measured": measured,
            "verified": verified,
            "reproduced": reproduced,
        }
    return {
        "complete": all(item["complete"] for item in details.values()),
        "goals": details,
        "unmet_goals": [goal for goal in _GOALS if not details[goal]["complete"]],
        "scope": "declared_evidence_status_not_independent_artifact_validation",
    }


def write_final_empirical_audit(
    destination: str | Path,
    *,
    evidence: Mapping[str, Mapping[str, object]],
    git_sha: str,
) -> dict[str, object]:
    """Persist a deterministic final audit without overwriting earlier evidence."""
    if not git_sha:
        raise ValueError("A Git revision is required.")
    report = {"git_sha": git_sha, **assess_final_empirical_goals(evidence)}
    path = Path(destination)
    if path.exists():
        raise FileExistsError("Refusing to overwrite final audit.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
