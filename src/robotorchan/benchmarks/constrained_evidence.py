"""Explicit eligibility and evidence gates for constrained empirical comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from robotorchan.benchmarks.heterogeneous_problems import (
    strength_conductivity_pass_labels,
)
from robotorchan.benchmarks.regression_binary_problems import strength_pass_labels
from robotorchan.benchmarks.runner import BenchmarkTrajectory

Eligibility = Literal["eligible_baseline", "blocked_classification"]
_LABELS = {
    "strength_pass": strength_pass_labels,
    "strength_conductivity_pass": strength_conductivity_pass_labels,
}
_BLOCKED = {
    "strength_pass": frozenset(("qEI", "qNEI")),
    "strength_conductivity_pass": frozenset(("qEHVI", "qNEHVI")),
}


@dataclass(frozen=True)
class ConstrainedEvidence:
    """Auditable separation of observed labels and evaluation-only margins."""

    problem: str
    strategy: str
    eligibility: Eligibility
    reason: str


def constrained_eligibility(problem: str, strategy: str) -> ConstrainedEvidence:
    """Reject unsupported constrained acquisitions without implicit fallback."""
    if problem not in _LABELS:
        raise ValueError("Unknown constrained empirical problem.")
    if strategy in ("random", "sobol"):
        return ConstrainedEvidence(
            problem, strategy, "eligible_baseline", "Outcome-independent candidate generation."
        )
    if strategy in _BLOCKED[problem]:
        return ConstrainedEvidence(
            problem,
            strategy,
            "blocked_classification",
            "Classifier observed-label history and probability-based acquisition "
            "constraint are not yet demonstrated in the comparative runner.",
        )
    raise ValueError("Strategy is not applicable to this constrained problem.")


def observed_pass_labels(problem: str, X: Tensor) -> Tensor:
    """Produce canonical 0/1 labels, never expose signed margins as training labels."""
    if problem not in _LABELS:
        raise ValueError("Unknown constrained empirical problem.")
    labels = _LABELS[problem](X)
    if labels.shape != (*X.shape[:-1], 1):
        raise ValueError("Classification labels must have shape (..., q, 1).")
    if not torch.isfinite(labels).all() or not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("Classification labels must be finite binary values.")
    return labels


def validate_constrained_trajectory(
    problem: str, strategy: str, run: BenchmarkTrajectory
) -> dict[str, object]:
    """Validate baseline evidence and report feasibility without leaking margins."""
    evidence = constrained_eligibility(problem, strategy)
    if evidence.eligibility != "eligible_baseline":
        raise ValueError("Blocked classification acquisitions cannot be certified.")
    if run.X.ndim != 2 or run.X.shape[0] != 70 or run.initial_points != 20:
        raise ValueError("Constrained empirical trajectory must contain 20 + 50 points.")
    if run.evaluation_count != 50 or not run.completed_batches:
        raise ValueError("Incomplete constrained empirical trajectory.")
    if run.completed_batches[-1] != 50:
        raise ValueError("Incorrect final evaluation count.")
    if not torch.isfinite(run.Y_truth).all() or not torch.isfinite(run.Y_observed).all():
        raise ValueError("Nonfinite objective outcomes.")
    if run.constraints.shape != (70, 1) or not torch.isfinite(run.constraints).all():
        raise ValueError("Missing or invalid evaluation-only constraint margins.")
    labels = observed_pass_labels(problem, run.X)
    feasible = (run.constraints >= 0).all(dim=-1, keepdim=True)
    if not torch.equal(labels.bool(), feasible):
        raise ValueError("Pass labels and evaluation-only feasibility disagree.")
    return {
        "problem": problem,
        "strategy": strategy,
        "seed": run.seed,
        "eligibility": evidence.eligibility,
        "feasible_count": int(feasible.sum().item()),
        "feasible_rate": float(feasible.double().mean().item()),
        "zero_feasible": not bool(feasible.any()),
        "label_counts": {
            "pass": int(labels.sum().item()),
            "fail": int(labels.numel() - labels.sum().item()),
        },
    }
