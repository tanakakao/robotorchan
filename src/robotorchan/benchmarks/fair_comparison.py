"""Validate paired benchmark experiments and shared initial designs."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from robotorchan.benchmarks.config import BenchmarkExperimentConfig
from robotorchan.benchmarks.runner import BenchmarkTrajectory


@dataclass(frozen=True)
class FairComparisonProtocol:
    """Common experimental settings for a pair of benchmark strategies."""

    first: BenchmarkExperimentConfig
    second: BenchmarkExperimentConfig

    def __post_init__(self) -> None:
        fields = (
            "problem",
            "seeds",
            "initial_points",
            "evaluation_budget",
            "q",
            "dtype",
            "device",
        )
        mismatched = [name for name in fields if getattr(self.first, name) != getattr(self.second, name)]
        if mismatched:
            raise ValueError(f"Unmatched benchmark settings: {', '.join(mismatched)}")
        if self.first.strategy == self.second.strategy:
            raise ValueError("Strategies must have distinct names.")

    def validate_trajectories(
        self,
        first: tuple[BenchmarkTrajectory, ...],
        second: tuple[BenchmarkTrajectory, ...],
    ) -> None:
        """Verify budgets, seed pairing, and identical initial observations."""
        expected = self.first.seeds
        if tuple(item.seed for item in first) != expected:
            raise ValueError("First trajectories must match configured seed order.")
        if tuple(item.seed for item in second) != expected:
            raise ValueError("Second trajectories must match configured seed order.")
        total = self.first.initial_points + self.first.evaluation_budget
        for a, b in zip(first, second, strict=True):
            for item in (a, b):
                if item.initial_points != self.first.initial_points or item.X.shape[0] != total:
                    raise ValueError("Trajectory evaluation budget does not match protocol.")
                if item.Y_observed.shape[0] != total:
                    raise ValueError("Observed history length does not match protocol.")
                if item.Y_truth.shape[0] != total:
                    raise ValueError("Truth history length does not match protocol.")
                if item.constraints.shape[0] != total or item.costs.shape[0] != total:
                    raise ValueError("Constraint or cost history length does not match protocol.")
            n = self.first.initial_points
            for name in ("X", "Y_observed", "Y_truth", "constraints", "costs"):
                left = getattr(a, name)[:n]
                right = getattr(b, name)[:n]
                if left.shape != right.shape or not torch.equal(left, right):
                    raise ValueError(f"Initial {name} must match for seed {a.seed}.")
