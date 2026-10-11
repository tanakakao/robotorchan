"""Validate paired initial designs across empirical optimization strategies."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

from robotorchan.benchmarks.runner import BenchmarkTrajectory


@dataclass(frozen=True)
class InitialPairingResult:
    """Report whether initial experimental conditions match across strategies."""

    seeds: tuple[int, ...]
    design_paired: bool
    truth_paired: bool
    observation_paired: bool

    @property
    def fully_paired(self) -> bool:
        """Whether designs, truths, and observations all match exactly."""
        return self.design_paired and self.truth_paired and self.observation_paired


def _initial(trajectory: BenchmarkTrajectory, field: str) -> Tensor:
    """Return a detached initial-history view for exact comparison."""
    return getattr(trajectory, field)[: trajectory.initial_points]


def validate_initial_pairing(
    trajectories: Mapping[str, tuple[BenchmarkTrajectory, ...]],
    *,
    require_observation_pairing: bool = True,
) -> InitialPairingResult:
    """Compare initial conditions across strategies for identical run seeds.

    Each strategy must provide exactly one trajectory per seed. Tensor equality
    is exact, including dtype and shape, to detect subtle fairness differences.
    Raises ValueError if required pairing conditions are not met.
    """
    if len(trajectories) < 2:
        raise ValueError("At least two strategies are required for paired comparison.")
    by_strategy: dict[str, dict[int, BenchmarkTrajectory]] = {}
    for name, runs in trajectories.items():
        if not runs:
            raise ValueError(f"Strategy {name!r} has no trajectories.")
        by_seed = {run.seed: run for run in runs}
        if len(by_seed) != len(runs):
            raise ValueError(f"Strategy {name!r} has duplicate seeds.")
        by_strategy[name] = by_seed
    names = tuple(by_strategy)
    seeds = tuple(sorted(by_strategy[names[0]]))
    if any(set(by_strategy[name]) != set(seeds) for name in names[1:]):
        raise ValueError("Strategies must have identical seed sets.")
    design_paired = True
    truth_paired = True
    observation_paired = True
    for seed in seeds:
        reference = by_strategy[names[0]][seed]
        for name in names[1:]:
            other = by_strategy[name][seed]
            if other.initial_points != reference.initial_points:
                raise ValueError("Strategies must use identical initial-point counts.")
            design_paired &= torch.equal(_initial(reference, "X"), _initial(other, "X"))
            truth_paired &= torch.equal(
                _initial(reference, "Y_truth"), _initial(other, "Y_truth")
            )
            observation_paired &= torch.equal(
                _initial(reference, "Y_observed"), _initial(other, "Y_observed")
            )
    result = InitialPairingResult(
        seeds=seeds,
        design_paired=design_paired,
        truth_paired=truth_paired,
        observation_paired=observation_paired,
    )
    if not result.design_paired:
        raise ValueError("Initial designs are not paired across strategies.")
    if not result.truth_paired:
        raise ValueError("Initial objective truths are not paired across strategies.")
    if require_observation_pairing and not result.observation_paired:
        raise ValueError("Initial observations are not paired across strategies.")
    return result
