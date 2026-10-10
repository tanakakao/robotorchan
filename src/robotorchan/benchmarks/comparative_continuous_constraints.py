"""Phase 13 continuous-outcome constraint benchmark comparisons."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from robotorchan.benchmarks.comparative_baselines import sobol_candidates
from robotorchan.benchmarks.config import BenchmarkExperimentConfig
from robotorchan.benchmarks.constrained_problems import register_constrained_problems
from robotorchan.benchmarks.metrics import cumulative_feasibility_rate, simple_regret_curve
from robotorchan.benchmarks.registry import BenchmarkProblemRegistry
from robotorchan.benchmarks.runner import (
    BenchmarkTrajectory,
    random_candidates,
    run_benchmark,
)

_PROBLEMS = frozenset(
    {
        "constrained_annulus",
        "constrained_disconnected",
        "constrained_narrow_band",
        "constrained_quadratic",
    }
)


@dataclass(frozen=True)
class ContinuousConstraintResult:
    """Truth-based constrained regret and cumulative feasibility by seed."""

    trajectories: tuple[BenchmarkTrajectory, ...]
    evaluations: Tensor
    regret: Tensor
    feasibility_rate: Tensor


def run_continuous_constraint_baseline(
    config: BenchmarkExperimentConfig,
) -> ContinuousConstraintResult:
    """Evaluate Random/Sobol with feasible regret and feasibility checkpoints.

    Only observations are passed to candidate generators. Constraint truth is
    retained for scoring, never supplied as surrogate training outcomes.
    """
    if config.problem not in _PROBLEMS:
        raise ValueError("Unknown continuous constraint benchmark problem.")
    if config.strategy not in ("random", "sobol"):
        raise ValueError("Continuous constraint baseline requires random or sobol.")
    registry = BenchmarkProblemRegistry()
    register_constrained_problems(registry)
    if config.strategy == "random":
        strategy = random_candidates
    else:

        def strategy(problem, X, Y, q, generator):
            return sobol_candidates(
                problem, X, Y, q, generator, initial_points=config.initial_points
            )

    trajectories = run_benchmark(config, strategy, registry=registry)
    initial = config.initial_points
    budget = config.evaluation_budget
    checkpoints = [0, *range(config.q, budget + 1, config.q)]
    if checkpoints[-1] != budget:
        checkpoints.append(budget)
    indices = torch.tensor(
        [initial + count - 1 for count in checkpoints],
        dtype=torch.long,
        device=trajectories[0].X.device,
    )
    problem = registry.create(config.problem)
    regret = torch.stack(
        [simple_regret_curve(problem, run).index_select(0, indices) for run in trajectories]
    )
    feasibility = torch.stack(
        [
            cumulative_feasibility_rate(run.constraints).index_select(0, indices)
            for run in trajectories
        ]
    )
    return ContinuousConstraintResult(
        trajectories=trajectories,
        evaluations=torch.tensor(checkpoints, dtype=torch.long),
        regret=regret,
        feasibility_rate=feasibility,
    )
