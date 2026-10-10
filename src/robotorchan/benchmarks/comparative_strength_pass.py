"""Phase 14 binary Pass-label benchmark evaluation and history contract."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from robotorchan.benchmarks.comparative_baselines import sobol_candidates
from robotorchan.benchmarks.comparative_matrix import ComparativeExperimentCell
from robotorchan.benchmarks.metrics import cumulative_feasibility_rate, simple_regret_curve
from robotorchan.benchmarks.problem import BenchmarkProblem
from robotorchan.benchmarks.registry import BenchmarkProblemRegistry
from robotorchan.benchmarks.regression_binary_problems import (
    register_regression_binary_problems,
    strength_pass,
    strength_pass_labels,
)
from robotorchan.benchmarks.runner import (
    BenchmarkTrajectory,
    random_candidates,
    run_benchmark,
)

@dataclass(frozen=True)
class CandidateProblemView:
    """Strategy-visible search domain without objective or constraint truth."""

    bounds: Tensor
    variable_types: tuple[str, ...]

    @property
    def dimension(self) -> int:
        """Number of input features."""
        return self.bounds.shape[1]


BinaryCandidateGenerator = Callable[
    [CandidateProblemView, Tensor, Tensor, Tensor, int, torch.Generator], Tensor
]


@dataclass(frozen=True)
class StrengthPassResult:
    """Per-seed observed regression, binary labels, and truth-based scores."""

    trajectories: tuple[BenchmarkTrajectory, ...]
    pass_labels: tuple[Tensor, ...]
    evaluations: Tensor
    feasible_regret: Tensor
    feasibility_rate: Tensor


def run_strength_pass(
    cell: ComparativeExperimentCell,
    *,
    candidate_generator: BinaryCandidateGenerator | None = None,
) -> StrengthPassResult:
    """Run Strength/Pass with canonical 0/1 labels available to the strategy.

    The signed constraint margin is reserved for evaluation; the candidate
    callback receives observed strength and binary Pass labels only.
    """
    config = cell.config
    if config.problem != "strength_pass":
        raise ValueError("Strength/Pass runner requires strength_pass.")
    if config.strategy not in ("random", "sobol") and candidate_generator is None:
        raise ValueError("A binary-aware candidate generator is required.")
    registry = BenchmarkProblemRegistry()
    register_regression_binary_problems(registry)

    def propose(
        problem: BenchmarkProblem,
        X: Tensor,
        Y: Tensor,
        q: int,
        generator: torch.Generator,
    ) -> Tensor:
        labels = strength_pass_labels(X)
        view = CandidateProblemView(
            bounds=problem.bounds.to(device=X.device, dtype=X.dtype).clone(),
            variable_types=problem.variable_types,
        )
        if candidate_generator is not None:
            return candidate_generator(view, X, Y, labels, q, generator)
        if config.strategy == "random":
            return random_candidates(view, X, Y, q, generator)
        return sobol_candidates(
            view, X, Y, q, generator, initial_points=config.initial_points
        )

    trajectories = run_benchmark(config, propose, registry=registry)
    problem = strength_pass()
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
    return StrengthPassResult(
        trajectories=trajectories,
        pass_labels=tuple(strength_pass_labels(run.X) for run in trajectories),
        evaluations=torch.tensor(checkpoints, dtype=torch.long),
        feasible_regret=torch.stack(
            [simple_regret_curve(problem, run).index_select(0, indices) for run in trajectories]
        ),
        feasibility_rate=torch.stack(
            [
                cumulative_feasibility_rate(run.constraints).index_select(0, indices)
                for run in trajectories
            ]
        ),
    )
