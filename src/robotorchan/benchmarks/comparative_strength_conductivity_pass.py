"""Phase 15 two-objective regression with binary Pass feasibility."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from robotorchan.benchmarks.comparative_baselines import sobol_candidates
from robotorchan.benchmarks.comparative_matrix import ComparativeExperimentCell
from robotorchan.benchmarks.comparative_strength_pass import (
    BinaryCandidateGenerator,
    CandidateProblemView,
)
from robotorchan.benchmarks.heterogeneous_problems import (
    register_heterogeneous_problems,
    strength_conductivity_pass,
    strength_conductivity_pass_labels,
)
from robotorchan.benchmarks.metrics import cumulative_feasibility_rate, hypervolume_curve
from robotorchan.benchmarks.problem import BenchmarkProblem
from robotorchan.benchmarks.registry import BenchmarkProblemRegistry
from robotorchan.benchmarks.runner import (
    BenchmarkTrajectory,
    random_candidates,
    run_benchmark,
)


@dataclass(frozen=True)
class StrengthConductivityPassResult:
    """Paired seed trajectories with feasible hypervolume and Pass labels."""

    trajectories: tuple[BenchmarkTrajectory, ...]
    pass_labels: tuple[Tensor, ...]
    evaluations: Tensor
    feasible_hypervolume: Tensor
    feasibility_rate: Tensor


def run_strength_conductivity_pass(
    cell: ComparativeExperimentCell,
    *,
    candidate_generator: BinaryCandidateGenerator | None = None,
) -> StrengthConductivityPassResult:
    """Run two regression outcomes with observed binary feasibility labels.

    The candidate callback receives search-domain metadata, observed outcomes,
    and canonical 0/1 labels, never the signed constraint margin.
    """
    config = cell.config
    if config.problem != "strength_conductivity_pass":
        raise ValueError("Runner requires strength_conductivity_pass.")
    if config.strategy not in ("random", "sobol") and candidate_generator is None:
        raise ValueError("A binary-aware candidate generator is required.")
    registry = BenchmarkProblemRegistry()
    register_heterogeneous_problems(registry)

    def propose(
        problem: BenchmarkProblem,
        X: Tensor,
        Y: Tensor,
        q: int,
        generator: torch.Generator,
    ) -> Tensor:
        labels = strength_conductivity_pass_labels(X)
        view = CandidateProblemView(
            bounds=problem.bounds.to(dtype=X.dtype, device=X.device).clone(),
            variable_types=problem.variable_types,
        )
        if candidate_generator is not None:
            return candidate_generator(view, X, Y, labels, q, generator)
        if config.strategy == "random":
            return random_candidates(view, X, Y, q, generator)
        return sobol_candidates(view, X, Y, q, generator, initial_points=config.initial_points)

    trajectories = run_benchmark(config, propose, registry=registry)
    problem = strength_conductivity_pass()
    checkpoints = [0, *range(config.q, config.evaluation_budget + 1, config.q)]
    if checkpoints[-1] != config.evaluation_budget:
        checkpoints.append(config.evaluation_budget)
    indices = torch.tensor(
        [config.initial_points + count - 1 for count in checkpoints],
        dtype=torch.long,
        device=trajectories[0].X.device,
    )
    return StrengthConductivityPassResult(
        trajectories=trajectories,
        pass_labels=tuple(strength_conductivity_pass_labels(run.X) for run in trajectories),
        evaluations=torch.tensor(checkpoints, dtype=torch.long),
        feasible_hypervolume=torch.stack(
            [hypervolume_curve(problem, run).index_select(0, indices) for run in trajectories]
        ),
        feasibility_rate=torch.stack(
            [
                cumulative_feasibility_rate(run.constraints).index_select(0, indices)
                for run in trajectories
            ]
        ),
    )
