"""Phase 9 reference for candidate-space constraint violation arithmetic."""

import torch

from robotorchan.optim import CandidateConstraints
from robotorchan.optim.constraints.evaluation import (
    candidate_constraint_violation,
    candidate_is_feasible,
    feasibility_first_ranks,
)


def test_constraint_violation_matches_explicit_q_batch_reference() -> None:
    """Combine intra/interpoint linear, equality and nonlinear residuals."""
    candidates = torch.tensor(
        [
            [[0.2, 0.3], [0.7, 0.6]],
            [[0.8, 0.7], [0.9, 0.2]],
            [[0.4, 0.5], [0.6, 0.4]],
        ],
        dtype=torch.float64,
    )
    constraints = CandidateConstraints(
        inequality_constraints=(
            (torch.tensor([0, 1]), torch.tensor([1.0, 1.0]), 0.9),
            (torch.tensor([[0, 0], [1, 0]]), torch.tensor([1.0, -1.0]), -0.3),
        ),
        equality_constraints=(
            (torch.tensor([1]), torch.tensor([1.0]), 0.5),
        ),
        nonlinear_inequality_constraints=(
            (lambda x: x[0] * x[1] - 0.1, True),
            (lambda x: x[1, 0] - x[0, 0] - 0.1, False),
        ),
    )
    tolerance = 0.05
    expected = []
    for batch in candidates.tolist():
        first, second = batch
        intra_linear = max(0.0, 0.9 - min(sum(first), sum(second)))
        inter_linear = max(0.0, -0.3 - (first[0] - second[0]))
        equality = max(
            0.0,
            abs(first[1] - 0.5) - tolerance,
            abs(second[1] - 0.5) - tolerance,
        )
        intra_nonlinear = sum(max(0.0, 0.1 - x * y) for x, y in batch)
        inter_nonlinear = max(0.0, 0.1 - (second[0] - first[0]))
        expected.append(
            intra_linear + inter_linear + equality + intra_nonlinear + inter_nonlinear
        )

    actual = candidate_constraint_violation(
        candidates, constraints, equality_tolerance=tolerance
    )
    torch.testing.assert_close(
        actual, torch.tensor(expected, dtype=torch.float64), atol=1e-12, rtol=1e-12
    )
    torch.testing.assert_close(
        candidate_is_feasible(candidates, constraints, equality_tolerance=tolerance),
        actual == 0,
    )


def test_feasibility_first_ordering_ignores_infeasible_objective_advantage() -> None:
    """Feasible points rank by objective; infeasible points by violation."""
    values = torch.tensor([0.5, 100.0, 0.9, 200.0], dtype=torch.float64)
    violation = torch.tensor([0.0, 0.2, 0.0, 0.1], dtype=torch.float64)

    ranks = feasibility_first_ranks(values, violation)

    torch.testing.assert_close(
        ranks, torch.tensor([3.0, 1.0, 4.0, 2.0], dtype=torch.float64)
    )
