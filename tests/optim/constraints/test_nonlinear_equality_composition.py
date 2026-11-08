"""Nonlinear equalities expressed with BoTorch-style inequality pairs."""

import torch

from robotorchan.optim.constraints import CandidateConstraints
from robotorchan.optim.constraints.evaluation import candidate_constraint_violation


def test_intrapoint_nonlinear_equality_tolerance() -> None:
    tolerance = 0.05

    def residual(x: torch.Tensor) -> torch.Tensor:
        return x[0].square() - 0.25

    constraints = CandidateConstraints(
        nonlinear_inequality_constraints=(
            (lambda x: residual(x) + tolerance, True),
            (lambda x: tolerance - residual(x), True),
        )
    )
    candidates = torch.tensor([[[0.5]], [[0.7]]], dtype=torch.double)
    violations = candidate_constraint_violation(candidates, constraints)
    torch.testing.assert_close(violations[0], torch.zeros_like(violations[0]))
    assert violations[1] > 0


def test_interpoint_nonlinear_equality_tolerance() -> None:
    tolerance = 0.01

    def residual(batch: torch.Tensor) -> torch.Tensor:
        return batch[:, 0].sum() - 1.0

    constraints = CandidateConstraints(
        nonlinear_inequality_constraints=(
            (lambda x: residual(x) + tolerance, False),
            (lambda x: tolerance - residual(x), False),
        )
    )
    candidates = torch.tensor(
        [[[0.4], [0.6]], [[0.4], [0.7]]], dtype=torch.double
    )
    violations = candidate_constraint_violation(candidates, constraints)
    torch.testing.assert_close(violations[0], torch.zeros_like(violations[0]))
    assert violations[1] > 0
