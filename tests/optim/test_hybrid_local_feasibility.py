"""Hybrid checks feasibility after local refinement."""

from unittest.mock import patch

import torch

from robotorchan.optim.backends.hybrid import optimize_acqf_hybrid
from robotorchan.optim.constraints import CandidateConstraints


def test_hybrid_falls_back_to_feasible_global_result() -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    constraints = CandidateConstraints(
        inequality_constraints=((torch.tensor([0]), torch.tensor([1.0], dtype=torch.double), 0.5),)
    )
    with (
        patch(
            "robotorchan.optim.backends.hybrid.optimize_acqf_de",
            return_value=(torch.tensor([[0.75]], dtype=torch.double), torch.tensor(0.0)),
        ),
        patch(
            "robotorchan.optim.backends.hybrid.optimize_acqf_botorch",
            return_value=(torch.tensor([[0.25]], dtype=torch.double), torch.tensor(1.0)),
        ),
    ):
        candidate, value = optimize_acqf_hybrid(None, bounds, q=1, constraints=constraints)
    torch.testing.assert_close(candidate, torch.tensor([[0.75]], dtype=torch.double))
    torch.testing.assert_close(value, torch.tensor(0.0))
