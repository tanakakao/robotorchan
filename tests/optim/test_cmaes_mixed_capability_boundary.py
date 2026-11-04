"""CMA-ES continuous and mixed dispatch capability contracts."""

from unittest.mock import patch

import pytest
import torch

from robotorchan.optim.capabilities import (
    CMAES_OPTIMIZER_CAPABILITIES,
    MIXED_CMAES_OPTIMIZER_CAPABILITIES,
)
from robotorchan.optim.dispatch import optimize_acqf
from robotorchan.optim.domains.variable_space import MixedVariableSpace


class SumAcquisition(torch.nn.Module):
    def forward(self, candidates):
        return candidates.sum(dim=(-1, -2))


def test_cmaes_capabilities_distinguish_continuous_and_mixed_backends():
    standard = CMAES_OPTIMIZER_CAPABILITIES
    mixed = MIXED_CMAES_OPTIMIZER_CAPABILITIES
    assert standard.continuous
    assert not standard.integer
    assert not standard.categorical
    assert not standard.mixed
    assert mixed.integer
    assert mixed.categorical
    assert mixed.mixed


@pytest.mark.parametrize(
    ("integer_dims", "categorical_values"),
    [
        ((1,), None),
        ((), {1: (0.0, 1.0, 2.0)}),
    ],
)
def test_cmaes_dispatch_routes_discrete_space_to_mixed_backend(
    integer_dims,
    categorical_values,
):
    bounds = torch.tensor([[0.0, 0.0], [1.0, 2.0]], dtype=torch.double)
    space = MixedVariableSpace(
        bounds,
        integer_dims=integer_dims,
        categorical_values=categorical_values,
    )
    candidate = torch.tensor([[0.5, 1.0]], dtype=torch.double)
    with patch(
        "robotorchan.optim.dispatch.optimize_acqf_mixed_cmaes",
        return_value=(candidate, torch.tensor(1.5, dtype=torch.double)),
    ) as backend:
        optimize_acqf(
            SumAcquisition(),
            bounds,
            q=1,
            optimizer="cmaes",
            variable_space=space,
        )
    assert backend.call_args.kwargs["variable_space"] is space
