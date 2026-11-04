"""CMA-ES mixed-domain capability boundary tests."""

from unittest.mock import patch

import pytest
import torch

from robotorchan.optim.capabilities import CMAES_OPTIMIZER_CAPABILITIES
from robotorchan.optim.dispatch import optimize_acqf
from robotorchan.optim.domains.variable_space import MixedVariableSpace


class SumAcquisition(torch.nn.Module):
    def forward(self, candidates):
        return candidates.sum(dim=(-1, -2))


def test_cmaes_capabilities_do_not_claim_unsupported_discrete_variables():
    capabilities = CMAES_OPTIMIZER_CAPABILITIES
    assert capabilities.continuous
    assert not capabilities.integer
    assert not capabilities.categorical
    assert not capabilities.mixed


@pytest.mark.parametrize(
    ("integer_dims", "categorical_values", "message"),
    [
        ((1,), None, "integer variables"),
        ((), {1: (0.0, 1.0, 2.0)}, "categorical variables"),
    ],
)
def test_cmaes_dispatch_rejects_unsupported_discrete_space_before_backend(
    integer_dims,
    categorical_values,
    message,
):
    bounds = torch.tensor([[0.0, 0.0], [1.0, 2.0]], dtype=torch.double)
    space = MixedVariableSpace(
        bounds,
        integer_dims=integer_dims,
        categorical_values=categorical_values,
    )
    with patch("robotorchan.optim.dispatch.optimize_acqf_cmaes") as backend:
        with pytest.raises(NotImplementedError, match=message):
            optimize_acqf(
                SumAcquisition(),
                bounds,
                q=1,
                optimizer="cmaes",
                variable_space=space,
            )
    backend.assert_not_called()
