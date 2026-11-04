"""Contract tests for mixed Hybrid optimizer dispatch."""

from unittest.mock import patch

import torch

from robotorchan.optim.capabilities import HYBRID_OPTIMIZER_CAPABILITIES
from robotorchan.optim.dispatch import optimize_acqf
from robotorchan.optim.domains.variable_space import MixedVariableSpace


class SumAcquisition(torch.nn.Module):
    def forward(self, candidates):
        return candidates.sum(dim=(-1, -2))


def test_hybrid_capabilities_advertise_existing_mixed_support():
    capabilities = HYBRID_OPTIMIZER_CAPABILITIES
    assert capabilities.continuous
    assert capabilities.integer
    assert capabilities.categorical
    assert capabilities.mixed
    assert capabilities.fixed_features


def test_hybrid_dispatch_forwards_mixed_variable_space_and_fixed_features():
    bounds = torch.tensor([[0.0, 1.0, 10.0], [1.0, 4.0, 90.0]], dtype=torch.double)
    variable_space = MixedVariableSpace(
        bounds,
        integer_dims=(1,),
        categorical_values={2: (10.0, 30.0, 90.0)},
    )
    expected = torch.tensor([[0.5, 2.0, 30.0]], dtype=torch.double)
    with patch(
        "robotorchan.optim.dispatch.optimize_acqf_hybrid",
        return_value=(expected, torch.tensor(32.5, dtype=torch.double)),
    ) as backend:
        candidate, _ = optimize_acqf(
            SumAcquisition(),
            bounds,
            q=1,
            optimizer="hybrid",
            variable_space=variable_space,
            fixed_features={2: 30.0},
            optimizer_options={
                "global_optimizer": "pso",
                "mixed_fixed_features_list": [{1: 2.0, 2: 30.0}],
            },
        )
    assert backend.call_args.kwargs["variable_space"] is variable_space
    assert backend.call_args.kwargs["fixed_features"] == {2: 30.0}
    assert backend.call_args.kwargs["global_optimizer"] == "pso"
    torch.testing.assert_close(candidate, expected)
