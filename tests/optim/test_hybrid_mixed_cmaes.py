"""Hybrid CMA-ES global-stage routing contracts."""

from unittest.mock import patch

import pytest
import torch

from robotorchan.optim.backends.hybrid import optimize_acqf_hybrid
from robotorchan.optim.domains.variable_space import MixedVariableSpace


class SumAcquisition(torch.nn.Module):
    def forward(self, candidates):
        return candidates.sum(dim=(-1, -2))


@pytest.mark.parametrize(
    ("integer_dims", "categorical_values"),
    [
        ((1,), None),
        ((), {1: (0.0, 1.0, 2.0)}),
    ],
)
def test_hybrid_routes_discrete_cmaes_to_mixed_backend(integer_dims, categorical_values):
    bounds = torch.tensor([[0.0, 0.0], [1.0, 2.0]], dtype=torch.double)
    space = MixedVariableSpace(
        bounds, integer_dims=integer_dims, categorical_values=categorical_values
    )
    global_candidate = torch.tensor([[0.5, 1.0]], dtype=torch.double)
    with (
        patch(
            "robotorchan.optim.backends.hybrid.optimize_acqf_mixed_cmaes",
            return_value=(global_candidate, torch.tensor(1.5)),
        ) as mixed_global,
        patch(
            "robotorchan.optim.backends.hybrid.optimize_acqf_mixed_botorch",
            return_value=(global_candidate, torch.tensor(1.5)),
        ) as local,
    ):
        optimize_acqf_hybrid(
            SumAcquisition(),
            bounds,
            1,
            global_optimizer="cmaes",
            variable_space=space,
            fixed_features={0: 0.5},
            mixed_fixed_features_list=[{1: 1.0}],
        )
    assert mixed_global.call_args.kwargs["variable_space"] is space
    assert mixed_global.call_args.kwargs["fixed_features"] == {0: 0.5}
    assert local.call_args.kwargs["batch_initial_conditions"].shape == (1, 1, 2)


def test_hybrid_continuous_cmaes_remains_unchanged():
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    global_candidate = torch.tensor([[0.5]], dtype=torch.double)
    with (
        patch(
            "robotorchan.optim.backends.hybrid.optimize_acqf_cmaes",
            return_value=(global_candidate, torch.tensor(0.5)),
        ) as continuous_global,
        patch(
            "robotorchan.optim.backends.hybrid.optimize_acqf_botorch",
            return_value=(global_candidate, torch.tensor(0.5)),
        ),
    ):
        optimize_acqf_hybrid(SumAcquisition(), bounds, 1, global_optimizer="cmaes")
    continuous_global.assert_called_once()


def test_hybrid_continuous_cmaes_rejects_unsupported_fixed_features():
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    with pytest.raises(NotImplementedError, match="fixed_features"):
        optimize_acqf_hybrid(
            SumAcquisition(),
            bounds,
            1,
            global_optimizer="cmaes",
            fixed_features={0: 0.5},
        )
