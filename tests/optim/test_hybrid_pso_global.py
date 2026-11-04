"""Hybrid PSO global-stage dispatch contract tests."""

from unittest.mock import patch

import torch

from robotorchan.optim.backends.hybrid import (
    _resolve_global_optimizer,
    optimize_acqf_hybrid,
)
from robotorchan.optim.backends.pso import optimize_acqf_pso
from robotorchan.optim.domains.variable_space import MixedVariableSpace


class SumAcquisition(torch.nn.Module):
    def forward(self, candidates):
        return candidates.sum(dim=(-1, -2))


def test_hybrid_resolves_existing_pso_backend():
    assert _resolve_global_optimizer("pso") is optimize_acqf_pso


def test_hybrid_forwards_mixed_space_to_pso_global_stage():
    bounds = torch.tensor([[0.0, 1.0, 10.0], [1.0, 4.0, 90.0]], dtype=torch.double)
    space = MixedVariableSpace(
        bounds,
        integer_dims=(1,),
        categorical_values={2: (10.0, 30.0, 90.0)},
    )
    global_candidate = torch.tensor([[0.5, 2.0, 30.0]], dtype=torch.double)
    with (
        patch(
            "robotorchan.optim.backends.hybrid.optimize_acqf_pso",
            return_value=(global_candidate, torch.tensor(32.5, dtype=torch.double)),
        ) as global_backend,
        patch(
            "robotorchan.optim.backends.hybrid.optimize_acqf_mixed_botorch",
            return_value=(global_candidate, torch.tensor(32.5, dtype=torch.double)),
        ) as local_backend,
    ):
        candidate, value = optimize_acqf_hybrid(
            SumAcquisition(),
            bounds,
            1,
            global_optimizer="pso",
            global_options={"swarm_size": 8, "iterations": 2},
            variable_space=space,
            mixed_fixed_features_list=[{1: 2.0, 2: 30.0}],
            seed=17,
        )
    assert global_backend.call_args.kwargs["variable_space"] is space
    assert global_backend.call_args.kwargs["seed"] == 17
    assert global_backend.call_args.kwargs["swarm_size"] == 8
    assert local_backend.call_args.kwargs["fixed_features_list"] == [
        {1: 2.0, 2: 30.0}
    ]
    torch.testing.assert_close(candidate, global_candidate)
    assert value.item() == 32.5
