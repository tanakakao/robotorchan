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


def test_hybrid_runs_real_mixed_cmaes_global_stage():
    pytest.importorskip("cmaes")
    bounds = torch.tensor([[0.0, 1.0, 10.0], [1.0, 3.0, 90.0]], dtype=torch.double)
    space = MixedVariableSpace(
        bounds,
        integer_dims=(1,),
        categorical_values={2: (10.0, 30.0, 90.0)},
    )
    with patch(
        "robotorchan.optim.backends.hybrid.optimize_acqf_mixed_botorch",
        side_effect=lambda **kwargs: (
            kwargs["batch_initial_conditions"].squeeze(0),
            SumAcquisition()(kwargs["batch_initial_conditions"]).reshape(()),
        ),
    ) as local:
        candidate, value = optimize_acqf_hybrid(
            SumAcquisition(),
            bounds,
            1,
            global_optimizer="cmaes",
            global_options={"population_size": 8, "max_generations": 2},
            variable_space=space,
            mixed_fixed_features_list=[{1: 2.0, 2: 30.0}],
            seed=13,
        )
    assert candidate.shape == (1, 3)
    assert candidate[0, 1].item() in {1.0, 2.0, 3.0}
    assert candidate[0, 2].item() in {10.0, 30.0, 90.0}
    assert local.call_args.kwargs["batch_initial_conditions"].shape == (1, 1, 3)
    torch.testing.assert_close(value, SumAcquisition()(candidate.unsqueeze(0)).reshape(()))


@pytest.mark.parametrize("global_optimizer", ["de", "cmaes", "mixed_ga", "pso"])
def test_hybrid_mixed_qbatch_handoff_preserves_global_candidates(global_optimizer):
    bounds = torch.tensor([[0.0, 0.0, 10.0], [1.0, 4.0, 90.0]], dtype=torch.double)
    space = MixedVariableSpace(
        bounds,
        integer_dims=(1,),
        categorical_values={2: (10.0, 30.0, 90.0)},
    )
    global_candidate = torch.tensor([[0.2, 1.0, 90.0], [0.8, 3.0, 10.0]], dtype=torch.double)
    backend = {
        "de": "optimize_acqf_de",
        "cmaes": "optimize_acqf_mixed_cmaes",
        "mixed_ga": "optimize_acqf_mixed_ga",
        "pso": "optimize_acqf_pso",
    }[global_optimizer]
    with (
        patch(
            f"robotorchan.optim.backends.hybrid.{backend}",
            return_value=(global_candidate, torch.tensor(0.0)),
        ) as global_stage,
        patch(
            "robotorchan.optim.backends.hybrid.optimize_acqf_mixed_botorch",
            return_value=(global_candidate, torch.tensor(0.0)),
        ) as local_stage,
    ):
        candidates, value = optimize_acqf_hybrid(
            SumAcquisition(),
            bounds,
            q=2,
            global_optimizer=global_optimizer,
            variable_space=space,
            mixed_fixed_features_list=[{1: 1.0, 2: 90.0}],
        )

    global_stage.assert_called_once()
    assert global_stage.call_args.kwargs["variable_space"] is space
    initial = local_stage.call_args.kwargs["batch_initial_conditions"]
    torch.testing.assert_close(initial, global_candidate.unsqueeze(0))
    assert local_stage.call_args.kwargs["q"] == 2
    assert local_stage.call_args.kwargs["fixed_features_list"] == [{1: 1.0, 2: 90.0}]
    torch.testing.assert_close(candidates, global_candidate)
    assert value.ndim == 0
