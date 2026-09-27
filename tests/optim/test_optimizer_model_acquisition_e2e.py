"""End-to-end model x acquisition x optimizer coverage."""

import torch
from botorch.acquisition.analytic import ExpectedImprovement, UpperConfidenceBound
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood

from robotorchan.optim.backends import (
    optimize_acqf_botorch,
    optimize_acqf_ga,
    optimize_acqf_pso,
    optimize_acqf_sampling,
)


def _fit_model() -> SingleTaskGP:
    train_x = torch.linspace(0.0, 1.0, 8, dtype=torch.double).unsqueeze(-1)
    train_y = -(train_x - 0.72).square() + 1.0
    model = SingleTaskGP(train_x, train_y)
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    model.eval()
    return model


def test_single_task_gp_ei_runs_with_botorch_optimizer() -> None:
    model = _fit_model()
    acq = ExpectedImprovement(model, best_f=model.train_targets.max())
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)

    candidate, value = optimize_acqf_botorch(
        acq,
        bounds,
        q=1,
        num_restarts=3,
        raw_samples=32,
    )

    assert candidate.shape == (1, 1)
    assert value.numel() == 1
    assert 0.0 <= candidate.item() <= 1.0


def test_single_task_gp_ucb_runs_with_sampling_ga_and_pso() -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    optimizers = (
        (
            optimize_acqf_sampling,
            {"num_samples": 64, "method": "sobol", "seed": 3},
        ),
        (
            optimize_acqf_ga,
            {"population_size": 24, "generations": 4, "seed": 3},
        ),
        (
            optimize_acqf_pso,
            {"swarm_size": 24, "iterations": 4, "seed": 3},
        ),
    )

    for optimizer, kwargs in optimizers:
        model = _fit_model()
        acq = UpperConfidenceBound(model, beta=0.1)
        candidate, value = optimizer(acq, bounds, q=1, **kwargs)
        assert candidate.shape == (1, 1)
        assert value.numel() == 1
        assert candidate.dtype == torch.double
        assert 0.0 <= candidate.item() <= 1.0
