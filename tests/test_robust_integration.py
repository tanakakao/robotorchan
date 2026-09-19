"""Integration tests for robust scenarios with existing surrogate families."""

import torch

from robotorchan.models import PCAGP, PLSGP, SingleTaskGP
from robotorchan.objectives import Expectation, WorstCase
from robotorchan.uncertainty import GaussianPerturbation


def _training_data() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    train_X = torch.rand(12, 4, dtype=torch.double)
    train_Y = train_X[:, :1].sin()
    return train_X, train_Y


def _scenario_posterior_mean(model: object) -> torch.Tensor:
    X = torch.full((2, 4), 0.5, dtype=torch.double)
    scenarios = GaussianPerturbation(std=0.02, dims=[0, 1]).sample(X, n_w=3)
    flat = scenarios.reshape(-1, scenarios.shape[-1])
    posterior = model.posterior(flat)
    return posterior.mean.squeeze(-1).reshape(2, 3)


def test_single_task_gp_composes_with_scenarios_and_risk() -> None:
    train_X, train_Y = _training_data()
    model = SingleTaskGP(train_X, train_Y)
    means = _scenario_posterior_mean(model)
    assert Expectation()(means).shape == (2,)
    assert WorstCase()(means).shape == (2,)


def test_pca_gp_composes_in_original_input_space() -> None:
    train_X, train_Y = _training_data()
    model = PCAGP(train_X, train_Y, latent_dim=2)
    means = _scenario_posterior_mean(model)
    assert Expectation()(means).shape == (2,)


def test_pls_gp_composes_in_original_input_space() -> None:
    train_X, train_Y = _training_data()
    model = PLSGP(train_X, train_Y, latent_dim=2)
    means = _scenario_posterior_mean(model)
    assert Expectation()(means).shape == (2,)
