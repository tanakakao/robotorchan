"""Tests for robust uncertainty scenario generators."""

import torch

from robotorchan.uncertainty import (
    CorrelatedGaussianPerturbation,
    EmpiricalScenarios,
    GaussianPerturbation,
    UniformPerturbation,
)


def test_gaussian_perturbation_preserves_unselected_dims() -> None:
    torch.manual_seed(0)
    X = torch.zeros(3, 4, dtype=torch.double)
    scenarios = GaussianPerturbation(std=0.1, dims=[1, 3]).sample(X, n_w=8)
    assert scenarios.shape == (3, 8, 4)
    assert torch.equal(scenarios[..., [0, 2]], torch.zeros(3, 8, 2, dtype=torch.double))


def test_uniform_perturbation_respects_bounds() -> None:
    X = torch.ones(2, 3)
    scenarios = UniformPerturbation(half_width=0.2, dims=[0]).sample(X, n_w=32)
    delta = scenarios[..., 0] - 1
    assert torch.all(delta.abs() <= 0.2)
    assert torch.equal(scenarios[..., 1:], torch.ones(2, 32, 2))


def test_correlated_gaussian_perturbation_shape() -> None:
    X = torch.zeros(2, 3)
    covariance = torch.tensor([[1.0, 0.5], [0.5, 1.0]])
    scenarios = CorrelatedGaussianPerturbation(covariance, dims=[0, 2]).sample(X, n_w=5)
    assert scenarios.shape == (2, 5, 3)
    assert torch.equal(scenarios[..., 1], torch.zeros(2, 5))


def test_empirical_scenarios_replace_environmental_dims() -> None:
    torch.manual_seed(0)
    X = torch.tensor([[10.0, -1.0, 20.0], [30.0, -1.0, 40.0]])
    environments = torch.tensor([[0.0], [1.0], [2.0]])
    scenarios = EmpiricalScenarios(environments, environmental_dims=[1]).sample(X, n_w=12)
    assert scenarios.shape == (2, 12, 3)
    assert torch.equal(scenarios[..., 0], X[:, None, 0].expand(2, 12))
    assert torch.equal(scenarios[..., 2], X[:, None, 2].expand(2, 12))
    assert set(scenarios[..., 1].unique().tolist()) <= {0.0, 1.0, 2.0}
