"""Joint Gaussian entropy acquisition: covariance and optimizer integration."""

import math

import pytest
import torch
from botorch.models import SingleTaskGP

from robotorchan.acquisition import qJointPosteriorEntropy
from robotorchan.acquisition.registry import get_acquisition_registry_entry
from robotorchan.optim.backends import optimize_acqf_sampling


@pytest.mark.parametrize("q", [1, 2, 4])
def test_joint_entropy_matches_posterior_covariance(q: int) -> None:
    train_x = torch.linspace(0, 1, 7, dtype=torch.double).unsqueeze(-1)
    train_y = (train_x * 4).sin()
    model = SingleTaskGP(train_x, train_y)
    model.eval()
    X = torch.linspace(0.1, 0.9, q, dtype=torch.double).reshape(1, q, 1)
    acquisition = qJointPosteriorEntropy(model, jitter=1e-6)
    covariance = model.posterior(X).mvn.covariance_matrix
    covariance = covariance + 1e-6 * torch.eye(q, dtype=torch.double)
    expected = 0.5 * (
        q * math.log(2 * math.pi * math.e) + torch.linalg.slogdet(covariance).logabsdet
    )
    torch.testing.assert_close(acquisition(X), expected)
    assert acquisition(X).shape == (1,)
    assert get_acquisition_registry_entry("qJointPosteriorEntropy").capabilities.max_q is None


def test_joint_entropy_accounts_for_correlation() -> None:
    train_x = torch.linspace(0, 1, 7, dtype=torch.double).unsqueeze(-1)
    model = SingleTaskGP(train_x, train_x.sin())
    model.eval()
    acquisition = qJointPosteriorEntropy(model, jitter=1e-8)
    distinct = torch.tensor([[[0.2], [0.8]]], dtype=torch.double)
    duplicate = torch.tensor([[[0.2], [0.2]]], dtype=torch.double)
    assert acquisition(distinct).item() > acquisition(duplicate).item()


def test_joint_entropy_optimizes_a_real_q_batch() -> None:
    train_x = torch.linspace(0, 1, 7, dtype=torch.double).unsqueeze(-1)
    model = SingleTaskGP(train_x, train_x.sin())
    model.eval()
    acquisition = qJointPosteriorEntropy(model, jitter=1e-8)
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    candidates, value = optimize_acqf_sampling(
        acquisition, bounds, q=4, num_samples=64, method="sobol", seed=11
    )
    assert candidates.shape == (4, 1)
    torch.testing.assert_close(value.reshape(()), acquisition(candidates.unsqueeze(0)).reshape(()))
    assert torch.isfinite(value).all()
