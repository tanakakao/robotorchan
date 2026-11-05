"""Batch marginal posterior variance and standard deviation semantics."""

import pytest
import torch
from botorch.models import SingleTaskGP

from robotorchan.acquisition import qPosteriorStd, qPosteriorVariance
from robotorchan.acquisition.registry import get_acquisition_registry_entry
from robotorchan.optim.backends import optimize_acqf_sampling


@pytest.mark.parametrize("q", [1, 2, 4])
@pytest.mark.parametrize("acquisition_type", [qPosteriorStd, qPosteriorVariance])
def test_marginal_variance_batch_mean_matches_reference(acquisition_type: type, q: int) -> None:
    train_x = torch.linspace(0, 1, 7, dtype=torch.double).unsqueeze(-1)
    model = SingleTaskGP(train_x, train_x.sin())
    model.eval()
    X = torch.linspace(0.15, 0.85, q, dtype=torch.double).reshape(1, q, 1)
    variance_mean = model.posterior(X).variance.squeeze(-1).mean(dim=-1)
    expected = variance_mean.sqrt() if acquisition_type is qPosteriorStd else variance_mean
    torch.testing.assert_close(acquisition_type(model)(X), expected)
    assert get_acquisition_registry_entry(acquisition_type.__name__).capabilities.max_q is None


def test_posterior_variance_q4_candidate_generation() -> None:
    train_x = torch.linspace(0, 1, 7, dtype=torch.double).unsqueeze(-1)
    model = SingleTaskGP(train_x, train_x.sin())
    model.eval()
    acquisition = qPosteriorVariance(model)
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    candidates, value = optimize_acqf_sampling(
        acquisition, bounds, q=4, num_samples=48, method="sobol", seed=5
    )
    assert candidates.shape == (4, 1)
    torch.testing.assert_close(value.reshape(()), acquisition(candidates.unsqueeze(0)).reshape(()))
