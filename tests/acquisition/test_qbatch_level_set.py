"""q-batch level-set acquisition semantics and candidate generation."""

import pytest
import torch

from robotorchan.acquisition import qBoundaryVariance, qRandomizedStraddle, qStraddle
from robotorchan.acquisition.registry import get_acquisition_registry_entry
from robotorchan.models import SingleTaskGP
from robotorchan.optim.backends import optimize_acqf_sampling


@pytest.mark.parametrize("acquisition_type", [qStraddle, qBoundaryVariance, qRandomizedStraddle])
@pytest.mark.parametrize("q", [1, 2, 4])
def test_level_set_batch_is_mean_of_pointwise_scores(acquisition_type: type, q: int) -> None:
    train_x = torch.linspace(0, 1, 8, dtype=torch.double).unsqueeze(-1)
    model = SingleTaskGP(train_x, (5 * train_x).sin())
    model.eval()
    acquisition = acquisition_type(model, target=0.0)
    X = torch.linspace(0.1, 0.9, q, dtype=torch.double).reshape(1, q, 1)
    expected = torch.stack([acquisition(X[..., i : i + 1, :]) for i in range(q)], dim=-1).mean(
        dim=-1
    )
    torch.testing.assert_close(acquisition(X), expected)
    assert get_acquisition_registry_entry(acquisition_type.__name__).capabilities.max_q is None


def test_randomized_straddle_resample_changes_shared_coefficient() -> None:
    train_x = torch.linspace(0, 1, 8, dtype=torch.double).unsqueeze(-1)
    model = SingleTaskGP(train_x, train_x.sin())
    model.eval()
    acquisition = qRandomizedStraddle(
        model, target=0.0, generator=torch.Generator().manual_seed(12)
    )
    X = torch.tensor([[[0.2], [0.6], [0.8]]], dtype=torch.double)
    initial = acquisition(X)
    torch.testing.assert_close(acquisition(X), initial)
    acquisition.resample()
    assert not torch.allclose(acquisition(X), initial)


def test_straddle_q4_optimizes_candidates() -> None:
    train_x = torch.linspace(0, 1, 8, dtype=torch.double).unsqueeze(-1)
    model = SingleTaskGP(train_x, (5 * train_x).sin())
    model.eval()
    acquisition = qStraddle(model, target=0.0)
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    candidates, value = optimize_acqf_sampling(
        acquisition, bounds, q=4, num_samples=48, method="sobol", seed=17
    )
    assert candidates.shape == (4, 1)
    torch.testing.assert_close(value.reshape(()), acquisition(candidates.unsqueeze(0)).reshape(()))
