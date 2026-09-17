import pytest
import torch
from botorch.acquisition.analytic import PosteriorMean
from torch import Tensor

from robotorchan.models.reduced import PCAGP, RandomProjectionGP
from robotorchan.optim import (
    LatentSpaceStrategy,
    PCAReconstruction,
    RandomProjectionReconstruction,
)


def _training_data(input_dim: int = 4) -> tuple[Tensor, Tensor]:
    torch.manual_seed(7)
    X = torch.rand(16, input_dim, dtype=torch.double)
    Y = -((X[:, :2] - 0.7) ** 2).sum(dim=-1, keepdim=True)
    return X, Y


def _bounds(input_dim: int = 4) -> Tensor:
    return torch.stack(
        [
            torch.zeros(input_dim, dtype=torch.double),
            torch.ones(input_dim, dtype=torch.double),
        ]
    )


def test_pca_strategy_returns_original_space_feasible_candidate() -> None:
    train_X, train_Y = _training_data()
    model = PCAGP(train_X, train_Y, n_components=2)
    assert model.input_reducer is not None
    reconstruction = PCAReconstruction(model.input_reducer)
    bounds = _bounds()
    strategy = LatentSpaceStrategy(
        bounds,
        reconstruction,
        num_restarts=2,
        raw_samples=16,
    )

    result = strategy.optimize(PosteriorMean(model), q=1)

    assert result.candidates.shape == (1, 4)
    assert torch.all(result.candidates >= bounds[0])
    assert torch.all(result.candidates <= bounds[1])
    assert result.acquisition_value is not None
    assert result.metadata["latent_candidates"].shape == (1, 2)
    assert result.metadata["reconstructed_candidates"].shape == (1, 4)
    assert result.metadata["feasibility_projection_distance"].shape == (1,)
    assert result.metadata["latent_round_trip_distance"].shape == (1,)


def test_random_projection_strategy_uses_original_acquisition_contract() -> None:
    train_X, train_Y = _training_data()
    model = RandomProjectionGP(train_X, train_Y, n_components=2, random_state=3)
    assert model.input_reducer is not None
    reconstruction = RandomProjectionReconstruction(model.input_reducer)
    bounds = _bounds()
    strategy = LatentSpaceStrategy(
        bounds,
        reconstruction,
        num_restarts=2,
        raw_samples=16,
    )
    acq_function = PosteriorMean(model)

    result = strategy.optimize(acq_function, q=1)

    expected = acq_function(result.candidates).reshape(())
    assert result.candidates.shape == (1, 4)
    assert result.acquisition_value is not None
    torch.testing.assert_close(result.acquisition_value, expected)


def test_strategy_validates_reconstruction_dimension() -> None:
    train_X, train_Y = _training_data()
    model = PCAGP(train_X, train_Y, n_components=2)
    assert model.input_reducer is not None
    reconstruction = PCAReconstruction(model.input_reducer)

    with pytest.raises(ValueError, match="dimension"):
        LatentSpaceStrategy(_bounds(3), reconstruction)


def test_strategy_validates_optimizer_arguments() -> None:
    train_X, train_Y = _training_data()
    model = PCAGP(train_X, train_Y, n_components=2)
    assert model.input_reducer is not None
    reconstruction = PCAReconstruction(model.input_reducer)
    bounds = _bounds()

    with pytest.raises(ValueError, match="num_restarts"):
        LatentSpaceStrategy(bounds, reconstruction, num_restarts=0)
    with pytest.raises(ValueError, match="raw_samples"):
        LatentSpaceStrategy(bounds, reconstruction, raw_samples=0)

    strategy = LatentSpaceStrategy(bounds, reconstruction, num_restarts=2, raw_samples=16)
    with pytest.raises(ValueError, match="q must be at least 1"):
        strategy.optimize(PosteriorMean(model), q=0)
