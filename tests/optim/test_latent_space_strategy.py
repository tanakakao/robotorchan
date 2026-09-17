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


def test_pca_strategy_returns_original_space_feasible_candidate() -> None:
    train_X, train_Y = _training_data()
    model = PCAGP(train_X, train_Y, n_components=2)
    reconstruction = PCAReconstruction(model.reducer)
    bounds = torch.stack(
        [
            torch.zeros(4, dtype=torch.double),
            torch.ones(4, dtype=torch.double),
        ]
    )
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
    reconstruction = RandomProjectionReconstruction(model.reducer)
    bounds = torch.stack(
        [
            torch.zeros(4, dtype=torch.double),
            torch.ones(4, dtype=torch.double),
        ]
    )
    strategy = LatentSpaceStrategy(
        bounds,
        reconstruction,
        num_restarts=2,
        raw_samples=16,
    )

    result = strategy.optimize(PosteriorMean(model), q=1)

    expected = PosteriorMean(model)(result.candidates.unsqueeze(-2))
    assert result.candidates.shape == (1, 4)
    assert torch.allclose(result.acquisition_value, expected)


def test_strategy_validates_reconstruction_dimension() -> None:
    train_X, train_Y = _training_data()
    model = PCAGP(train_X, train_Y, n_components=2)
    reconstruction = PCAReconstruction(model.reducer)
    bad_bounds = torch.stack(
        [
            torch.zeros(3, dtype=torch.double),
            torch.ones(3, dtype=torch.double),
        ]
    )

    try:
        LatentSpaceStrategy(bad_bounds, reconstruction)
    except ValueError as error:
        assert "dimension" in str(error)
    else:
        raise AssertionError("Expected dimension mismatch to raise ValueError.")


def test_strategy_validates_optimizer_arguments() -> None:
    train_X, train_Y = _training_data()
    model = PCAGP(train_X, train_Y, n_components=2)
    reconstruction = PCAReconstruction(model.reducer)
    bounds = torch.stack(
        [
            torch.zeros(4, dtype=torch.double),
            torch.ones(4, dtype=torch.double),
        ]
    )

    for kwargs in ({"num_restarts": 0}, {"raw_samples": 0}):
        try:
            LatentSpaceStrategy(bounds, reconstruction, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("Expected invalid optimizer argument to raise ValueError.")

    strategy = LatentSpaceStrategy(bounds, reconstruction, num_restarts=2, raw_samples=16)
    try:
        strategy.optimize(PosteriorMean(model), q=0)
    except ValueError as error:
        assert "q must be at least 1" in str(error)
    else:
        raise AssertionError("Expected q=0 to raise ValueError.")
