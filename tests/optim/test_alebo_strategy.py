"""Tests for ALEBO embedding geometry and constrained optimization."""

import pytest
import torch
from botorch.acquisition.acquisition import AcquisitionFunction

from robotorchan.optim import ALEBOStrategy


def _bounds(input_dim: int = 6) -> torch.Tensor:
    return torch.stack(
        [
            torch.zeros(input_dim, dtype=torch.double),
            torch.ones(input_dim, dtype=torch.double),
        ]
    )


def test_embedding_is_reproducible_and_has_unit_hypersphere_columns() -> None:
    bounds = _bounds()
    first = ALEBOStrategy(bounds, embedding_dim=2, seed=5)
    second = ALEBOStrategy(bounds, embedding_dim=2, seed=5)

    torch.testing.assert_close(first.embedding, second.embedding)
    assert first.embedding.shape == (2, 6)
    torch.testing.assert_close(
        first.embedding.norm(dim=0),
        torch.ones(6, dtype=torch.double),
    )
    torch.testing.assert_close(
        first.embedding @ first.embedding_pinv,
        torch.eye(2, dtype=torch.double),
    )


def test_embedding_preserves_bounds_dtype_and_device() -> None:
    bounds = _bounds()
    strategy = ALEBOStrategy(bounds, embedding_dim=2, seed=3)

    assert strategy.embedding.dtype == bounds.dtype
    assert strategy.embedding.device == bounds.device


def test_projection_returns_original_box_points() -> None:
    bounds = _bounds()
    strategy = ALEBOStrategy(bounds, embedding_dim=2, seed=3)
    Z = torch.tensor([[0.2, -0.2], [-0.2, 0.2]], dtype=torch.double)

    X = strategy.project(Z)

    assert X.shape == (2, 6)
    assert torch.all(bounds[0] <= X)
    assert torch.all(bounds[1] >= X)


def test_projection_is_invariant_to_public_input_units() -> None:
    unit_bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
    scaled_bounds = torch.tensor([[-10.0, 100.0], [30.0, 500.0]], dtype=torch.double)
    unit = ALEBOStrategy(unit_bounds, embedding_dim=1, seed=13)
    scaled = ALEBOStrategy(scaled_bounds, embedding_dim=1, seed=13)
    Z = torch.tensor([[0.6]], dtype=torch.double)

    unit_X = unit.project(Z)
    scaled_X = scaled.project(Z)
    scaled_normalized = (scaled_X - scaled_bounds[0]) / (scaled_bounds[1] - scaled_bounds[0])

    torch.testing.assert_close(unit.embedding, scaled.embedding)
    torch.testing.assert_close(unit_X, scaled_normalized)


def test_polytope_constraints_match_feasibility() -> None:
    strategy = ALEBOStrategy(_bounds(), embedding_dim=2, seed=3)
    A, b = strategy.linear_constraints
    feasible = torch.zeros(1, 2, dtype=torch.double)
    infeasible = torch.tensor([[10.0, -10.0]], dtype=torch.double)

    assert A.shape == (12, 2)
    assert b.shape == (12,)
    assert torch.all(A @ feasible[0] <= b)
    assert bool(strategy.is_feasible(feasible).item())
    assert not bool(strategy.is_feasible(infeasible).item())


def test_projection_rejects_infeasible_points_instead_of_clipping() -> None:
    strategy = ALEBOStrategy(_bounds(), embedding_dim=2, seed=3)
    Z = torch.tensor([[10.0, -10.0]], dtype=torch.double)

    with pytest.raises(ValueError, match="polytope"):
        strategy.project(Z)


def test_seed_none_uses_global_rng_stream() -> None:
    bounds = _bounds()
    torch.manual_seed(101)
    first = ALEBOStrategy(bounds, embedding_dim=2)
    torch.manual_seed(102)
    second = ALEBOStrategy(bounds, embedding_dim=2)

    assert not torch.equal(first.embedding, second.embedding)


def test_validates_embedding_dimension_and_projection_shape() -> None:
    bounds = _bounds()
    with pytest.raises(ValueError, match="embedding_dim"):
        ALEBOStrategy(bounds, embedding_dim=0)
    with pytest.raises(ValueError, match="embedding_dim"):
        ALEBOStrategy(bounds, embedding_dim=7)

    strategy = ALEBOStrategy(bounds, embedding_dim=2)
    with pytest.raises(ValueError, match="Z last dimension"):
        strategy.project(torch.zeros(1, 3, dtype=torch.double))


class _QuadraticAcquisition(AcquisitionFunction):
    def __init__(self) -> None:
        super().__init__(model=None)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return -X.square().sum(dim=(-1, -2))


def test_optimize_returns_feasible_original_space_candidate() -> None:
    strategy = ALEBOStrategy(_bounds(), embedding_dim=2, seed=0, num_restarts=3, raw_samples=32)

    result = strategy.optimize(_QuadraticAcquisition(), q=1)

    assert result.candidates.shape == (1, 6)
    assert torch.all(result.candidates >= strategy.bounds[0])
    assert torch.all(result.candidates <= strategy.bounds[1])
    embedded = result.metadata["embedded_candidates"]
    assert bool(strategy.is_feasible(embedded).all())
    torch.testing.assert_close(result.candidates, strategy.project(embedded))


def test_optimize_validates_q_and_optimizer_settings() -> None:
    bounds = _bounds()
    with pytest.raises(ValueError, match="num_restarts"):
        ALEBOStrategy(bounds, embedding_dim=2, num_restarts=0)
    with pytest.raises(ValueError, match="raw_samples"):
        ALEBOStrategy(bounds, embedding_dim=2, raw_samples=0)

    strategy = ALEBOStrategy(bounds, embedding_dim=2)
    with pytest.raises(ValueError, match="q"):
        strategy.optimize(_QuadraticAcquisition(), q=0)


def test_project_clamps_only_feasible_numerical_boundary_overshoot() -> None:
    strategy = ALEBOStrategy(
        bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double),
        embedding_dim=2,
        seed=0,
    )
    strategy.embedding = torch.eye(2, dtype=torch.double)
    strategy.embedding_pinv = torch.eye(2, dtype=torch.double)
    Z = torch.tensor([[1.0 + 5e-9, -1.0 - 5e-9]], dtype=torch.double)

    projected = strategy.project(Z)

    torch.testing.assert_close(
        projected,
        torch.tensor([[1.0, 0.0]], dtype=torch.double),
    )


def test_sample_feasible_draws_points_inside_alebo_polytope() -> None:
    bounds = torch.stack([torch.zeros(6, dtype=torch.double), torch.ones(6, dtype=torch.double)])
    strategy = ALEBOStrategy(bounds, embedding_dim=2, seed=7)

    samples = strategy.sample_feasible(32, seed=19)

    assert samples.shape == (32, 2)
    assert bool(strategy.is_feasible(samples).all())
