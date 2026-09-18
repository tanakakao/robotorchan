"""Tests for the ALEBO Phase 1 embedding foundation."""

import pytest
import torch

from robotorchan.optim import ALEBOStrategy


def _bounds(input_dim: int = 6) -> torch.Tensor:
    return torch.stack(
        [
            torch.zeros(input_dim, dtype=torch.double),
            torch.ones(input_dim, dtype=torch.double),
        ]
    )


def test_embedding_is_reproducible_and_has_orthonormal_rows() -> None:
    bounds = _bounds()
    first = ALEBOStrategy(bounds, embedding_dim=2, seed=5)
    second = ALEBOStrategy(bounds, embedding_dim=2, seed=5)

    torch.testing.assert_close(first.embedding, second.embedding)
    assert first.embedding.shape == (2, 6)
    torch.testing.assert_close(
        first.embedding @ first.embedding.transpose(-2, -1),
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


def test_phase1_does_not_approximate_alebo_optimization() -> None:
    strategy = ALEBOStrategy(_bounds(), embedding_dim=2, seed=0)

    with pytest.raises(NotImplementedError, match="Phase 2"):
        strategy.optimize(None)  # type: ignore[arg-type]
