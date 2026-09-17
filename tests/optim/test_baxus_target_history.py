"""Regression tests for the BAxUS target-space closed-loop contract."""

import pytest
import torch

from robotorchan.optim import BAxUSState, BAxUSStrategy


def _bounds(input_dim: int = 8) -> torch.Tensor:
    return torch.stack(
        [torch.zeros(input_dim, dtype=torch.double), torch.ones(input_dim, dtype=torch.double)]
    )


def test_feedback_moves_trust_region_center_to_best_target_observation() -> None:
    strategy = BAxUSStrategy(
        _bounds(),
        state=BAxUSState(dim=8, eval_budget=40, target_dim=2),
        seed=3,
    )
    target_X = torch.tensor([[0.4, -0.2], [-0.3, 0.6]], dtype=torch.double)
    target_Y = torch.tensor([[-0.5], [0.2]], dtype=torch.double)

    strategy.update_state(target_Y, target_candidates=target_X)

    torch.testing.assert_close(strategy.target_center, target_X[1])
    expected = torch.stack(
        [
            (target_X[1] - strategy.state.length).clamp_min(-1.0),
            (target_X[1] + strategy.state.length).clamp_max(1.0),
        ]
    )
    torch.testing.assert_close(strategy.target_bounds, expected)


def test_q_batch_feedback_is_recorded_as_joint_target_history() -> None:
    strategy = BAxUSStrategy(
        _bounds(),
        state=BAxUSState(dim=8, eval_budget=40, target_dim=2),
        seed=4,
    )
    target_X = torch.tensor(
        [[-0.6, 0.1], [0.2, 0.3], [0.7, -0.4]], dtype=torch.double
    )
    target_Y = torch.tensor([[-1.0], [0.5], [0.1]], dtype=torch.double)

    strategy.update_state(target_Y, target_candidates=target_X)

    torch.testing.assert_close(strategy.target_X, target_X)
    torch.testing.assert_close(strategy.target_Y, target_Y.flatten())
    torch.testing.assert_close(strategy.target_center, target_X[1])
    assert strategy.state.best_value == pytest.approx(0.5)


def test_expansion_preserves_history_and_original_space_projection() -> None:
    state = BAxUSState(
        dim=8,
        eval_budget=40,
        new_bins_on_split=3,
        target_dim=2,
        length=0.1,
        length_min=0.15,
        restart_triggered=True,
    )
    strategy = BAxUSStrategy(_bounds(), state=state, seed=5)
    old_target_X = torch.tensor([[0.25, -0.5], [-0.75, 0.4]], dtype=torch.double)
    old_target_Y = torch.tensor([0.2, -0.1], dtype=torch.double)
    strategy.target_X = old_target_X.clone()
    strategy.target_Y = old_target_Y.clone()
    old_embedding = strategy.embedding.clone()
    old_projection = old_target_X @ old_embedding.transpose(-2, -1)

    assert strategy.expand_subspace()

    new_projection = strategy.target_X @ strategy.embedding.transpose(-2, -1)
    torch.testing.assert_close(new_projection, old_projection)
    torch.testing.assert_close(strategy.target_Y, old_target_Y)
    assert strategy.target_X.shape[0] == old_target_X.shape[0]
    assert strategy.target_X.shape[1] == strategy.target_dim


def test_feedback_rejects_stale_target_dimension_after_expansion() -> None:
    state = BAxUSState(
        dim=8,
        eval_budget=40,
        target_dim=2,
        length=0.1,
        length_min=0.15,
        restart_triggered=True,
    )
    strategy = BAxUSStrategy(_bounds(), state=state, seed=6)
    old_dim = strategy.target_dim
    assert strategy.expand_subspace()
    assert strategy.target_dim > old_dim

    with pytest.raises(ValueError, match="target_candidates"):
        strategy.update_state(
            torch.tensor([0.1], dtype=torch.double),
            target_candidates=torch.zeros(1, old_dim, dtype=torch.double),
        )
