"""Sequential mixed preference optimization preserves pairwise history."""

import torch

from robotorchan.optim.preference_loop import run_mixed_preference_loop


def test_mixed_preference_feedback_loop() -> None:
    torch.manual_seed(0)
    points = torch.tensor(
        [[0.0, 0.0], [0.3, 1.0], [0.7, 0.0], [1.0, 1.0]],
        dtype=torch.double,
    )
    comparisons = torch.tensor([[1, 0], [2, 3], [1, 2]], dtype=torch.long)
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)

    def feedback(pair: torch.Tensor) -> int:
        utility = torch.sin(2 * torch.pi * pair[:, 0]) + pair[:, 1] * 0.1
        return int(torch.argmax(utility).item())

    updated_points, updated_comparisons, pairs = run_mixed_preference_loop(
        points,
        comparisons,
        cat_dims=[1],
        bounds=bounds,
        fixed_features_list=[{1: 0.0}, {1: 1.0}],
        feedback=feedback,
        steps=2,
        fit_optimizer_kwargs={"options": {"maxiter": 10}},
        num_restarts=2,
        raw_samples=8,
        options={"maxiter": 10},
    )
    assert updated_points.shape == (8, 2)
    assert updated_comparisons.shape == (5, 2)
    assert len(pairs) == 2
    assert torch.equal(updated_comparisons[:3], comparisons)
    for index, pair in enumerate(pairs):
        assert pair.shape == (2, 2)
        assert torch.isfinite(pair).all()
        assert set(pair[:, 1].tolist()) <= {0.0, 1.0}
        assert set(updated_comparisons[3 + index].tolist()) == {4 + 2 * index, 5 + 2 * index}
