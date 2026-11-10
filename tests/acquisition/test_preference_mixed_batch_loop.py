"""Multiple independent mixed q=2 pairs per preference round."""

import torch

from robotorchan.optim.preference_loop import run_mixed_preference_batch_loop


def test_mixed_preference_batch_loop() -> None:
    torch.manual_seed(0)
    points = torch.tensor(
        [[0.0, 0.0], [0.3, 1.0], [0.7, 0.0], [1.0, 1.0]],
        dtype=torch.double,
    )
    comparisons = torch.tensor([[1, 0], [2, 3], [1, 2]], dtype=torch.long)
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
    observed_shapes = []

    def feedback(pairs: torch.Tensor) -> list[int]:
        observed_shapes.append(tuple(pairs.shape))
        return [int(torch.argmax(pair[:, 0]).item()) for pair in pairs]

    updated_points, updated_comparisons, batches = run_mixed_preference_batch_loop(
        points,
        comparisons,
        cat_dims=[1],
        bounds=bounds,
        fixed_features_list=[{1: 0.0}, {1: 1.0}],
        feedback=feedback,
        pairs_per_step=2,
        steps=2,
        fit_optimizer_kwargs={"options": {"maxiter": 10}},
        num_restarts=2,
        raw_samples=8,
        options={"maxiter": 10},
    )
    assert observed_shapes == [(2, 2, 2), (2, 2, 2)]
    assert updated_points.shape == (12, 2)
    assert updated_comparisons.shape == (7, 2)
    assert len(batches) == 2
    assert torch.equal(updated_comparisons[:3], comparisons)
    for step, batch in enumerate(batches):
        assert torch.isfinite(batch).all()
        assert set(batch[:, :, 1].flatten().tolist()) <= {0.0, 1.0}
        for i in range(2):
            row = updated_comparisons[3 + 2 * step + i]
            offset = 4 + 4 * step + 2 * i
            assert set(row.tolist()) == {offset, offset + 1}
