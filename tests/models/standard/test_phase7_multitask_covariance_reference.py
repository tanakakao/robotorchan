"""Independent Phase 7 task-covariance checks for long-format multi-task GP."""

import torch

from robotorchan.models import MultiTaskGP


def test_multitask_task_covariance_matches_explicit_task_kernel() -> None:
    train_x = torch.tensor(
        [[0.1, 0.0], [0.4, 1.0], [0.7, 0.0], [0.9, 1.0]],
        dtype=torch.double,
    )
    train_y = torch.tensor([[0.2], [0.5], [-0.1], [0.8]], dtype=torch.double)
    model = MultiTaskGP(
        train_X=train_x,
        train_Y=train_y,
        task_feature=-1,
        outcome_transform=None,
    )

    task_kernel = model.covar_module.kernels[1]
    unique_tasks = torch.tensor([[0], [1]])
    repeated_tasks = torch.tensor([[0], [0], [1]])
    with torch.no_grad():
        task_covariance = task_kernel(unique_tasks, unique_tasks).to_dense()
        # IndexKernel expects n x 1 inputs; n-vectors are interpreted as a single point.
        repeated = task_kernel(repeated_tasks, repeated_tasks).to_dense()

    expected = task_covariance[
        torch.tensor([0, 0, 1])[:, None],
        torch.tensor([0, 0, 1])[None, :],
    ]
    torch.testing.assert_close(repeated, expected, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(task_covariance, task_covariance.T, atol=1e-10, rtol=1e-10)
    assert torch.linalg.eigvalsh(task_covariance).min() >= -1e-10
