"""Independent covariance reference for block-design Kronecker multi-task GP."""

import torch

from robotorchan.models import KroneckerMultiTaskGP


def test_kronecker_covariance_matches_explicit_input_task_indexing() -> None:
    """Check every covariance entry against a scalar product reference."""
    train_x = torch.tensor([[0.1], [0.4], [0.8]], dtype=torch.float64)
    train_y = torch.tensor(
        [[0.2, -0.1], [0.4, 0.3], [-0.2, 0.7]],
        dtype=torch.float64,
    )
    model = KroneckerMultiTaskGP(
        train_X=train_x,
        train_Y=train_y,
        outcome_transform=None,
    )
    kernel = model.covar_module
    with torch.no_grad():
        input_cov = kernel.data_covar_module(train_x).to_dense()
        task_cov = kernel.task_covar_module.covar_matrix.to_dense()
        actual = kernel(train_x).to_dense()

    n, m = train_y.shape
    assert actual.shape == (n * m, n * m)
    assert input_cov.shape == (n, n)
    assert task_cov.shape == (m, m)

    # MultitaskKernel uses input-major (interleaved) flattening:
    # flat_index = input_index * num_tasks + task_index.
    expected = torch.empty_like(actual)
    for i in range(n):
        for a in range(m):
            for j in range(n):
                for b in range(m):
                    expected[i * m + a, j * m + b] = input_cov[i, j] * task_cov[a, b]

    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(actual, actual.T, atol=1e-10, rtol=1e-10)
    assert torch.linalg.eigvalsh(actual).min().item() >= -1e-10

    # Swapping task indices must swap the corresponding rows and columns.
    task_swap = torch.tensor([1, 0], dtype=torch.long)
    indices = (torch.arange(n, dtype=torch.long)[:, None] * m + task_swap[None, :]).reshape(-1)
    swapped = actual.index_select(0, indices).index_select(1, indices)
    swapped_reference = torch.kron(input_cov, task_cov[task_swap][:, task_swap])
    torch.testing.assert_close(swapped, swapped_reference, atol=1e-10, rtol=1e-10)
