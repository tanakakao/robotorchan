"""Independent mathematical properties of PCA input reduction."""

import torch

from robotorchan.reduction import PCAInputReducer


def test_pca_projection_matches_centered_gram_eigenspace() -> None:
    """Verify PCA variance, orthogonality, and retained projection geometry."""
    train_x = torch.tensor(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.0],
            [2.0, 2.0, 0.0],
            [3.0, 1.0, 3.0],
            [4.0, 3.0, 1.0],
            [5.0, 2.0, 4.0],
        ],
        dtype=torch.float64,
    )
    reducer = PCAInputReducer(n_components=2)
    reducer.fit(train_x)
    projected = reducer.transform(train_x)

    centered = train_x - train_x.mean(dim=0)
    covariance = centered.T @ centered / (train_x.shape[0] - 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).flip(0)

    basis = reducer.components
    assert basis is not None
    assert projected.shape == (6, 2)
    torch.testing.assert_close(basis.T @ basis, torch.eye(2, dtype=torch.float64))
    torch.testing.assert_close(projected, centered @ basis)
    torch.testing.assert_close(reducer.explained_variance, eigenvalues[:2])
    torch.testing.assert_close(
        projected.T @ projected / (train_x.shape[0] - 1),
        torch.diag(eigenvalues[:2]),
        atol=1e-10,
        rtol=1e-10,
    )

    # PCA reconstruction is an orthogonal projection onto the retained subspace.
    reconstructed = projected @ basis.T
    residual = centered - reconstructed
    torch.testing.assert_close(residual @ basis, torch.zeros(6, 2, dtype=torch.float64))
    assert torch.linalg.norm(residual) <= torch.linalg.norm(centered) + 1e-10
