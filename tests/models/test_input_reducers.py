from __future__ import annotations

import pytest
import torch

from robotorchan.models.reduction import (
    PCAInputReducer,
    PLSInputReducer,
    RandomProjectionInputReducer,
)


def test_pca_input_reducer_matches_svd_projection() -> None:
    torch.manual_seed(0)
    X = torch.randn(12, 5, dtype=torch.double)
    reducer = PCAInputReducer(n_components=3)

    transformed = reducer.fit_transform(X)

    centered = X - X.mean(dim=0)
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    expected_components = vh[:3].transpose(-2, -1)
    expected = centered @ expected_components

    assert transformed.shape == torch.Size([12, 3])
    torch.testing.assert_close(transformed.abs(), expected.abs())
    assert reducer.explained_variance_ratio is not None
    assert 0 < reducer.explained_variance_ratio.sum() <= 1


def test_pca_input_reducer_preserves_leading_dimensions() -> None:
    torch.manual_seed(1)
    train_X = torch.randn(10, 6)
    reducer = PCAInputReducer(n_components=2).fit(train_X)

    X = torch.randn(3, 4, 5, 6)
    transformed = reducer.transform(X)

    assert transformed.shape == torch.Size([3, 4, 5, 2])


def test_pca_input_reducer_uses_frozen_training_basis() -> None:
    torch.manual_seed(2)
    train_X = torch.randn(20, 4)
    reducer = PCAInputReducer(n_components=2).fit(train_X)
    components_before = reducer.components.clone()
    mean_before = reducer.mean.clone()

    reducer.transform(torch.randn(8, 4))

    torch.testing.assert_close(reducer.components, components_before)
    torch.testing.assert_close(reducer.mean, mean_before)


def test_random_projection_is_reproducible_and_frozen() -> None:
    X = torch.randn(15, 7, dtype=torch.double)
    first = RandomProjectionInputReducer(n_components=3, random_state=123).fit(X)
    second = RandomProjectionInputReducer(n_components=3, random_state=123).fit(X)

    assert first.projection is not None
    assert second.projection is not None
    torch.testing.assert_close(first.projection, second.projection)

    projection_before = first.projection.clone()
    transformed = first.transform(torch.randn(2, 4, 7, dtype=torch.double))

    assert transformed.shape == torch.Size([2, 4, 3])
    torch.testing.assert_close(first.projection, projection_before)


def test_pls_input_reducer_requires_targets() -> None:
    X = torch.randn(10, 5)
    reducer = PLSInputReducer(n_components=2)

    with pytest.raises(ValueError, match="requires paired target data"):
        reducer.fit(X)


def test_pls_input_reducer_extracts_supervised_latent_space() -> None:
    torch.manual_seed(3)
    X = torch.randn(30, 6, dtype=torch.double)
    Y = torch.stack(
        [
            2.0 * X[:, 0] - X[:, 1],
            -0.5 * X[:, 0] + 1.5 * X[:, 2],
        ],
        dim=-1,
    )
    reducer = PLSInputReducer(n_components=2)

    transformed = reducer.fit_transform(X, Y)

    assert transformed.shape == torch.Size([30, 2])
    assert reducer.rotation is not None
    assert reducer.rotation.shape == torch.Size([6, 2])
    assert torch.isfinite(transformed).all()


def test_pls_input_reducer_preserves_candidate_batch_shape() -> None:
    torch.manual_seed(4)
    train_X = torch.randn(24, 5)
    train_Y = train_X[:, :2].sum(dim=-1, keepdim=True) + 0.1 * torch.randn(24, 1)
    reducer = PLSInputReducer(n_components=2).fit(train_X, train_Y)

    X = torch.randn(2, 3, 7, 5)
    transformed = reducer.transform(X)

    assert transformed.shape == torch.Size([2, 3, 7, 2])


def test_input_reducer_buffers_follow_dtype_conversion() -> None:
    X = torch.randn(16, 5, dtype=torch.float32)
    reducer = PCAInputReducer(n_components=2).fit(X).double()

    assert reducer.mean is not None
    assert reducer.components is not None
    assert reducer.mean.dtype == torch.double
    assert reducer.components.dtype == torch.double
