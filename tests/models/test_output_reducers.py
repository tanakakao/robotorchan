from __future__ import annotations

import pytest
import torch

from robotorchan.models.output_reduction import OutputPCAReducer, OutputPLSReducer


def test_output_pca_matches_svd_projection() -> None:
    torch.manual_seed(10)
    train_Y = torch.randn(20, 6, dtype=torch.double)
    reducer = OutputPCAReducer(n_components=3)

    latent = reducer.fit_transform(train_Y)

    centered = train_Y - train_Y.mean(dim=0)
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    expected = centered @ vh[:3].transpose(-2, -1)

    assert latent.shape == torch.Size([20, 3])
    torch.testing.assert_close(latent.abs(), expected.abs())
    assert reducer.explained_variance_ratio is not None
    assert 0 < reducer.explained_variance_ratio.sum() <= 1


def test_output_pca_inverse_preserves_leading_dimensions() -> None:
    torch.manual_seed(11)
    train_Y = torch.randn(24, 7, dtype=torch.double)
    reducer = OutputPCAReducer(n_components=3).fit(train_Y)

    latent = torch.randn(2, 4, 3, dtype=torch.double)
    restored = reducer.inverse_transform(latent)

    assert restored.shape == torch.Size([2, 4, 7])


def test_output_pca_full_rank_round_trip() -> None:
    torch.manual_seed(12)
    train_Y = torch.randn(18, 5, dtype=torch.double)
    reducer = OutputPCAReducer(n_components=5)

    latent = reducer.fit_transform(train_Y)
    restored = reducer.inverse_transform(latent)

    torch.testing.assert_close(restored, train_Y, atol=1e-10, rtol=1e-10)


def test_output_pca_buffers_follow_dtype_conversion() -> None:
    train_Y = torch.randn(16, 5, dtype=torch.float32)
    reducer = OutputPCAReducer(n_components=2).fit(train_Y).double()

    assert reducer.mean is not None
    assert reducer.components is not None
    assert reducer.mean.dtype == torch.double
    assert reducer.components.dtype == torch.double


def test_output_pls_requires_paired_predictors() -> None:
    train_Y = torch.randn(20, 6)
    reducer = OutputPLSReducer(n_components=2)

    with pytest.raises(ValueError, match="requires paired predictor data"):
        reducer.fit(train_Y)


def test_output_pls_extracts_supervised_latent_space_and_restores_shape() -> None:
    torch.manual_seed(13)
    train_X = torch.randn(32, 4, dtype=torch.double)
    train_Y = torch.stack(
        [
            2.0 * train_X[:, 0] - train_X[:, 1],
            -0.5 * train_X[:, 0] + 1.5 * train_X[:, 2],
            train_X[:, 1] + 0.2 * train_X[:, 3],
            0.3 * train_X[:, 0] - train_X[:, 2],
            train_X[:, 3],
            train_X[:, 0] + train_X[:, 1] - train_X[:, 2],
        ],
        dim=-1,
    )
    reducer = OutputPLSReducer(n_components=3)

    latent = reducer.fit_transform(train_Y, train_X)
    restored = reducer.inverse_transform(latent)

    assert latent.shape == torch.Size([32, 3])
    assert restored.shape == train_Y.shape
    assert reducer.rotation is not None
    assert reducer.decoder is not None
    assert torch.isfinite(latent).all()
    assert torch.isfinite(restored).all()


def test_output_pls_inverse_preserves_arbitrary_leading_dimensions() -> None:
    torch.manual_seed(14)
    train_X = torch.randn(30, 5, dtype=torch.double)
    train_Y = torch.cat(
        [
            train_X[:, :3],
            train_X[:, :3].square(),
        ],
        dim=-1,
    )
    reducer = OutputPLSReducer(n_components=2).fit(train_Y, train_X)

    latent = torch.randn(2, 3, 4, 2, dtype=torch.double)
    restored = reducer.inverse_transform(latent)

    assert restored.shape == torch.Size([2, 3, 4, 6])


def test_output_reducers_defer_posterior_restoration_to_next_phase() -> None:
    train_X = torch.randn(12, 3)
    train_Y = torch.randn(12, 5)
    pca = OutputPCAReducer(n_components=2).fit(train_Y)
    pls = OutputPLSReducer(n_components=2).fit(train_Y, train_X)

    with pytest.raises(NotImplementedError, match="Phase 5"):
        pca.restore_posterior(None)  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError, match="Phase 5"):
        pls.restore_posterior(None)  # type: ignore[arg-type]
