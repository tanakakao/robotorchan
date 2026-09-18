from __future__ import annotations

import pytest
import torch
from botorch.posteriors import Posterior
from torch import Tensor

from robotorchan.reduction import (
    LinearOutputPosterior,
    OutputPCAReducer,
    OutputPLSReducer,
)


class IndependentNormalPosterior(Posterior):
    def __init__(self, mean: Tensor, variance: Tensor) -> None:
        self._mean = mean
        self._variance = variance

    @property
    def mean(self) -> Tensor:
        return self._mean

    @property
    def variance(self) -> Tensor:
        return self._variance

    @property
    def device(self) -> torch.device:
        return self._mean.device

    @property
    def dtype(self) -> torch.dtype:
        return self._mean.dtype

    @property
    def base_sample_shape(self) -> torch.Size:
        return self._mean.shape

    @property
    def batch_range(self) -> tuple[int, int]:
        return (0, -2)

    def _extended_shape(
        self,
        sample_shape: torch.Size = torch.Size(),  # noqa: B008
    ) -> torch.Size:
        return sample_shape + self._mean.shape

    def rsample(self, sample_shape: torch.Size | None = None) -> Tensor:
        if sample_shape is None:
            sample_shape = torch.Size()
        base_samples = torch.randn(
            sample_shape + self.base_sample_shape,
            dtype=self.dtype,
            device=self.device,
        )
        return self.rsample_from_base_samples(sample_shape, base_samples)

    def rsample_from_base_samples(
        self,
        sample_shape: torch.Size,
        base_samples: Tensor,
    ) -> Tensor:
        del sample_shape
        return self._mean + self._variance.sqrt() * base_samples


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


def test_output_pca_restores_posterior_mean_variance_and_shape() -> None:
    torch.manual_seed(15)
    train_Y = torch.randn(24, 5, dtype=torch.double)
    reducer = OutputPCAReducer(n_components=2).fit(train_Y)
    latent_mean = torch.randn(3, 2, dtype=torch.double)
    latent_variance = torch.rand(3, 2, dtype=torch.double) + 0.1
    latent_posterior = IndependentNormalPosterior(latent_mean, latent_variance)

    posterior = reducer.restore_posterior(latent_posterior)

    assert isinstance(posterior, LinearOutputPosterior)
    assert posterior._extended_shape() == torch.Size([3, 5])
    assert posterior._extended_shape(torch.Size([7])) == torch.Size([7, 3, 5])
    assert posterior.base_sample_shape == latent_posterior.base_sample_shape
    assert posterior.batch_range == latent_posterior.batch_range
    torch.testing.assert_close(posterior.mean, reducer.inverse_transform(latent_mean))
    assert reducer.components is not None
    decoder = reducer.components.transpose(-2, -1)
    torch.testing.assert_close(posterior.variance, latent_variance @ decoder.square())


def test_output_pca_base_samples_are_transformed_to_original_output_space() -> None:
    torch.manual_seed(16)
    train_Y = torch.randn(20, 6, dtype=torch.double)
    reducer = OutputPCAReducer(n_components=3).fit(train_Y)
    latent_mean = torch.randn(2, 4, 3, dtype=torch.double)
    latent_variance = torch.rand(2, 4, 3, dtype=torch.double) + 0.1
    latent_posterior = IndependentNormalPosterior(latent_mean, latent_variance)
    posterior = reducer.restore_posterior(latent_posterior)
    sample_shape = torch.Size([5])
    base_samples = torch.randn(
        sample_shape + latent_posterior.base_sample_shape,
        dtype=torch.double,
    )

    samples = posterior.rsample_from_base_samples(sample_shape, base_samples)
    latent_samples = latent_posterior.rsample_from_base_samples(sample_shape, base_samples)
    expected = reducer.inverse_transform(latent_samples)

    assert samples.shape == torch.Size([5, 2, 4, 6])
    torch.testing.assert_close(samples, expected)


def test_output_pls_restores_posterior_to_original_output_dimension() -> None:
    torch.manual_seed(17)
    train_X = torch.randn(30, 4, dtype=torch.double)
    train_Y = torch.randn(30, 7, dtype=torch.double)
    train_Y[:, :4] += train_X
    reducer = OutputPLSReducer(n_components=2).fit(train_Y, train_X)
    latent_posterior = IndependentNormalPosterior(
        mean=torch.randn(3, 2, dtype=torch.double),
        variance=torch.rand(3, 2, dtype=torch.double) + 0.1,
    )

    posterior = reducer.restore_posterior(latent_posterior)

    assert posterior.mean.shape == torch.Size([3, 7])
    assert posterior.variance.shape == torch.Size([3, 7])
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()


def test_output_posterior_rejects_incompatible_latent_dimension() -> None:
    train_Y = torch.randn(16, 5, dtype=torch.double)
    reducer = OutputPCAReducer(n_components=2).fit(train_Y)
    latent_posterior = IndependentNormalPosterior(
        mean=torch.randn(4, 3, dtype=torch.double),
        variance=torch.ones(4, 3, dtype=torch.double),
    )

    with pytest.raises(ValueError, match="does not match"):
        reducer.restore_posterior(latent_posterior)
