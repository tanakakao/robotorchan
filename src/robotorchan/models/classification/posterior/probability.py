"""BoTorch-native posterior for class-probability vectors."""

from __future__ import annotations

from collections.abc import Callable

import torch
from botorch.posteriors import Posterior
from torch import Tensor


class ClassificationProbabilityPosterior(Posterior):
    """Push a latent BoTorch posterior through a class-probability link.

    The latent and probability spaces may have different output dimensions.
    Base samples retain the latent posterior's shape, while samples, mean,
    and variance expose the full class dimension.
    """

    def __init__(
        self,
        latent_posterior: Posterior,
        *,
        sample_transform: Callable[[Tensor], Tensor],
        mean: Tensor,
        variance: Tensor,
        quantile_transform: Callable[[float], tuple[Tensor, Tensor]] | None = None,
    ) -> None:
        if mean.ndim < 2 or mean.shape[-1] < 2 or mean.shape != variance.shape:
            raise ValueError("mean and variance must have matching [..., q, classes] shapes.")
        if mean.shape[:-1] != latent_posterior.mean.shape[:-1]:
            raise ValueError("Probability moments must preserve latent batch and candidate axes.")
        if mean.dtype != latent_posterior.dtype or mean.device != latent_posterior.device:
            raise ValueError("Probability moments must match latent dtype and device.")
        self._latent_posterior = latent_posterior
        self._sample_transform = sample_transform
        self._mean = mean
        self._variance = variance
        self._quantile_transform = quantile_transform

    @property
    def mean(self) -> Tensor:
        """Return the class-probability posterior mean, not a latent mean."""
        return self._mean

    @property
    def variance(self) -> Tensor:
        """Return epistemic class-probability variance, not label variance."""
        return self._variance

    @property
    def device(self) -> torch.device:
        return self._mean.device

    @property
    def dtype(self) -> torch.dtype:
        return self._mean.dtype

    @property
    def base_sample_shape(self) -> torch.Size:
        """Retain the latent base-sample shape for BoTorch MC samplers."""
        return self._latent_posterior.base_sample_shape

    @property
    def batch_range(self) -> tuple[int, int]:
        return self._latent_posterior.batch_range

    def _extended_shape(self, sample_shape: torch.Size = torch.Size()) -> torch.Size:  # noqa: B008
        return sample_shape + self._mean.shape

    def rsample(self, sample_shape: torch.Size | None = None) -> Tensor:
        """Reparameterize latent samples and map them to class probabilities."""
        return self._sample_transform(self._latent_posterior.rsample(sample_shape=sample_shape))

    def rsample_from_base_samples(
        self,
        sample_shape: torch.Size,
        base_samples: Tensor,
    ) -> Tensor:
        """Map correlated latent samples generated from common base samples."""
        latent = self._latent_posterior.rsample_from_base_samples(sample_shape, base_samples)
        return self._sample_transform(latent)

    def credible_interval(
        self,
        confidence_level: float = 0.95,
        *,
        num_samples: int = 4096,
    ) -> tuple[Tensor, Tensor]:
        """Return marginal equal-tailed probability credible limits per class.

        Uses exact latent Gaussian quantiles if provided; otherwise estimates
        the marginal limits with posterior samples.
        """
        if not 0.0 < confidence_level < 1.0:
            raise ValueError("confidence_level must be strictly between zero and one.")
        if self._quantile_transform is not None:
            return self._quantile_transform(confidence_level)
        if num_samples < 2:
            raise ValueError("num_samples must be at least 2.")
        samples = self.rsample(torch.Size([num_samples]))
        tail = (1.0 - confidence_level) / 2.0
        limits = torch.tensor(
            [tail, 1.0 - tail], device=samples.device, dtype=samples.dtype
        )
        lower, upper = torch.quantile(samples, limits, dim=0).unbind(0)
        return lower, upper
