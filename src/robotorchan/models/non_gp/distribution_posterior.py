"""Distribution-aware posterior for Gaussian external surrogates."""

from __future__ import annotations

import torch
from botorch.posteriors.posterior import Posterior
from torch import Tensor


class GaussianDistributionPosterior(Posterior):
    """Independent Gaussian predictive distribution with BoTorch shape semantics."""

    def __init__(self, mean: Tensor, variance: Tensor) -> None:
        if mean.shape != variance.shape:
            raise ValueError("mean and variance must have identical shapes.")
        if mean.ndim < 2 or mean.shape[-1] != 1:
            raise ValueError("Expected posterior tensors with shape ... x q x 1.")
        if torch.any(variance < 0):
            raise ValueError("variance must be non-negative.")
        self._mean = mean
        self._variance = variance

    @property
    def device(self) -> torch.device:
        return self._mean.device

    @property
    def dtype(self) -> torch.dtype:
        return self._mean.dtype

    @property
    def mean(self) -> Tensor:
        return self._mean

    @property
    def variance(self) -> Tensor:
        return self._variance

    @property
    def base_sample_shape(self) -> torch.Size:
        return self._mean.shape

    @property
    def batch_range(self) -> tuple[int, int]:
        return (0, -2)

    def rsample(
        self,
        sample_shape: torch.Size | None = None,
    ) -> Tensor:
        sample_shape = torch.Size() if sample_shape is None else sample_shape
        noise = torch.randn(
            sample_shape + self._mean.shape,
            dtype=self.dtype,
            device=self.device,
        )
        return self._mean + self._variance.clamp_min(0).sqrt() * noise

    def rsample_from_base_samples(self, sample_shape: torch.Size, base_samples: Tensor) -> Tensor:
        expected_shape = sample_shape + self.base_sample_shape
        if base_samples.shape != expected_shape:
            raise ValueError(
                f"base_samples must have shape {tuple(expected_shape)}, "
                f"got {tuple(base_samples.shape)}."
            )
        return self._mean + self._variance.clamp_min(0).sqrt() * base_samples
