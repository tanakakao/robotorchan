"""Posterior adapter for stochastic DeepGP predictions."""

from __future__ import annotations

import torch
from botorch.posteriors.posterior import Posterior
from torch import Tensor


class DeepGPPosterior(Posterior):
    """Monte Carlo posterior backed by DeepGP predictive samples."""

    def __init__(self, samples: Tensor) -> None:
        if samples.ndim < 3 or samples.shape[-1] != 1:
            raise ValueError("samples must have shape n_samples x ... x q x 1.")
        self._samples = samples

    @property
    def device(self) -> torch.device:
        return self._samples.device

    @property
    def dtype(self) -> torch.dtype:
        return self._samples.dtype

    @property
    def mean(self) -> Tensor:
        return self._samples.mean(dim=0)

    @property
    def variance(self) -> Tensor:
        return self._samples.var(dim=0, correction=0)

    @property
    def base_sample_shape(self) -> torch.Size:
        return self._samples.shape[1:]

    @property
    def batch_range(self) -> tuple[int, int]:
        return (0, max(self._samples.ndim - 2, 0))

    def rsample(self, sample_shape: torch.Size | None = None) -> Tensor:
        """Resample trajectories from the empirical DeepGP posterior."""
        sample_shape = torch.Size() if sample_shape is None else sample_shape
        if not sample_shape:
            indices = torch.randint(self._samples.shape[0], (), device=self.device)
            return self._samples[indices]
        indices = torch.randint(
            self._samples.shape[0],
            sample_shape,
            device=self.device,
        )
        return self._samples[indices]

    def rsample_from_base_samples(
        self,
        sample_shape: torch.Size,
        base_samples: Tensor,
    ) -> Tensor:
        """Map normal base samples deterministically to stored trajectories."""
        del sample_shape
        scores = torch.sigmoid(base_samples[..., 0, 0])
        indices = torch.clamp(
            (scores * self._samples.shape[0]).long(),
            max=self._samples.shape[0] - 1,
        )
        return self._samples[indices]
