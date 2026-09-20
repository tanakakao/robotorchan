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
        scores = torch.special.ndtr(base_samples[..., 0, 0])
        indices = torch.clamp(
            (scores * self._samples.shape[0]).long(),
            max=self._samples.shape[0] - 1,
        )
        stored = self._samples
        batch_shape = stored.shape[1:-2]
        if not batch_shape:
            return stored[indices]

        flat_batch = int(torch.tensor(batch_shape).prod().item())
        stored = stored.reshape(stored.shape[0], flat_batch, *stored.shape[-2:])
        indices = indices.reshape(*indices.shape[:-len(batch_shape)], flat_batch)
        batch_indices = torch.arange(flat_batch, device=self.device)
        selected = stored[indices, batch_indices]
        return selected.reshape(*indices.shape[:-1], *batch_shape, *stored.shape[-2:])
