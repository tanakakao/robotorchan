"""Randomized Straddle for level-set estimation."""

from __future__ import annotations

import torch
from torch import Tensor

from robotorchan.acquisition.active_learning.straddle import Straddle


class RandomizedStraddle(Straddle):
    """Straddle with a chi-squared random confidence coefficient."""

    def __init__(
        self,
        model,
        *,
        target: float | Tensor,
        output_index: int | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__(model=model, target=target, beta=0.0, output_index=output_index)
        self.generator = generator

    def _sample_beta(self, reference: Tensor) -> Tensor:
        uniform = torch.rand(
            (),
            dtype=reference.dtype,
            device=reference.device,
            generator=self.generator,
        )
        tiny = torch.finfo(reference.dtype).tiny
        return -2.0 * torch.log1p(-uniform.clamp_max(1.0 - tiny))

    def forward(self, X: Tensor) -> Tensor:
        """Evaluate Straddle using beta sampled from chi-square(2)."""
        if X.shape[-2] != 1:
            raise ValueError("RandomizedStraddle supports q=1.")
        posterior = self.model.posterior(X)
        mean = self._select_output(posterior.mean, X)
        variance = self._select_output(posterior.variance, X)
        std = variance.clamp_min(0.0).sqrt()
        target = torch.as_tensor(self.target, dtype=mean.dtype, device=mean.device)
        beta = self._sample_beta(mean)
        score = beta.sqrt() * std - (mean - target).abs()
        return score.squeeze(-1).squeeze(-1)
