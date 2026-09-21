"""Randomized Straddle for level-set estimation."""

from __future__ import annotations

import torch
from botorch.models.model import Model
from botorch.posteriors.ensemble import EnsemblePosterior
from botorch.utils.transforms import average_over_ensemble_models
from torch import Tensor

from robotorchan.acquisition.active_learning.straddle import Straddle


class RandomizedStraddle(Straddle):
    """Straddle with one randomized confidence coefficient per selection round."""

    def __init__(
        self,
        model: Model,
        *,
        target: float | Tensor,
        output_index: int | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__(model=model, target=target, beta=0.0, output_index=output_index)
        self.generator = generator
        self.register_buffer("_random_beta", None, persistent=False)

    def resample(self) -> None:
        """Invalidate the coefficient so the next evaluation starts a new round."""
        self._random_beta = None

    def _beta_for(self, reference: Tensor) -> Tensor:
        if self._random_beta is None:
            sample_device = (
                torch.device(self.generator.device)
                if self.generator is not None
                else reference.device
            )
            uniform = torch.rand(
                (),
                dtype=reference.dtype,
                device=sample_device,
                generator=self.generator,
            )
            uniform = uniform.to(reference.device)
            eps = torch.finfo(reference.dtype).eps
            beta = -2.0 * torch.log1p(-uniform.clamp_max(1.0 - eps))
            self._random_beta = beta.detach().to(
                dtype=reference.dtype,
                device=reference.device,
            )
        return self._random_beta.to(dtype=reference.dtype, device=reference.device)

    @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        """Evaluate one fixed randomized Straddle surface."""
        if X.shape[-2] != 1:
            raise ValueError("RandomizedStraddle supports q=1.")
        if getattr(self.model, "_is_ensemble", False):
            raise ValueError("RandomizedStraddle does not yet support ensemble posteriors.")
        posterior = self.model.posterior(X)
        if isinstance(posterior, EnsemblePosterior):
            raise ValueError("RandomizedStraddle does not yet support ensemble posteriors.")
        mean = self._select_output(posterior.mean, X)
        variance = self._select_output(posterior.variance, X)
        std = variance.clamp_min(0.0).sqrt()
        target = torch.as_tensor(self.target, dtype=mean.dtype, device=mean.device)
        beta = self._beta_for(mean)
        score = beta.sqrt() * std - (mean - target).abs()
        return score.squeeze(-1).squeeze(-1)
