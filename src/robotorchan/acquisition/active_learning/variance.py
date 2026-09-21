"""Posterior-uncertainty acquisition functions for regression active learning."""

from __future__ import annotations

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.models.model import Model
from torch import Tensor


class PosteriorVariance(AcquisitionFunction):
    """Select q=1 inputs with high posterior variance."""

    def __init__(self, model: Model, *, output_index: int | None = None) -> None:
        super().__init__(model=model)
        self.output_index = output_index

    def forward(self, X: Tensor) -> Tensor:
        """Evaluate posterior variance for q=1 candidates."""
        if X.shape[-2] != 1:
            raise ValueError("PosteriorVariance supports q=1; use qNegIntegratedPosteriorVariance for batch AL.")
        variance = self.model.posterior(X).variance
        if variance.shape[-1] != 1:
            if self.output_index is None:
                raise ValueError("output_index is required for multi-output posteriors.")
            variance = variance[..., self.output_index : self.output_index + 1]
        return variance.squeeze(-1).squeeze(-1)


class PosteriorStd(PosteriorVariance):
    """Select q=1 inputs with high posterior standard deviation."""

    def forward(self, X: Tensor) -> Tensor:
        """Evaluate posterior standard deviation for q=1 candidates."""
        return torch.sqrt(super().forward(X).clamp_min(0.0))
