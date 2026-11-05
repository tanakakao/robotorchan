"""Posterior-uncertainty acquisition functions for regression active learning."""

from __future__ import annotations

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.models.model import Model
from botorch.posteriors.ensemble import EnsemblePosterior
from botorch.utils.transforms import average_over_ensemble_models
from torch import Tensor


class PosteriorVariance(AcquisitionFunction):
    """Select batches by mean marginal posterior variance (without diversity)."""

    def __init__(self, model: Model, *, output_index: int | None = None) -> None:
        super().__init__(model=model)
        self.output_index = output_index

    @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        """Return mean marginal variance across q candidates."""

        if getattr(self.model, "_is_ensemble", False):
            raise ValueError("PosteriorVariance does not yet support ensemble posteriors.")
        posterior = self.model.posterior(X)
        if isinstance(posterior, EnsemblePosterior):
            raise ValueError("PosteriorVariance does not yet support ensemble posteriors.")
        variance = posterior.variance
        if variance.ndim != X.ndim:
            raise ValueError(
                "PosteriorVariance does not support structured-output posteriors; "
                "scalarize the model output first."
            )

        num_outputs = variance.shape[-1]
        if self.output_index is not None:
            if not 0 <= self.output_index < num_outputs:
                raise ValueError(
                    f"output_index must be in [0, {num_outputs}); got {self.output_index}."
                )
            variance = variance[..., self.output_index : self.output_index + 1]
        elif num_outputs != 1:
            raise ValueError("output_index is required for multi-output posteriors.")

        return variance.squeeze(-1).mean(dim=-1)


class PosteriorStd(PosteriorVariance):
    """Select batches by square root of mean marginal posterior variance."""

    def forward(self, X: Tensor) -> Tensor:
        """Return the square root of the batch mean variance."""
        return torch.sqrt(super().forward(X).clamp_min(0.0))
