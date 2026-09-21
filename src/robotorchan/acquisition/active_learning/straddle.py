"""Level-set and boundary-learning acquisitions for continuous regression."""

from __future__ import annotations

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.models.model import Model
from botorch.posteriors.ensemble import EnsemblePosterior
from botorch.utils.transforms import average_over_ensemble_models
from torch import Tensor


class Straddle(AcquisitionFunction):
    """Prefer uncertain points whose posterior mean is near a target level."""

    def __init__(
        self,
        model: Model,
        *,
        target: float | Tensor,
        beta: float = 1.96,
        output_index: int | None = None,
    ) -> None:
        if beta < 0:
            raise ValueError("beta must be non-negative.")
        super().__init__(model=model)
        self.target = target
        self.beta = beta
        self.output_index = output_index

    def _select_output(self, value: Tensor, X: Tensor) -> Tensor:
        if value.ndim != X.ndim:
            raise ValueError(
                "Straddle does not support structured-output posteriors; "
                "scalarize the model output first."
            )
        num_outputs = value.shape[-1]
        if self.output_index is not None:
            if not 0 <= self.output_index < num_outputs:
                raise ValueError(
                    f"output_index must be in [0, {num_outputs}); got {self.output_index}."
                )
            return value[..., self.output_index : self.output_index + 1]
        if num_outputs != 1:
            raise ValueError("output_index is required for multi-output posteriors.")
        return value

    @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        """Evaluate the straddle score for q=1 candidates."""
        if X.shape[-2] != 1:
            raise ValueError("Straddle supports q=1.")
        posterior = self.model.posterior(X)
        if getattr(self.model, "_is_ensemble", False) or isinstance(posterior, EnsemblePosterior):
            raise ValueError("Straddle does not yet support ensemble posteriors.")
        mean = self._select_output(posterior.mean, X)
        variance = self._select_output(posterior.variance, X)
        std = variance.clamp_min(0.0).sqrt()
        target = torch.as_tensor(self.target, dtype=mean.dtype, device=mean.device)
        score = self.beta * std - (mean - target).abs()
        return score.squeeze(-1).squeeze(-1)


class BoundaryVariance(Straddle):
    """Weight posterior variance by proximity to a target boundary."""

    @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        """Evaluate variance divided by target-distance uncertainty."""
        if X.shape[-2] != 1:
            raise ValueError("BoundaryVariance supports q=1.")
        posterior = self.model.posterior(X)
        if getattr(self.model, "_is_ensemble", False) or isinstance(posterior, EnsemblePosterior):
            raise ValueError("BoundaryVariance does not yet support ensemble posteriors.")
        mean = self._select_output(posterior.mean, X)
        variance = self._select_output(posterior.variance, X).clamp_min(0.0)
        target = torch.as_tensor(self.target, dtype=mean.dtype, device=mean.device)
        scale = variance.sqrt().clamp_min(torch.finfo(variance.dtype).eps)
        proximity = torch.exp(-0.5 * ((mean - target) / scale).square())
        score = variance * proximity
        return score.squeeze(-1).squeeze(-1)
