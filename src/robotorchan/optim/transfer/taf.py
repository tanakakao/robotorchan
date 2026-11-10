"""Transfer acquisition combining source-model improvement and target EI."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from botorch.acquisition.analytic import AnalyticAcquisitionFunction, ExpectedImprovement
from botorch.models.model import Model
from torch import Tensor


class TransferAcquisitionFunction(AnalyticAcquisitionFunction):
    """Weighted source improvement and target expected improvement.

    For q=1, each source contributes the nonnegative improvement of its
    posterior mean over the target's incumbent. The target contributes
    analytic expected improvement. Source and target weights sum to one.
    This is a mean-based TAF variant, not a full predictive-mixture EI.
    """

    def __init__(
        self,
        source_models: Sequence[Model],
        target_model: Model,
        weights: Tensor,
        best_f: float | Tensor,
        *,
        maximize: bool = True,
    ) -> None:
        if target_model.num_outputs != 1 or any(model.num_outputs != 1 for model in source_models):
            raise ValueError("TAF supports scalar-output models only")
        if weights.ndim != 1 or weights.numel() != len(source_models) + 1:
            raise ValueError("weights must include each source and the target")
        if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
            raise ValueError("weights must be finite and nonnegative")
        if not bool(torch.isclose(weights.sum(), weights.new_tensor(1.0))):
            raise ValueError("weights must sum to one")
        super().__init__(model=target_model)
        self.source_models = torch.nn.ModuleList(source_models)
        self.register_buffer("weights", weights.detach().clone())
        self.register_buffer("best_f", torch.as_tensor(best_f).detach().clone())
        self.maximize = maximize
        self.target_ei = ExpectedImprovement(target_model, best_f=best_f, maximize=maximize)

    def forward(self, X: Tensor) -> Tensor:
        """Evaluate TAF for t-batch x 1 x d inputs."""
        if X.ndim < 2 or X.shape[-2] != 1:
            raise ValueError("TAF supports q=1 only")
        target_ei = self.target_ei(X)
        result = self.weights[-1] * target_ei
        incumbent = self.best_f.to(dtype=target_ei.dtype, device=target_ei.device)
        for weight, model in zip(self.weights[:-1], self.source_models, strict=True):
            mean = model.posterior(X).mean.squeeze(-1).squeeze(-1)
            improvement = mean - incumbent if self.maximize else incumbent - mean
            result = result + weight * improvement.clamp_min(0)
        return result
