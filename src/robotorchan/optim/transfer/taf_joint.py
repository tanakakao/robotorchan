"""Joint q-batch transfer acquisition using target MC improvement."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.acquisition.objective import GenericMCObjective
from botorch.models.model import Model
from botorch.sampling.normal import SobolQMCNormalSampler
from torch import Tensor


class qTransferAcquisitionFunction(AcquisitionFunction):
    """Joint TAF: target qEI plus source maximum posterior-mean improvement.

    Source improvement is deterministic, as in the analytic q=1 TAF.
    Target improvement is MC qEI, including within-batch correlation.
    """

    def __init__(
        self,
        source_models: Sequence[Model],
        target_model: Model,
        weights: Tensor,
        best_f: float,
        *,
        maximize: bool = True,
        mc_samples: int = 128,
    ) -> None:
        if target_model.num_outputs != 1 or any(model.num_outputs != 1 for model in source_models):
            raise ValueError("TAF supports scalar-output models only")
        if weights.ndim != 1 or weights.numel() != len(source_models) + 1:
            raise ValueError("weights must include each source and the target")
        if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
            raise ValueError("weights must be finite and nonnegative")
        if not bool(torch.isclose(weights.sum(), weights.new_tensor(1.0))):
            raise ValueError("weights must sum to one")
        if mc_samples < 1:
            raise ValueError("mc_samples must be positive")
        super().__init__(model=target_model)
        self.source_models = torch.nn.ModuleList(source_models)
        self.register_buffer("weights", weights.detach().clone())
        self.register_buffer("best_f", torch.as_tensor(best_f).detach().clone())
        self.maximize = maximize
        self.target_qei = qExpectedImprovement(
            model=target_model,
            best_f=best_f if maximize else -best_f,
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples])),
            objective=None if maximize else GenericMCObjective(lambda samples, X=None: -samples.squeeze(-1)),
        )

    def forward(self, X: Tensor) -> Tensor:
        """Return a joint improvement value for each t-batch."""
        result = self.weights[-1] * self.target_qei(X)
        incumbent = self.best_f.to(dtype=result.dtype, device=result.device)
        for weight, model in zip(self.weights[:-1], self.source_models, strict=True):
            mean = model.posterior(X).mean.squeeze(-1)
            improvement = mean - incumbent if self.maximize else incumbent - mean
            result = result + weight * improvement.max(dim=-1).values.clamp_min(0)
        return result
