"""Temperature scaling for classification probabilities."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from robotorchan.models.classification.calibration.base import ProbabilityCalibrator


class TemperatureScalingCalibrator(ProbabilityCalibrator):
    """Calibrate class probabilities with a positive scalar temperature."""

    def __init__(self, temperature: float = 1.0) -> None:
        super().__init__()
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("temperature must be finite and positive.")
        self.log_temperature = nn.Parameter(torch.tensor(math.log(temperature)))

    @property
    def temperature(self) -> Tensor:
        """Return the positive calibration temperature."""
        return self.log_temperature.exp()

    def forward(self, probabilities: Tensor) -> Tensor:
        """Apply temperature scaling in log-probability space."""
        if probabilities.ndim < 1 or probabilities.shape[-1] < 2:
            raise ValueError("probabilities must end with at least two classes.")
        if not torch.is_floating_point(probabilities):
            raise ValueError("probabilities must use a floating-point dtype.")
        if not torch.isfinite(probabilities).all():
            raise ValueError("probabilities must be finite.")
        if torch.any((probabilities < 0.0) | (probabilities > 1.0)):
            raise ValueError("probabilities must lie in [0, 1].")
        sums = probabilities.sum(dim=-1)
        if not torch.allclose(sums, torch.ones_like(sums), atol=1e-6, rtol=1e-6):
            raise ValueError("Class probabilities must sum to one.")
        tiny = torch.finfo(probabilities.dtype).tiny
        logits = probabilities.clamp_min(tiny).log()
        return torch.softmax(logits / self.temperature.to(probabilities), dim=-1)

    def fit(
        self,
        probabilities: Tensor,
        targets: Tensor,
        *,
        max_iter: int = 100,
    ) -> TemperatureScalingCalibrator:
        """Fit temperature by validation-set negative log likelihood."""
        if max_iter < 1:
            raise ValueError("max_iter must be positive.")
        calibrated = self(probabilities)
        if targets.shape != probabilities.shape[:-1]:
            raise ValueError("targets must match probability leading dimensions.")
        if targets.dtype not in (torch.int32, torch.int64):
            raise ValueError("targets must contain integer class indices.")
        if torch.any((targets < 0) | (targets >= probabilities.shape[-1])):
            raise ValueError("targets contain an invalid class index.")
        del calibrated
        optimizer = torch.optim.LBFGS([self.log_temperature], max_iter=max_iter)

        def closure() -> Tensor:
            optimizer.zero_grad()
            loss = F.nll_loss(self(probabilities).log(), targets.reshape(-1))
            loss.backward()
            return loss

        optimizer.step(closure)
        return self
