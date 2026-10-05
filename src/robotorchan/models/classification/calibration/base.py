"""Post-hoc probability calibration contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor, nn


class ProbabilityCalibrator(nn.Module, ABC):
    """Transform predictive class probabilities without refitting a classifier."""

    @abstractmethod
    def forward(self, probabilities: Tensor) -> Tensor:
        """Return calibrated probabilities with the same shape."""
