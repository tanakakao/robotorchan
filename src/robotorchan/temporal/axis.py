"""Time coordinates and explicit training-only normalization."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

import torch
from torch import Tensor


def datetime_seconds(values: Sequence[datetime], *, origin: datetime) -> Tensor:
    """Convert timezone-aware datetimes to elapsed seconds from a supplied origin."""
    if origin.utcoffset() is None or any(value.utcoffset() is None for value in values):
        raise ValueError("timezone-aware datetimes are required")
    return torch.tensor(
        [value.timestamp() - origin.timestamp() for value in values], dtype=torch.float64
    )


def validate_times(times: Tensor) -> None:
    """Require a finite, nonempty one-dimensional floating-point time coordinate."""
    if times.ndim != 1 or not times.numel() or not times.is_floating_point():
        raise ValueError("times must be a nonempty floating-point vector")
    if not torch.isfinite(times).all():
        raise ValueError("times must be finite")


@dataclass(frozen=True)
class TimeAxis:
    """Affine time transform fitted exclusively on supplied training times."""

    origin: Tensor
    scale: Tensor

    @classmethod
    def fit(cls, train_times: Tensor) -> TimeAxis:
        """Fit an origin and positive range; use unit scale for constant times."""
        validate_times(train_times)
        origin = train_times.min().detach().clone()
        span = train_times.max() - origin
        if not torch.isfinite(span):
            raise ValueError("time range must be finite")
        scale = torch.where(span > 0, span, torch.ones_like(span)).detach().clone()
        return cls(origin, scale)

    def transform(self, times: Tensor) -> Tensor:
        """Map future and historical times without clipping or refitting."""
        validate_times(times)
        return (times - self.origin.to(times)) / self.scale.to(times)

    def inverse_transform(self, times: Tensor) -> Tensor:
        """Restore original time coordinates."""
        validate_times(times)
        return times * self.scale.to(times) + self.origin.to(times)
