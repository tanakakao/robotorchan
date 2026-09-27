"""Contract tests for the explicit TuRBO incumbent API."""

import pytest
import torch

from robotorchan.optim import TuRBOStrategy


def test_turbo_requires_explicit_center() -> None:
    bounds = torch.stack([torch.zeros(3), torch.ones(3)])

    with pytest.raises(TypeError, match="center"):
        TuRBOStrategy(bounds)  # type: ignore[call-arg]


def test_turbo_rejects_out_of_bounds_center() -> None:
    bounds = torch.stack([torch.zeros(3), torch.ones(3)])

    with pytest.raises(ValueError, match="center must lie inside bounds"):
        TuRBOStrategy(bounds, center=torch.tensor([0.5, 1.2, 0.5]))
