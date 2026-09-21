"""Acquisition compatibility helpers for non-GP surrogate models."""

from __future__ import annotations

from collections.abc import Callable

from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.models.model import Model


def validate_non_gp_acquisition(
    model: Model,
    acquisition: AcquisitionFunction,
) -> None:
    """Validate that an acquisition is compatible with a non-GP surrogate.

    Non-Gaussian empirical ensemble posteriors are integrated through BoTorch's
    Monte Carlo acquisition interface. Analytic Gaussian acquisition functions
    are intentionally rejected rather than relying on distributional
    assumptions that tree ensembles do not satisfy.
    """
    if getattr(model, "supports_mll", True):
        raise TypeError("validate_non_gp_acquisition expects a non-GP model.")
    if not isinstance(acquisition, MCAcquisitionFunction):
        raise TypeError(
            "Non-GP empirical ensemble surrogates require a Monte Carlo acquisition function."
        )


def make_non_gp_acquisition(
    model: Model,
    factory: Callable[..., MCAcquisitionFunction],
    /,
    **kwargs: object,
) -> MCAcquisitionFunction:
    """Construct and validate a BoTorch MC acquisition for a non-GP model."""
    acquisition = factory(model=model, **kwargs)
    validate_non_gp_acquisition(model, acquisition)
    return acquisition
