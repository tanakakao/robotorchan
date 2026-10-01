"""Acquisition compatibility helpers for non-GP surrogate models."""

from __future__ import annotations

from collections.abc import Callable

from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.models.model import Model
from botorch.sampling.index_sampler import IndexSampler
from botorch.sampling.normal import MCSampler

from robotorchan.models.registry import MODEL_REGISTRY
from robotorchan.models.capabilities import PosteriorSamplingType


def validate_non_gp_acquisition(
    model: Model,
    acquisition: AcquisitionFunction,
) -> None:
    """Validate that an acquisition is compatible with a non-GP surrogate.

    Non-GP posteriors are integrated through BoTorch's Monte Carlo acquisition
    interface. The required sampler follows the model registry's posterior
    sampling contract instead of assuming every non-GP model is an ensemble.
    """
    if getattr(model, "supports_mll", True):
        raise TypeError("validate_non_gp_acquisition expects a non-GP model.")
    if not isinstance(acquisition, MCAcquisitionFunction):
        raise TypeError("Non-GP surrogates require a Monte Carlo acquisition function.")

    model_name = type(model).__name__
    entry = MODEL_REGISTRY.get(model_name)
    if entry is None:
        raise TypeError(f"Non-GP model {model_name!r} is not registered.")
    sampling_type = entry.capabilities.posterior_sampling_type

    sampler = acquisition.sampler
    if sampler is None:
        return
    if sampling_type is PosteriorSamplingType.ENSEMBLE and not isinstance(sampler, IndexSampler):
        raise TypeError("Ensemble non-GP surrogates require IndexSampler for Monte Carlo sampling.")
    if sampling_type is PosteriorSamplingType.GAUSSIAN and isinstance(sampler, IndexSampler):
        raise TypeError("Gaussian non-GP surrogates require a Gaussian Monte Carlo sampler.")
    if sampling_type is PosteriorSamplingType.NONE:
        raise TypeError(f"Non-GP model {model_name!r} does not support posterior sampling.")
    if not isinstance(sampler, MCSampler):
        raise TypeError("Non-GP surrogates require a BoTorch Monte Carlo sampler.")


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
