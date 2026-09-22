"""Acquisition-function extensions for BoTorch."""

from robotorchan.acquisition.active_learning import (
    BoundaryVariance,
    ExpectedPredictiveInformationGain,
    PosteriorStd,
    PosteriorVariance,
    RandomizedStraddle,
    Straddle,
)
from robotorchan.acquisition.non_gp import make_non_gp_acquisition, validate_non_gp_acquisition
from robotorchan.acquisition.samplers import make_model_sampler
from robotorchan.acquisition.sampling import select_thompson_candidates

__all__ = [
    "BoundaryVariance",
    "ExpectedPredictiveInformationGain",
    "PosteriorStd",
    "PosteriorVariance",
    "RandomizedStraddle",
    "Straddle",
    "make_model_sampler",
    "make_non_gp_acquisition",
    "select_thompson_candidates",
    "validate_non_gp_acquisition",
]
