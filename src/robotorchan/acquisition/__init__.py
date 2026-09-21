"""Acquisition-function extensions for BoTorch."""

from robotorchan.acquisition.active_learning import PosteriorStd, PosteriorVariance
from robotorchan.acquisition.non_gp import make_non_gp_acquisition, validate_non_gp_acquisition
from robotorchan.acquisition.sampling import select_thompson_candidates

__all__ = [
    "make_non_gp_acquisition",
    "PosteriorStd",
    "PosteriorVariance",
    "select_thompson_candidates",
    "validate_non_gp_acquisition",
]
