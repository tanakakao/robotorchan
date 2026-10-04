"""Acquisition-function extensions for BoTorch."""

from robotorchan.acquisition.active_learning import (
    BoundaryVariance,
    ExpectedPredictiveInformationGain,
    PosteriorStd,
    PosteriorVariance,
    RandomizedStraddle,
    Straddle,
)
from robotorchan.acquisition.capabilities import (
    AcquisitionCapabilities,
    AcquisitionTarget,
    AcquisitionPurpose,
    AcquisitionRegistryEntry,
    PosteriorRequirement,
)
from robotorchan.acquisition.non_gp import make_non_gp_acquisition, validate_non_gp_acquisition
from robotorchan.acquisition.registry import get_acquisition_registry_entry
from robotorchan.acquisition.sampling import select_thompson_candidates

__all__ = [
    "AcquisitionCapabilities",
    "AcquisitionTarget",
    "AcquisitionPurpose",
    "AcquisitionRegistryEntry",
    "BoundaryVariance",
    "ExpectedPredictiveInformationGain",
    "PosteriorRequirement",
    "PosteriorStd",
    "PosteriorVariance",
    "RandomizedStraddle",
    "Straddle",
    "get_acquisition_registry_entry",
    "make_non_gp_acquisition",
    "select_thompson_candidates",
    "validate_non_gp_acquisition",
]
