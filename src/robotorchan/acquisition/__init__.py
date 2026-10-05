"""Acquisition-function extensions for BoTorch."""

from robotorchan.acquisition.active_learning import (
    BALD,
    BoundaryVariance,
    ExpectedPredictiveInformationGain,
    LatentStraddle,
    MarginUncertainty,
    PosteriorStd,
    PosteriorVariance,
    PredictiveEntropy,
    ProbabilityVariance,
    RandomizedStraddle,
    Straddle,
)
from robotorchan.acquisition.capabilities import (
    AcquisitionCapabilities,
    AcquisitionPurpose,
    AcquisitionRegistryEntry,
    PosteriorRequirement,
)
from robotorchan.acquisition.classification_constraints import (
    ClassificationProbabilityOfFeasibility,
    FeasibilityWeightedAcquisition,
    RobustClassificationProbabilityOfFeasibility,
)
from robotorchan.acquisition.non_gp import make_non_gp_acquisition, validate_non_gp_acquisition
from robotorchan.acquisition.registry import get_acquisition_registry_entry
from robotorchan.acquisition.sampling import select_thompson_candidates

__all__ = [
    "BALD",
    "AcquisitionCapabilities",
    "AcquisitionPurpose",
    "AcquisitionRegistryEntry",
    "BoundaryVariance",
    "ClassificationProbabilityOfFeasibility",
    "FeasibilityWeightedAcquisition",
    "ExpectedPredictiveInformationGain",
    "LatentStraddle",
    "MarginUncertainty",
    "PosteriorRequirement",
    "PosteriorStd",
    "PosteriorVariance",
    "PredictiveEntropy",
    "ProbabilityVariance",
    "RandomizedStraddle",
    "RobustClassificationProbabilityOfFeasibility",
    "Straddle",
    "get_acquisition_registry_entry",
    "make_non_gp_acquisition",
    "select_thompson_candidates",
    "validate_non_gp_acquisition",
]
