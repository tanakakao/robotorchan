"""Active-learning acquisition functions."""

from robotorchan.acquisition.active_learning.classification import (
    BALD,
    LatentStraddle,
    MarginUncertainty,
    PredictiveEntropy,
    ProbabilityVariance,
)
from robotorchan.acquisition.active_learning.epig import ExpectedPredictiveInformationGain
from robotorchan.acquisition.active_learning.randomized_straddle import RandomizedStraddle
from robotorchan.acquisition.active_learning.straddle import BoundaryVariance, Straddle
from robotorchan.acquisition.active_learning.variance import PosteriorStd, PosteriorVariance

__all__ = [
    "BALD",
    "BoundaryVariance",
    "ExpectedPredictiveInformationGain",
    "LatentStraddle",
    "MarginUncertainty",
    "PosteriorStd",
    "PosteriorVariance",
    "PredictiveEntropy",
    "ProbabilityVariance",
    "RandomizedStraddle",
    "Straddle",
]
