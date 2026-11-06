"""Active-learning acquisition functions."""

from robotorchan.acquisition.active_learning.classification import (
    BALD,
    LatentStraddle,
    MarginUncertainty,
    PredictiveEntropy,
    ProbabilityVariance,
    qMarginUncertainty,
    qPredictiveEntropy,
)
from robotorchan.acquisition.active_learning.epig import ExpectedPredictiveInformationGain
from robotorchan.acquisition.active_learning.joint_entropy import qJointPosteriorEntropy
from robotorchan.acquisition.active_learning.randomized_straddle import qRandomizedStraddle
from robotorchan.acquisition.active_learning.straddle import qBoundaryVariance, qStraddle
from robotorchan.acquisition.active_learning.variance import qPosteriorStd, qPosteriorVariance

__all__ = [
    "BALD",
    "ExpectedPredictiveInformationGain",
    "LatentStraddle",
    "MarginUncertainty",
    "PredictiveEntropy",
    "ProbabilityVariance",
    "qBoundaryVariance",
    "qJointPosteriorEntropy",
    "qMarginUncertainty",
    "qPosteriorStd",
    "qPosteriorVariance",
    "qPredictiveEntropy",
    "qRandomizedStraddle",
    "qStraddle",
]
