"""Classification posterior contracts."""

from robotorchan.models.classification.posterior.ensemble import (
    ClassificationEnsemblePosterior,
    make_classification_ensemble_posterior,
)
from robotorchan.models.classification.posterior.probability import (
    ClassificationProbabilityPosterior,
)

__all__ = [
    "ClassificationEnsemblePosterior",
    "ClassificationProbabilityPosterior",
    "make_classification_ensemble_posterior",
]
