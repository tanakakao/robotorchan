"""High-dimensional classification surrogate models."""

from robotorchan.models.classification.high_dimensional.map_saas import (
    MapSaasBinarySingleTaskGPClassifier,
)
from robotorchan.models.classification.high_dimensional.saas import (
    SaasBinarySingleTaskGPClassifier,
)

__all__ = ["MapSaasBinarySingleTaskGPClassifier", "SaasBinarySingleTaskGPClassifier"]
