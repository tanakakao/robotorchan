"""Binary classification ensemble models."""

from robotorchan.models.classification.binary.ensemble.gp import (
    GPBinaryClassificationEnsemble,
)
from robotorchan.models.classification.binary.ensemble.heterogeneous import (
    HeterogeneousBinaryClassificationEnsemble,
)

__all__ = [
    "GPBinaryClassificationEnsemble",
    "HeterogeneousBinaryClassificationEnsemble",
]
