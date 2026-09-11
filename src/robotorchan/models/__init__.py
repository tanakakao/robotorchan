"""Surrogate-model extensions and BoTorch-compatible wrappers."""

from robotorchan.models.base import UnsupportedModelOperationError
from robotorchan.models.mixed import MixedSingleTaskGP
from robotorchan.models.multi_fidelity import SingleTaskMultiFidelityGP
from robotorchan.models.single_task import SingleTaskGP

__all__ = [
    "MixedSingleTaskGP",
    "SingleTaskGP",
    "SingleTaskMultiFidelityGP",
    "UnsupportedModelOperationError",
]
