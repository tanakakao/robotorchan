"""Surrogate-model extensions and BoTorch-compatible wrappers."""

from robotorchan.models.base import UnsupportedModelOperationError
from robotorchan.models.single_task import SingleTaskGP

__all__ = ["SingleTaskGP", "UnsupportedModelOperationError"]
