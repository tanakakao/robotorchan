"""Surrogate-model extensions and BoTorch-compatible wrappers."""

from robotorchan.models.base import UnsupportedModelOperationError
from robotorchan.models.mixed import MixedSingleTaskGP
from robotorchan.models.model_list import ModelListGP
from robotorchan.models.multi_fidelity import SingleTaskMultiFidelityGP
from robotorchan.models.multitask import KroneckerMultiTaskGP, MultiTaskGP
from robotorchan.models.single_task import SingleTaskGP
from robotorchan.models.variational import SingleTaskVariationalGP

__all__ = [
    "KroneckerMultiTaskGP",
    "MixedSingleTaskGP",
    "ModelListGP",
    "MultiTaskGP",
    "SingleTaskGP",
    "SingleTaskMultiFidelityGP",
    "SingleTaskVariationalGP",
    "UnsupportedModelOperationError",
]
