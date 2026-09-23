"""Standard BoTorch-compatible surrogate models."""

from robotorchan.models.standard.model_list import ModelListGP
from robotorchan.models.standard.multi_fidelity import (
    MixedSingleTaskMultiFidelityGP,
    PCAMultiFidelityGP,
    PLSMultiFidelityGP,
    RandomProjectionMultiFidelityGP,
    SingleTaskMultiFidelityGP,
)
from robotorchan.models.standard.multitask import (
    KroneckerMultiTaskGP,
    MixedKroneckerMultiTaskGP,
    MixedMultiTaskGP,
    MultiTaskGP,
)
from robotorchan.models.standard.single_task import MixedSingleTaskGP, SingleTaskGP
from robotorchan.models.standard.variational import (
    MixedSingleTaskVariationalGP,
    SingleTaskVariationalGP,
)

__all__ = [
    "KroneckerMultiTaskGP",
    "MixedKroneckerMultiTaskGP",
    "MixedMultiTaskGP",
    "MixedSingleTaskGP",
    "MixedSingleTaskMultiFidelityGP",
    "MixedSingleTaskVariationalGP",
    "ModelListGP",
    "MultiTaskGP",
    "PCAMultiFidelityGP",
    "PLSMultiFidelityGP",
    "RandomProjectionMultiFidelityGP",
    "SingleTaskGP",
    "SingleTaskMultiFidelityGP",
    "SingleTaskVariationalGP",
]
