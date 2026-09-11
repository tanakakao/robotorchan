"""Surrogate-model extensions and BoTorch-compatible wrappers."""

from robotorchan.models.base import UnsupportedModelOperationError
from robotorchan.models.fully_bayesian import (
    SaasFullyBayesianMultiTaskGP,
    SaasFullyBayesianSingleTaskGP,
)
from robotorchan.models.higher_order import HigherOrderGP
from robotorchan.models.latent_kronecker import LatentKroneckerGP
from robotorchan.models.mixed import MixedSingleTaskGP
from robotorchan.models.model_list import ModelListGP
from robotorchan.models.multi_fidelity import SingleTaskMultiFidelityGP
from robotorchan.models.multitask import KroneckerMultiTaskGP, MultiTaskGP
from robotorchan.models.pairwise import PairwiseGP
from robotorchan.models.single_task import SingleTaskGP
from robotorchan.models.variational import SingleTaskVariationalGP

__all__ = [
    "HigherOrderGP",
    "KroneckerMultiTaskGP",
    "LatentKroneckerGP",
    "MixedSingleTaskGP",
    "ModelListGP",
    "MultiTaskGP",
    "PairwiseGP",
    "SaasFullyBayesianMultiTaskGP",
    "SaasFullyBayesianSingleTaskGP",
    "SingleTaskGP",
    "SingleTaskMultiFidelityGP",
    "SingleTaskVariationalGP",
    "UnsupportedModelOperationError",
]
