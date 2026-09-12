"""Surrogate-model extensions and BoTorch-compatible wrappers."""

from robotorchan.models.additive import OrthogonalAdditiveGP
from robotorchan.models.base import UnsupportedModelOperationError
from robotorchan.models.contextual import LCEAGP, LCEMGP, SACGP
from robotorchan.models.fully_bayesian import (
    SaasFullyBayesianMultiTaskGP,
    SaasFullyBayesianSingleTaskGP,
)
from robotorchan.models.heterogeneous import HeterogeneousMTGP
from robotorchan.models.hierarchical import (
    HierarchicalConditionalKernelGP,
    HierarchicalConditionalKernelMultiTaskGP,
)
from robotorchan.models.higher_order import HigherOrderGP
from robotorchan.models.latent_kronecker import LatentKroneckerGP
from robotorchan.models.map_saas import (
    AdditiveMapSaasSingleTaskGP,
    EnsembleMapSaasSingleTaskGP,
)
from robotorchan.models.model_list import ModelListGP
from robotorchan.models.multi_fidelity import SingleTaskMultiFidelityGP
from robotorchan.models.multitask import KroneckerMultiTaskGP, MultiTaskGP
from robotorchan.models.pairwise import PairwiseGP
from robotorchan.models.reduction import (
    InputReducer,
    OutputReducer,
    ReducerNotFittedError,
    ReductionMixin,
    TensorReducer,
)
from robotorchan.models.robust import RobustRelevancePursuitSingleTaskGP
from robotorchan.models.single_task import MixedSingleTaskGP, SingleTaskGP
from robotorchan.models.variational import SingleTaskVariationalGP

__all__ = [
    "LCEAGP",
    "LCEMGP",
    "SACGP",
    "AdditiveMapSaasSingleTaskGP",
    "EnsembleMapSaasSingleTaskGP",
    "HeterogeneousMTGP",
    "HierarchicalConditionalKernelGP",
    "HierarchicalConditionalKernelMultiTaskGP",
    "HigherOrderGP",
    "InputReducer",
    "KroneckerMultiTaskGP",
    "LatentKroneckerGP",
    "MixedSingleTaskGP",
    "ModelListGP",
    "MultiTaskGP",
    "OrthogonalAdditiveGP",
    "OutputReducer",
    "PairwiseGP",
    "ReducerNotFittedError",
    "ReductionMixin",
    "RobustRelevancePursuitSingleTaskGP",
    "SaasFullyBayesianMultiTaskGP",
    "SaasFullyBayesianSingleTaskGP",
    "SingleTaskGP",
    "SingleTaskMultiFidelityGP",
    "SingleTaskVariationalGP",
    "TensorReducer",
    "UnsupportedModelOperationError",
]
