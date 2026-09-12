"""Surrogate-model extensions and BoTorch-compatible wrappers."""

from robotorchan.models.additive import MixedOrthogonalAdditiveGP, OrthogonalAdditiveGP
from robotorchan.models.base import UnsupportedModelOperationError
from robotorchan.models.contextual import LCEAGP, LCEMGP, SACGP, MixedLCEMGP
from robotorchan.models.fully_bayesian import (
    MixedSaasFullyBayesianMultiTaskGP,
    MixedSaasFullyBayesianSingleTaskGP,
    SaasFullyBayesianMultiTaskGP,
    SaasFullyBayesianSingleTaskGP,
)
from robotorchan.models.heterogeneous import HeterogeneousMTGP, MixedHeterogeneousMTGP
from robotorchan.models.hierarchical import (
    HierarchicalConditionalKernelGP,
    HierarchicalConditionalKernelMultiTaskGP,
    MixedHierarchicalConditionalKernelGP,
    MixedHierarchicalConditionalKernelMultiTaskGP,
)
from robotorchan.models.higher_order import HigherOrderGP, MixedHigherOrderGP
from robotorchan.models.latent_kronecker import LatentKroneckerGP, MixedLatentKroneckerGP
from robotorchan.models.map_saas import (
    AdditiveMapSaasSingleTaskGP,
    EnsembleMapSaasSingleTaskGP,
    MixedAdditiveMapSaasSingleTaskGP,
    MixedEnsembleMapSaasSingleTaskGP,
)
from robotorchan.models.model_list import ModelListGP
from robotorchan.models.multi_fidelity import (
    MixedSingleTaskMultiFidelityGP,
    SingleTaskMultiFidelityGP,
)
from robotorchan.models.multitask import (
    KroneckerMultiTaskGP,
    MixedKroneckerMultiTaskGP,
    MixedMultiTaskGP,
    MultiTaskGP,
)
from robotorchan.models.pairwise import PairwiseGP
from robotorchan.models.robust import (
    MixedRobustRelevancePursuitSingleTaskGP,
    RobustRelevancePursuitSingleTaskGP,
)
from robotorchan.models.single_task import MixedSingleTaskGP, SingleTaskGP
from robotorchan.models.variational import MixedSingleTaskVariationalGP, SingleTaskVariationalGP

__all__ = [
    "AdditiveMapSaasSingleTaskGP",
    "EnsembleMapSaasSingleTaskGP",
    "HeterogeneousMTGP",
    "HierarchicalConditionalKernelGP",
    "HierarchicalConditionalKernelMultiTaskGP",
    "HigherOrderGP",
    "KroneckerMultiTaskGP",
    "LCEAGP",
    "LCEMGP",
    "LatentKroneckerGP",
    "MixedAdditiveMapSaasSingleTaskGP",
    "MixedEnsembleMapSaasSingleTaskGP",
    "MixedHeterogeneousMTGP",
    "MixedHierarchicalConditionalKernelGP",
    "MixedHierarchicalConditionalKernelMultiTaskGP",
    "MixedHigherOrderGP",
    "MixedKroneckerMultiTaskGP",
    "MixedLCEMGP",
    "MixedLatentKroneckerGP",
    "MixedMultiTaskGP",
    "MixedOrthogonalAdditiveGP",
    "MixedRobustRelevancePursuitSingleTaskGP",
    "MixedSaasFullyBayesianMultiTaskGP",
    "MixedSaasFullyBayesianSingleTaskGP",
    "MixedSingleTaskGP",
    "MixedSingleTaskMultiFidelityGP",
    "MixedSingleTaskVariationalGP",
    "ModelListGP",
    "MultiTaskGP",
    "OrthogonalAdditiveGP",
    "PairwiseGP",
    "RobustRelevancePursuitSingleTaskGP",
    "SACGP",
    "SaasFullyBayesianMultiTaskGP",
    "SaasFullyBayesianSingleTaskGP",
    "SingleTaskGP",
    "SingleTaskMultiFidelityGP",
    "SingleTaskVariationalGP",
    "UnsupportedModelOperationError",
]
