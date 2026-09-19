"""Surrogate-model extensions and BoTorch-compatible wrappers."""

from robotorchan.models.additive import MixedOrthogonalAdditiveGP, OrthogonalAdditiveGP
from robotorchan.models.alebo import ALEBOGP
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
from robotorchan.models.reduced.base import (
    PCAGP,
    PLSGP,
    AutoEncoderGP,
    MixedAutoEncoderGP,
    MixedPCAGP,
    MixedPLSGP,
    MixedRandomProjectionGP,
    OutputPCAGP,
    OutputPLSGP,
    RandomProjectionGP,
    ReducedGP,
)
from robotorchan.models.reduced.joint_neural import (
    HybridAutoEncoderGP,
    JointEncoderGP,
    MixedHybridAutoEncoderGP,
    MixedJointEncoderGP,
)
from robotorchan.models.reduced.joint_vae import JointVAEGP, MixedJointVAEGP
from robotorchan.models.reduced.mixed import MixedReducedGP
from robotorchan.models.reduced.multitask import (
    AutoEncoderKroneckerMultiTaskGP,
    AutoEncoderMultiTaskGP,
    PCAKroneckerMultiTaskGP,
    PCAMultiTaskGP,
    PLSKroneckerMultiTaskGP,
    PLSMultiTaskGP,
    RandomProjectionKroneckerMultiTaskGP,
    RandomProjectionMultiTaskGP,
    ReducedKroneckerMultiTaskGP,
    ReducedMultiTaskGP,
    SupervisedAutoEncoderKroneckerMultiTaskGP,
    SupervisedAutoEncoderMultiTaskGP,
    SupervisedVAEKroneckerMultiTaskGP,
    SupervisedVAEMultiTaskGP,
    VAEKroneckerMultiTaskGP,
    VAEMultiTaskGP,
)
from robotorchan.models.reduced.supervised_neural import (
    MixedSupervisedAutoEncoderGP,
    SupervisedAutoEncoderGP,
)
from robotorchan.models.reduced.vae import (
    VAEGP,
    MixedSupervisedVAEGP,
    MixedVAEGP,
    SupervisedVAEGP,
)
from robotorchan.models.robust import (
    MixedRobustRelevancePursuitSingleTaskGP,
    RobustRelevancePursuitSingleTaskGP,
)
from robotorchan.models.single_task import MixedSingleTaskGP, SingleTaskGP
from robotorchan.models.variational import (
    MixedSingleTaskVariationalGP,
    SingleTaskVariationalGP,
)

__all__ = [
    "ALEBOGP",
    "LCEAGP",
    "LCEMGP",
    "PCAGP",
    "PLSGP",
    "SACGP",
    "VAEGP",
    "AdditiveMapSaasSingleTaskGP",
    "AutoEncoderGP",
    "AutoEncoderKroneckerMultiTaskGP",
    "AutoEncoderMultiTaskGP",
    "EnsembleMapSaasSingleTaskGP",
    "HeterogeneousMTGP",
    "HierarchicalConditionalKernelGP",
    "HierarchicalConditionalKernelMultiTaskGP",
    "HigherOrderGP",
    "HybridAutoEncoderGP",
    "JointEncoderGP",
    "JointVAEGP",
    "KroneckerMultiTaskGP",
    "LatentKroneckerGP",
    "MixedAdditiveMapSaasSingleTaskGP",
    "MixedAutoEncoderGP",
    "MixedEnsembleMapSaasSingleTaskGP",
    "MixedHeterogeneousMTGP",
    "MixedHierarchicalConditionalKernelGP",
    "MixedHierarchicalConditionalKernelMultiTaskGP",
    "MixedHigherOrderGP",
    "MixedHybridAutoEncoderGP",
    "MixedJointEncoderGP",
    "MixedJointVAEGP",
    "MixedKroneckerMultiTaskGP",
    "MixedLCEMGP",
    "MixedLatentKroneckerGP",
    "MixedMultiTaskGP",
    "MixedOrthogonalAdditiveGP",
    "MixedPCAGP",
    "MixedPLSGP",
    "MixedRandomProjectionGP",
    "MixedReducedGP",
    "MixedRobustRelevancePursuitSingleTaskGP",
    "MixedSaasFullyBayesianMultiTaskGP",
    "MixedSaasFullyBayesianSingleTaskGP",
    "MixedSingleTaskGP",
    "MixedSingleTaskMultiFidelityGP",
    "MixedSingleTaskVariationalGP",
    "MixedSupervisedAutoEncoderGP",
    "MixedSupervisedVAEGP",
    "MixedVAEGP",
    "ModelListGP",
    "MultiTaskGP",
    "OrthogonalAdditiveGP",
    "OutputPCAGP",
    "OutputPLSGP",
    "PCAKroneckerMultiTaskGP",
    "PCAMultiTaskGP",
    "PLSKroneckerMultiTaskGP",
    "PLSMultiTaskGP",
    "PairwiseGP",
    "RandomProjectionGP",
    "RandomProjectionKroneckerMultiTaskGP",
    "RandomProjectionMultiTaskGP",
    "ReducedGP",
    "ReducedKroneckerMultiTaskGP",
    "ReducedMultiTaskGP",
    "RobustRelevancePursuitSingleTaskGP",
    "SaasFullyBayesianMultiTaskGP",
    "SaasFullyBayesianSingleTaskGP",
    "SingleTaskGP",
    "SingleTaskMultiFidelityGP",
    "SingleTaskVariationalGP",
    "SupervisedAutoEncoderGP",
    "SupervisedAutoEncoderKroneckerMultiTaskGP",
    "SupervisedAutoEncoderMultiTaskGP",
    "SupervisedVAEGP",
    "SupervisedVAEKroneckerMultiTaskGP",
    "SupervisedVAEMultiTaskGP",
    "UnsupportedModelOperationError",
    "VAEKroneckerMultiTaskGP",
    "VAEMultiTaskGP",
]
