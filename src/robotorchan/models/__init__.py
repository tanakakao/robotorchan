"""Surrogate-model extensions and BoTorch-compatible wrappers."""

from robotorchan.models.additive import MixedOrthogonalAdditiveGP, OrthogonalAdditiveGP
from robotorchan.models.alebo import ALEBOGP
from robotorchan.models.base import UnsupportedModelOperationError
from robotorchan.models.contaminated import (
    ContaminatedMultiTaskGP,
    ContaminatedSingleTaskGP,
    MixedContaminatedMultiTaskGP,
    MixedContaminatedSingleTaskGP,
)
from robotorchan.models.contextual import LCEAGP, LCEMGP, SACGP, MixedLCEMGP
from robotorchan.models.deep_gp import (
    MixedMultiTaskDeepGP,
    MixedSingleTaskDeepGP,
    MultiTaskDeepGP,
    SingleTaskDeepGP,
)
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
from robotorchan.models.infinite_width_bnn import (
    InfiniteWidthBNNGP,
    InfiniteWidthBNNMultiTaskGP,
    MixedInfiniteWidthBNNGP,
    MixedInfiniteWidthBNNMultiTaskGP,
)
from robotorchan.models.joint_heteroskedastic import (
    JointHeteroskedasticSingleTaskGP,
    MixedJointHeteroskedasticSingleTaskGP,
)
from robotorchan.models.latent_kronecker import LatentKroneckerGP, MixedLatentKroneckerGP
from robotorchan.models.map_saas import (
    AdditiveMapSaasSingleTaskGP,
    EnsembleMapSaasSingleTaskGP,
    MixedAdditiveMapSaasSingleTaskGP,
    MixedEnsembleMapSaasSingleTaskGP,
)
from robotorchan.models.standard.model_list import ModelListGP
from robotorchan.models.standard.multi_fidelity import (
    MixedSingleTaskMultiFidelityGP,
    SingleTaskMultiFidelityGP,
)
from robotorchan.models.standard.multitask import (
    KroneckerMultiTaskGP,
    MixedKroneckerMultiTaskGP,
    MixedMultiTaskGP,
    MultiTaskGP,
)
from robotorchan.models.nonstationary import (
    MixedNonstationaryMultiTaskGP,
    MixedNonstationarySingleTaskGP,
    NonstationaryMultiTaskGP,
    NonstationarySingleTaskGP,
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
from robotorchan.models.reduced.joint_multitask import (
    HybridAutoEncoderKroneckerMultiTaskGP,
    HybridAutoEncoderMultiTaskGP,
    JointEncoderKroneckerMultiTaskGP,
    JointEncoderMultiTaskGP,
    JointVAEKroneckerMultiTaskGP,
    JointVAEMultiTaskGP,
    MixedJointEncoderMultiTaskGP,
)
from robotorchan.models.reduced.joint_neural import (
    HybridAutoEncoderGP,
    JointEncoderGP,
    MixedHybridAutoEncoderGP,
    MixedJointEncoderGP,
)
from robotorchan.models.reduced.joint_vae import JointVAEGP, MixedJointVAEGP
from robotorchan.models.reduced.mixed import MixedReducedGP
from robotorchan.models.reduced.mixed_multitask import (
    MixedReducedKroneckerMultiTaskGP,
    MixedReducedMultiTaskGP,
)
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
from robotorchan.models.replicate_noise import (
    MixedReplicateNoiseSingleTaskGP,
    ReplicateNoiseSingleTaskGP,
)
from robotorchan.models.robust import (
    HeteroskedasticMultiTaskGP,
    HeteroskedasticSingleTaskGP,
    MixedHeteroskedasticMultiTaskGP,
    MixedHeteroskedasticSingleTaskGP,
    MixedRobustRelevancePursuitMultiTaskGP,
    MixedRobustRelevancePursuitSingleTaskGP,
    RobustRelevancePursuitMultiTaskGP,
    RobustRelevancePursuitSingleTaskGP,
)
from robotorchan.models.standard.single_task import MixedSingleTaskGP, SingleTaskGP
from robotorchan.models.spectral_mixture import (
    MixedSpectralMixtureGP,
    MixedSpectralMixtureMultiTaskGP,
    SpectralMixtureGP,
    SpectralMixtureMultiTaskGP,
)
from robotorchan.models.student_t import (
    MixedStudentTMultiTaskGP,
    MixedStudentTSingleTaskGP,
    StudentTMultiTaskGP,
    StudentTSingleTaskGP,
)
from robotorchan.models.uncertain_categorical import UncertainCategoricalSingleTaskGP
from robotorchan.models.uncertain_input import (
    MixedUncertainInputSingleTaskGP,
    UncertainInputSingleTaskGP,
)
from robotorchan.models.standard.variational import (
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
    "ContaminatedMultiTaskGP",
    "ContaminatedSingleTaskGP",
    "EnsembleMapSaasSingleTaskGP",
    "HeterogeneousMTGP",
    "HeteroskedasticMultiTaskGP",
    "HeteroskedasticSingleTaskGP",
    "HierarchicalConditionalKernelGP",
    "HierarchicalConditionalKernelMultiTaskGP",
    "HigherOrderGP",
    "HybridAutoEncoderGP",
    "HybridAutoEncoderKroneckerMultiTaskGP",
    "HybridAutoEncoderMultiTaskGP",
    "InfiniteWidthBNNGP",
    "InfiniteWidthBNNMultiTaskGP",
    "JointEncoderGP",
    "JointEncoderKroneckerMultiTaskGP",
    "JointEncoderMultiTaskGP",
    "JointHeteroskedasticSingleTaskGP",
    "JointVAEGP",
    "JointVAEKroneckerMultiTaskGP",
    "JointVAEMultiTaskGP",
    "KroneckerMultiTaskGP",
    "LatentKroneckerGP",
    "MixedAdditiveMapSaasSingleTaskGP",
    "MixedAutoEncoderGP",
    "MixedContaminatedMultiTaskGP",
    "MixedContaminatedSingleTaskGP",
    "MixedEnsembleMapSaasSingleTaskGP",
    "MixedHeterogeneousMTGP",
    "MixedHeteroskedasticMultiTaskGP",
    "MixedHeteroskedasticSingleTaskGP",
    "MixedHierarchicalConditionalKernelGP",
    "MixedHierarchicalConditionalKernelMultiTaskGP",
    "MixedHigherOrderGP",
    "MixedHybridAutoEncoderGP",
    "MixedInfiniteWidthBNNGP",
    "MixedInfiniteWidthBNNMultiTaskGP",
    "MixedJointEncoderGP",
    "MixedJointEncoderMultiTaskGP",
    "MixedJointHeteroskedasticSingleTaskGP",
    "MixedJointVAEGP",
    "MixedKroneckerMultiTaskGP",
    "MixedLCEMGP",
    "MixedLatentKroneckerGP",
    "MixedMultiTaskDeepGP",
    "MixedMultiTaskGP",
    "MixedNonstationaryMultiTaskGP",
    "MixedNonstationarySingleTaskGP",
    "MixedOrthogonalAdditiveGP",
    "MixedPCAGP",
    "MixedPLSGP",
    "MixedRandomProjectionGP",
    "MixedReducedGP",
    "MixedReducedKroneckerMultiTaskGP",
    "MixedReducedMultiTaskGP",
    "MixedReplicateNoiseSingleTaskGP",
    "MixedRobustRelevancePursuitMultiTaskGP",
    "MixedRobustRelevancePursuitSingleTaskGP",
    "MixedSaasFullyBayesianMultiTaskGP",
    "MixedSaasFullyBayesianSingleTaskGP",
    "MixedSingleTaskDeepGP",
    "MixedSingleTaskGP",
    "MixedSingleTaskMultiFidelityGP",
    "MixedSingleTaskVariationalGP",
    "MixedSpectralMixtureGP",
    "MixedSpectralMixtureMultiTaskGP",
    "MixedStudentTMultiTaskGP",
    "MixedStudentTSingleTaskGP",
    "MixedSupervisedAutoEncoderGP",
    "MixedSupervisedVAEGP",
    "MixedUncertainInputSingleTaskGP",
    "MixedVAEGP",
    "ModelListGP",
    "MultiTaskDeepGP",
    "MultiTaskGP",
    "NonstationaryMultiTaskGP",
    "NonstationarySingleTaskGP",
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
    "ReplicateNoiseSingleTaskGP",
    "RobustRelevancePursuitMultiTaskGP",
    "RobustRelevancePursuitSingleTaskGP",
    "SaasFullyBayesianMultiTaskGP",
    "SaasFullyBayesianSingleTaskGP",
    "SingleTaskDeepGP",
    "SingleTaskGP",
    "SingleTaskMultiFidelityGP",
    "SingleTaskVariationalGP",
    "SpectralMixtureGP",
    "SpectralMixtureMultiTaskGP",
    "StudentTMultiTaskGP",
    "StudentTSingleTaskGP",
    "SupervisedAutoEncoderGP",
    "SupervisedAutoEncoderKroneckerMultiTaskGP",
    "SupervisedAutoEncoderMultiTaskGP",
    "SupervisedVAEGP",
    "SupervisedVAEKroneckerMultiTaskGP",
    "SupervisedVAEMultiTaskGP",
    "UncertainCategoricalSingleTaskGP",
    "UncertainInputSingleTaskGP",
    "UnsupportedModelOperationError",
    "VAEKroneckerMultiTaskGP",
    "VAEMultiTaskGP",
]
