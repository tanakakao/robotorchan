"""Surrogate-model extensions and BoTorch-compatible wrappers."""

from robotorchan.models.base import UnsupportedModelOperationError
from robotorchan.models.expressive.deep_gp import (
    MixedMultiTaskDeepGP,
    MixedSingleTaskDeepGP,
    MultiTaskDeepGP,
    SingleTaskDeepGP,
)
from robotorchan.models.expressive.infinite_width_bnn import (
    InfiniteWidthBNNGP,
    InfiniteWidthBNNMultiTaskGP,
    MixedInfiniteWidthBNNGP,
    MixedInfiniteWidthBNNMultiTaskGP,
)
from robotorchan.models.expressive.spectral_mixture import (
    MixedSpectralMixtureGP,
    MixedSpectralMixtureMultiTaskGP,
    SpectralMixtureGP,
    SpectralMixtureMultiTaskGP,
)
from robotorchan.models.high_dimensional.alebo import ALEBOGP
from robotorchan.models.high_dimensional.fully_bayesian import (
    MixedSaasFullyBayesianMultiTaskGP,
    MixedSaasFullyBayesianSingleTaskGP,
    SaasFullyBayesianMultiTaskGP,
    SaasFullyBayesianSingleTaskGP,
)
from robotorchan.models.high_dimensional.map_saas import (
    AdditiveMapSaasSingleTaskGP,
    EnsembleMapSaasSingleTaskGP,
    MixedAdditiveMapSaasSingleTaskGP,
    MixedEnsembleMapSaasSingleTaskGP,
)
from robotorchan.models.high_dimensional.reduced.base import (
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
from robotorchan.models.high_dimensional.reduced.joint_multitask import (
    HybridAutoEncoderKroneckerMultiTaskGP,
    HybridAutoEncoderMultiTaskGP,
    JointEncoderKroneckerMultiTaskGP,
    JointEncoderMultiTaskGP,
    JointVAEKroneckerMultiTaskGP,
    JointVAEMultiTaskGP,
    MixedJointEncoderMultiTaskGP,
)
from robotorchan.models.high_dimensional.reduced.joint_neural import (
    HybridAutoEncoderGP,
    JointEncoderGP,
    MixedHybridAutoEncoderGP,
    MixedJointEncoderGP,
)
from robotorchan.models.high_dimensional.reduced.joint_vae import JointVAEGP, MixedJointVAEGP
from robotorchan.models.high_dimensional.reduced.mixed import MixedReducedGP
from robotorchan.models.high_dimensional.reduced.mixed_multitask import (
    MixedReducedKroneckerMultiTaskGP,
    MixedReducedMultiTaskGP,
)
from robotorchan.models.high_dimensional.reduced.multitask import (
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
from robotorchan.models.high_dimensional.reduced.supervised_neural import (
    MixedSupervisedAutoEncoderGP,
    SupervisedAutoEncoderGP,
)
from robotorchan.models.high_dimensional.reduced.vae import (
    VAEGP,
    MixedSupervisedVAEGP,
    MixedVAEGP,
    SupervisedVAEGP,
)
from robotorchan.models.non_gp.extra_trees import ExtraTreesSurrogate
from robotorchan.models.non_gp.gradient_boosting import GradientBoostingSurrogate
from robotorchan.models.non_gp.hist_gradient_boosting import HistGradientBoostingSurrogate
from robotorchan.models.non_gp.ngboost import NGBoostSurrogate
from robotorchan.models.non_gp.random_forest import RandomForestSurrogate
from robotorchan.models.preference.pairwise import PairwiseGP
from robotorchan.models.robust.contaminated import (
    ContaminatedMultiTaskGP,
    ContaminatedSingleTaskGP,
    MixedContaminatedMultiTaskGP,
    MixedContaminatedSingleTaskGP,
)
from robotorchan.models.robust.joint_heteroskedastic import (
    JointHeteroskedasticSingleTaskGP,
    MixedJointHeteroskedasticSingleTaskGP,
)
from robotorchan.models.robust.nonstationary import (
    MixedNonstationaryMultiTaskGP,
    MixedNonstationarySingleTaskGP,
    NonstationaryKroneckerMultiTaskGP,
    NonstationaryMultiTaskGP,
    NonstationarySingleTaskGP,
)
from robotorchan.models.robust.replicate_noise import (
    MixedReplicateNoiseSingleTaskGP,
    ReplicateNoiseMultiFidelityGP,
    ReplicateNoiseSingleTaskGP,
)
from robotorchan.models.robust.robust import (
    HeteroskedasticMultiFidelityGP,
    HeteroskedasticMultiTaskGP,
    HeteroskedasticSingleTaskGP,
    MixedHeteroskedasticMultiTaskGP,
    MixedHeteroskedasticSingleTaskGP,
    MixedRobustRelevancePursuitMultiTaskGP,
    MixedRobustRelevancePursuitSingleTaskGP,
    RobustRelevancePursuitMultiTaskGP,
    RobustRelevancePursuitSingleTaskGP,
)
from robotorchan.models.robust.student_t import (
    MixedStudentTMultiTaskGP,
    MixedStudentTSingleTaskGP,
    StudentTMultiTaskGP,
    StudentTSingleTaskGP,
)
from robotorchan.models.standard.model_list import ModelListGP
from robotorchan.models.standard.multi_fidelity import (
    MixedSingleTaskMultiFidelityGP,
    PCAMultiFidelityGP,
    PLSMultiFidelityGP,
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
from robotorchan.models.structured.additive import MixedOrthogonalAdditiveGP, OrthogonalAdditiveGP
from robotorchan.models.structured.contextual import LCEAGP, LCEMGP, SACGP, MixedLCEMGP
from robotorchan.models.structured.heterogeneous import HeterogeneousMTGP, MixedHeterogeneousMTGP
from robotorchan.models.structured.hierarchical import (
    HierarchicalConditionalKernelGP,
    HierarchicalConditionalKernelMultiTaskGP,
    MixedHierarchicalConditionalKernelGP,
    MixedHierarchicalConditionalKernelMultiTaskGP,
)
from robotorchan.models.structured.higher_order import HigherOrderGP, MixedHigherOrderGP
from robotorchan.models.structured.latent_kronecker import LatentKroneckerGP, MixedLatentKroneckerGP
from robotorchan.models.uncertain.uncertain_categorical import UncertainCategoricalSingleTaskGP
from robotorchan.models.uncertain.uncertain_input import (
    MixedUncertainInputSingleTaskGP,
    UncertainInputSingleTaskGP,
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
    "ExtraTreesSurrogate",
    "GradientBoostingSurrogate",
    "HeterogeneousMTGP",
    "HeteroskedasticMultiFidelityGP",
    "HeteroskedasticMultiTaskGP",
    "HeteroskedasticSingleTaskGP",
    "HierarchicalConditionalKernelGP",
    "HierarchicalConditionalKernelMultiTaskGP",
    "HigherOrderGP",
    "HistGradientBoostingSurrogate",
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
    "NGBoostSurrogate",
    "NonstationaryKroneckerMultiTaskGP",
    "NonstationaryMultiTaskGP",
    "NonstationarySingleTaskGP",
    "OrthogonalAdditiveGP",
    "OutputPCAGP",
    "OutputPLSGP",
    "PCAKroneckerMultiTaskGP",
    "PCAMultiFidelityGP",
    "PCAMultiTaskGP",
    "PLSKroneckerMultiTaskGP",
    "PLSMultiFidelityGP",
    "PLSKroneckerMultiTaskGP",
    "PLSMultiTaskGP",
    "PairwiseGP",
    "RandomForestSurrogate",
    "RandomProjectionGP",
    "RandomProjectionKroneckerMultiTaskGP",
    "RandomProjectionMultiTaskGP",
    "ReducedGP",
    "ReducedKroneckerMultiTaskGP",
    "ReducedMultiTaskGP",
    "ReplicateNoiseMultiFidelityGP",
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
