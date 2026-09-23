"""Robust, noisy-observation, and nonstationary surrogate models."""

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
    HeteroskedasticKroneckerMultiTaskGP,
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

__all__ = [
    "ContaminatedMultiTaskGP",
    "ContaminatedSingleTaskGP",
    "HeteroskedasticKroneckerMultiTaskGP",
    "HeteroskedasticMultiFidelityGP",
    "HeteroskedasticMultiTaskGP",
    "HeteroskedasticSingleTaskGP",
    "JointHeteroskedasticSingleTaskGP",
    "MixedContaminatedMultiTaskGP",
    "MixedContaminatedSingleTaskGP",
    "MixedHeteroskedasticMultiTaskGP",
    "MixedHeteroskedasticSingleTaskGP",
    "MixedJointHeteroskedasticSingleTaskGP",
    "MixedNonstationaryMultiTaskGP",
    "MixedNonstationarySingleTaskGP",
    "MixedReplicateNoiseSingleTaskGP",
    "MixedRobustRelevancePursuitMultiTaskGP",
    "MixedRobustRelevancePursuitSingleTaskGP",
    "MixedStudentTMultiTaskGP",
    "MixedStudentTSingleTaskGP",
    "NonstationaryKroneckerMultiTaskGP",
    "NonstationaryMultiTaskGP",
    "NonstationarySingleTaskGP",
    "ReplicateNoiseMultiFidelityGP",
    "ReplicateNoiseSingleTaskGP",
    "RobustRelevancePursuitMultiTaskGP",
    "RobustRelevancePursuitSingleTaskGP",
    "StudentTMultiTaskGP",
    "StudentTSingleTaskGP",
]
