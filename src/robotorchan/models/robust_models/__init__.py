"""Robust, noisy-observation, and nonstationary surrogate models."""

from robotorchan.models.robust_models.contaminated import (
    ContaminatedMultiTaskGP,
    ContaminatedSingleTaskGP,
    MixedContaminatedMultiTaskGP,
    MixedContaminatedSingleTaskGP,
)
from robotorchan.models.robust_models.joint_heteroskedastic import (
    JointHeteroskedasticSingleTaskGP,
    MixedJointHeteroskedasticSingleTaskGP,
)
from robotorchan.models.robust_models.nonstationary import (
    MixedNonstationaryMultiTaskGP,
    MixedNonstationarySingleTaskGP,
    NonstationaryMultiTaskGP,
    NonstationarySingleTaskGP,
)
from robotorchan.models.robust_models.replicate_noise import (
    MixedReplicateNoiseSingleTaskGP,
    ReplicateNoiseSingleTaskGP,
)
from robotorchan.models.robust_models.robust import (
    HeteroskedasticMultiTaskGP,
    HeteroskedasticSingleTaskGP,
    MixedHeteroskedasticMultiTaskGP,
    MixedHeteroskedasticSingleTaskGP,
    MixedRobustRelevancePursuitMultiTaskGP,
    MixedRobustRelevancePursuitSingleTaskGP,
    RobustRelevancePursuitMultiTaskGP,
    RobustRelevancePursuitSingleTaskGP,
)
from robotorchan.models.robust_models.student_t import (
    MixedStudentTMultiTaskGP,
    MixedStudentTSingleTaskGP,
    StudentTMultiTaskGP,
    StudentTSingleTaskGP,
)

__all__ = [
    "ContaminatedMultiTaskGP",
    "HeteroskedasticMultiTaskGP",
    "JointHeteroskedasticSingleTaskGP",
    "MixedContaminatedMultiTaskGP",
    "MixedHeteroskedasticMultiTaskGP",
    "MixedNonstationaryMultiTaskGP",
    "MixedReplicateNoiseSingleTaskGP",
    "MixedRobustRelevancePursuitMultiTaskGP",
    "MixedStudentTMultiTaskGP",
    "NonstationaryMultiTaskGP",
    "RobustRelevancePursuitMultiTaskGP",
    "StudentTMultiTaskGP",
]
