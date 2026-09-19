"""Dimensionality-reduced Gaussian-process models."""

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
    PCAKroneckerMultiTaskGP,
    PCAMultiTaskGP,
    RandomProjectionKroneckerMultiTaskGP,
    RandomProjectionMultiTaskGP,
    ReducedKroneckerMultiTaskGP,
    ReducedMultiTaskGP,
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

__all__ = [
    "PCAGP",
    "PCAKroneckerMultiTaskGP",
    "PCAMultiTaskGP",
    "PLSGP",
    "VAEGP",
    "AutoEncoderGP",
    "HybridAutoEncoderGP",
    "JointEncoderGP",
    "JointVAEGP",
    "MixedAutoEncoderGP",
    "MixedHybridAutoEncoderGP",
    "MixedJointEncoderGP",
    "MixedJointVAEGP",
    "MixedPCAGP",
    "MixedPLSGP",
    "MixedRandomProjectionGP",
    "MixedReducedGP",
    "MixedSupervisedAutoEncoderGP",
    "MixedSupervisedVAEGP",
    "MixedVAEGP",
    "OutputPCAGP",
    "OutputPLSGP",
    "RandomProjectionGP",
    "RandomProjectionKroneckerMultiTaskGP",
    "RandomProjectionMultiTaskGP",
    "ReducedKroneckerMultiTaskGP",
    "ReducedMultiTaskGP",
    "ReducedGP",
    "SupervisedAutoEncoderGP",
    "SupervisedVAEGP",
]
