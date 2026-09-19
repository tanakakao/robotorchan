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
from robotorchan.models.reduced.joint_neural import HybridAutoEncoderGP, JointEncoderGP
from robotorchan.models.reduced.joint_vae import JointVAEGP
from robotorchan.models.reduced.mixed import MixedReducedGP
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
    "PLSGP",
    "VAEGP",
    "AutoEncoderGP",
    "HybridAutoEncoderGP",
    "JointEncoderGP",
    "JointVAEGP",
    "MixedAutoEncoderGP",
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
    "ReducedGP",
    "SupervisedAutoEncoderGP",
    "SupervisedVAEGP",
]
