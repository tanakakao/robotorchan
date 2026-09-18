"""Dimensionality-reduced Gaussian-process models."""

from robotorchan.models.reduced.base import (
    PCAGP,
    PLSGP,
    AutoEncoderGP,
    OutputPCAGP,
    OutputPLSGP,
    RandomProjectionGP,
    ReducedGP,
)
from robotorchan.models.reduced.joint_neural import HybridAutoEncoderGP, JointEncoderGP
from robotorchan.models.reduced.joint_vae import JointVAEGP
from robotorchan.models.reduced.mixed import (
    MixedAutoEncoderGP,
    MixedPCAGP,
    MixedPLSGP,
    MixedRandomProjectionGP,
    MixedReducedGP,
    MixedVAEGP,
)
from robotorchan.models.reduced.supervised_neural import SupervisedAutoEncoderGP
from robotorchan.models.reduced.vae import VAEGP, SupervisedVAEGP

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
    "OutputPCAGP",
    "OutputPLSGP",
    "RandomProjectionGP",
    "ReducedGP",
    "SupervisedAutoEncoderGP",
    "SupervisedVAEGP",
]
