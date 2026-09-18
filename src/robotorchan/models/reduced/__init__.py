"""Dimensionality-reduced Gaussian-process models."""

from robotorchan.models.reduced.base import AutoEncoderGP, OutputPCAGP, OutputPLSGP, PCAGP, PLSGP, RandomProjectionGP, ReducedGP
from robotorchan.models.reduced.joint_neural import HybridAutoEncoderGP, JointEncoderGP
from robotorchan.models.reduced.joint_vae import JointVAEGP
from robotorchan.models.reduced.supervised_neural import SupervisedAutoEncoderGP
from robotorchan.models.reduced.vae import SupervisedVAEGP, VAEGP

__all__ = ["AutoEncoderGP", "HybridAutoEncoderGP", "JointEncoderGP", "JointVAEGP", "OutputPCAGP", "OutputPLSGP", "PCAGP", "PLSGP", "RandomProjectionGP", "ReducedGP", "SupervisedAutoEncoderGP", "SupervisedVAEGP", "VAEGP"]
