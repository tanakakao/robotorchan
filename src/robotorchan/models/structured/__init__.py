"""Structured-input, structured-output, and contextual surrogate models."""

from robotorchan.models.structured.additive import MixedOrthogonalAdditiveGP, OrthogonalAdditiveGP
from robotorchan.models.structured.contextual import LCEAGP, LCEMGP, MixedLCEMGP, SACGP
from robotorchan.models.structured.heterogeneous import HeterogeneousMTGP, MixedHeterogeneousMTGP
from robotorchan.models.structured.hierarchical import (
    HierarchicalConditionalKernelGP,
    HierarchicalConditionalKernelMultiTaskGP,
    MixedHierarchicalConditionalKernelGP,
    MixedHierarchicalConditionalKernelMultiTaskGP,
)
from robotorchan.models.structured.higher_order import HigherOrderGP, MixedHigherOrderGP
from robotorchan.models.structured.latent_kronecker import LatentKroneckerGP, MixedLatentKroneckerGP

__all__ = [
    "HeterogeneousMTGP",
    "HierarchicalConditionalKernelGP",
    "HierarchicalConditionalKernelMultiTaskGP",
    "HigherOrderGP",
    "LCEAGP",
    "LCEMGP",
    "LatentKroneckerGP",
    "MixedHeterogeneousMTGP",
    "MixedHierarchicalConditionalKernelGP",
    "MixedHierarchicalConditionalKernelMultiTaskGP",
    "MixedHigherOrderGP",
    "MixedLCEMGP",
    "MixedLatentKroneckerGP",
    "MixedOrthogonalAdditiveGP",
    "OrthogonalAdditiveGP",
    "SACGP",
]
