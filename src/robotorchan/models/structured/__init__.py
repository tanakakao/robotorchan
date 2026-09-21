"""Structured-input, structured-output, and contextual surrogate models."""

from robotorchan.models.structured.additive import OrthogonalAdditiveGP
from robotorchan.models.structured.contextual import LCEAGP, SACGP
from robotorchan.models.structured.heterogeneous import HeterogeneousMTGP
from robotorchan.models.structured.hierarchical import (
    HierarchicalConditionalKernelGP,
    HierarchicalConditionalKernelMultiTaskGP,
)
from robotorchan.models.structured.higher_order import HigherOrderGP
from robotorchan.models.structured.latent_kronecker import LatentKroneckerGP

__all__ = [
    "LCEAGP",
    "SACGP",
    "HeterogeneousMTGP",
    "HierarchicalConditionalKernelGP",
    "HierarchicalConditionalKernelMultiTaskGP",
    "HigherOrderGP",
    "LatentKroneckerGP",
    "OrthogonalAdditiveGP",
]
