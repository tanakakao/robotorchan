"""Posterior helpers shared by empirical non-GP surrogate ensembles."""

from __future__ import annotations

import torch
from botorch.posteriors.ensemble import EnsemblePosterior
from torch import Tensor


def make_ensemble_posterior(values: Tensor) -> EnsemblePosterior:
    """Build a BoTorch ensemble posterior from predictive function samples.

    Args:
        values: Predictive values with shape ``... x s x q x m``, where ``s``
            is the ensemble dimension, ``q`` is the candidate dimension, and
            ``m`` is the output dimension.

    Returns:
        A BoTorch ``EnsemblePosterior`` preserving the input dtype, device, and
        autograd graph.

    Raises:
        ValueError: If the tensor cannot represent ``s x q x m`` values or the
            ensemble dimension is empty.
    """
    if values.ndim < 3:
        raise ValueError("Ensemble values must have shape ... x s x q x m.")
    if values.shape[-3] < 1:
        raise ValueError("Ensemble values must contain at least one member.")
    if not torch.is_floating_point(values):
        raise ValueError("Ensemble values must use a floating-point dtype.")
    return EnsemblePosterior(values=values)
