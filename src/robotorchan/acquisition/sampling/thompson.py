"""Posterior-sampling helpers for discrete candidate sets."""

from __future__ import annotations

import torch
from botorch.generation.sampling import MaxPosteriorSampling
from botorch.models.model import Model
from torch import Tensor


def select_thompson_candidates(
    model: Model,
    choices: Tensor,
    num_samples: int = 1,
    *,
    replacement: bool = False,
) -> Tensor:
    """Select candidates from a finite set by Thompson-style posterior sampling.

    Args:
        model: BoTorch-compatible posterior model.
        choices: Candidate tensor with shape n x d.
        num_samples: Number of candidates to select.
        replacement: Whether the same candidate may be selected more than once.

    Returns:
        Selected candidates with shape num_samples x d.

    Raises:
        ValueError: If the candidate tensor or sampling request is invalid.
    """
    if choices.ndim != 2:
        raise ValueError("choices must have shape n x d.")
    if choices.shape[0] == 0:
        raise ValueError("choices must contain at least one candidate.")
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1.")
    if not replacement and num_samples > choices.shape[0]:
        raise ValueError("num_samples cannot exceed the number of choices without replacement.")

    sampler = MaxPosteriorSampling(model=model, replacement=replacement)
    with torch.no_grad():
        return sampler(choices, num_samples=num_samples)
