"""Posterior-sampling helpers for discrete candidate sets."""

from __future__ import annotations

import torch
from botorch.acquisition.objective import MCAcquisitionObjective
from botorch.generation.sampling import MaxPosteriorSampling
from botorch.models.model import Model
from botorch.utils.transforms import is_ensemble
from torch import Tensor


def select_thompson_candidates(
    model: Model,
    choices: Tensor,
    num_samples: int = 1,
    *,
    replacement: bool = False,
    objective: MCAcquisitionObjective | None = None,
) -> Tensor:
    """Select candidates from a finite set by Thompson-style posterior sampling.

    Args:
        model: BoTorch-compatible posterior model.
        choices: Candidate tensor with shape n x d.
        num_samples: Number of candidates to select.
        replacement: Whether the same candidate may be selected more than once.
        objective: Optional objective used to scalarize multi-output posterior samples.

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

    if model.num_outputs != 1 and objective is None:
        raise ValueError("objective is required for multi-output posterior sampling.")

    sampler = MaxPosteriorSampling(
        model=model,
        objective=objective,
        replacement=replacement,
    )
    with torch.no_grad():
        selected = sampler(choices, num_samples=num_samples)

    if is_ensemble(model) and selected.ndim > 2:
        ensemble_shape = selected.shape[:-2]
        if len(ensemble_shape) != 1:
            raise ValueError(
                "ensemble Thompson sampling requires exactly one ensemble batch dimension."
            )
        member = torch.randint(ensemble_shape[0], (), device=selected.device)
        selected = selected[member]

    return selected
