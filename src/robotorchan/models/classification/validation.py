"""Label validation for classification surrogate models."""

from __future__ import annotations

import torch
from torch import Tensor


def validate_binary_labels(train_Y: Tensor) -> Tensor:
    """Validate canonical binary labels and return them unchanged.

    The tensor-facing classification API accepts only numeric or boolean labels
    representing 0 and 1. Label encoding belongs to higher-level tabular APIs.

    Args:
        train_Y: Binary training labels.

    Returns:
        The original ``train_Y`` tensor.

    Raises:
        TypeError: If ``train_Y`` is not a tensor or has a complex dtype.
        ValueError: If labels are empty, non-finite, or not binary 0/1 values.
    """
    if not isinstance(train_Y, Tensor):
        raise TypeError("train_Y must be a torch.Tensor.")
    if train_Y.ndim == 0:
        raise ValueError("train_Y must include an observation dimension.")
    if train_Y.numel() == 0 or train_Y.shape[0] == 0:
        raise ValueError("train_Y must contain at least one label.")
    if train_Y.is_complex():
        raise TypeError("train_Y must use a real-valued or boolean dtype.")
    if train_Y.is_floating_point() and not torch.isfinite(train_Y).all():
        raise ValueError("train_Y must contain only finite labels.")

    valid = (train_Y == 0) | (train_Y == 1)
    if not bool(valid.all()):
        raise ValueError("Binary train_Y labels must contain only 0 and 1.")
    return train_Y
