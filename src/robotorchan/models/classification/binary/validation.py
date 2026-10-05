"""Label validation for binary classification models."""

import torch
from torch import Tensor


def validate_binary_labels(train_Y: Tensor) -> Tensor:
    """Validate canonical binary labels and return them unchanged."""
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
