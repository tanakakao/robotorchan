"""Mixed-variable helpers for sklearn-backed tree surrogates."""

from __future__ import annotations

import torch
from torch import Tensor


def validate_categorical_values(X: Tensor, cat_dims: tuple[int, ...]) -> None:
    """Require categorical coordinates to use integer-valued numeric labels."""
    if not cat_dims:
        return
    categorical = X[..., list(cat_dims)]
    if not torch.isfinite(categorical).all():
        raise ValueError("Categorical values must be finite.")
    if not torch.equal(categorical, categorical.round()):
        raise ValueError("Categorical values must use integer-valued labels.")
