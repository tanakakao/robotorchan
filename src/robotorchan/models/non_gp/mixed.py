"""Mixed-variable helpers for sklearn-backed tree surrogates."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor


def normalize_cat_dims(cat_dims: Sequence[int], input_dim: int) -> tuple[int, ...]:
    """Normalize categorical dimensions and reject duplicates/out-of-range indices."""
    normalized = tuple(dim % input_dim if dim < 0 else dim for dim in cat_dims)
    if any(dim < 0 or dim >= input_dim for dim in normalized):
        raise ValueError("cat_dims contains an index outside the input dimensions.")
    if len(set(normalized)) != len(normalized):
        raise ValueError("cat_dims must not contain duplicate dimensions.")
    return normalized


def validate_categorical_values(X: Tensor, cat_dims: tuple[int, ...]) -> None:
    """Require categorical coordinates to use integer-valued numeric labels."""
    if not cat_dims:
        return
    categorical = X[..., list(cat_dims)]
    if not torch.isfinite(categorical).all():
        raise ValueError("Categorical values must be finite.")
    if not torch.equal(categorical, categorical.round()):
        raise ValueError("Categorical values must use integer-valued labels.")
