"""Tensor utilities for independently observed model outputs."""

import torch
from torch import Tensor


def split_partial_observations(
    X: Tensor,
    Y: Tensor,
    evaluation_mask: Tensor,
    Yvar: Tensor | None = None,
) -> tuple[tuple[Tensor, Tensor, Tensor | None], ...]:
    """Return (X, Y, Yvar) for each output, preserving order and dtype.

    Unobserved entries are ignored, including NaN placeholders. An output
    with no observations returns empty tensors; fitting it is caller-owned.
    """
    if X.ndim != 2 or Y.ndim != 2 or X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must be aligned two-dimensional tensors")
    if Y.shape[1] == 0:
        raise ValueError("Y must contain at least one output")
    if evaluation_mask.shape != Y.shape or evaluation_mask.dtype != torch.bool:
        raise ValueError("evaluation_mask must be boolean with the same shape as Y")
    if X.device != Y.device or evaluation_mask.device != Y.device:
        raise ValueError("X, Y and evaluation_mask must share a device")
    if not X.is_floating_point() or not torch.isfinite(X).all():
        raise ValueError("X must contain finite floating-point values")
    if not torch.isfinite(Y[evaluation_mask]).all():
        raise ValueError("observed Y values must be finite")
    if Yvar is not None:
        if Yvar.shape != Y.shape or Yvar.device != Y.device:
            raise ValueError("Yvar must match Y shape and device")
        observed_var = Yvar[evaluation_mask]
        if (
            not Yvar.is_floating_point()
            or not torch.isfinite(observed_var).all()
            or (observed_var < 0).any()
        ):
            raise ValueError("observed variances must be finite and nonnegative")
    return tuple(
        (
            X[evaluation_mask[:, j]],
            Y[evaluation_mask[:, j], j : j + 1],
            None if Yvar is None else Yvar[evaluation_mask[:, j], j : j + 1],
        )
        for j in range(Y.shape[1])
    )


def append_partial_observations(
    observations: tuple[tuple[Tensor, Tensor, Tensor | None], ...],
    X: Tensor,
    Y: Tensor,
    evaluation_mask: Tensor,
    Yvar: Tensor | None = None,
) -> tuple[tuple[Tensor, Tensor, Tensor | None], ...]:
    """Append only observed rows to existing output-specific datasets."""
    updates = split_partial_observations(X, Y, evaluation_mask, Yvar)
    if len(observations) != len(updates):
        raise ValueError("existing observations must match the number of outputs")
    result = []
    for (old_x, old_y, old_var), (new_x, new_y, new_var) in zip(observations, updates, strict=True):
        if new_x.shape[0] == 0:
            result.append((old_x, old_y, old_var))
            continue
        if (old_var is None) != (new_var is None):
            raise ValueError("cannot mix known and unspecified observation variances")
        pairs = [(old_x, new_x), (old_y, new_y)]
        if old_var is not None and new_var is not None:
            pairs.append((old_var, new_var))
        if any(a.dtype != b.dtype or a.device != b.device for a, b in pairs):
            raise ValueError("existing and new observations must share dtype and device")
        result.append(
            (
                torch.cat((old_x, new_x)),
                torch.cat((old_y, new_y)),
                None if old_var is None or new_var is None else torch.cat((old_var, new_var)),
            )
        )
    return tuple(result)
