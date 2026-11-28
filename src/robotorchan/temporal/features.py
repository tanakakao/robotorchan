"""Leakage-aware feature construction for ordered and irregular observations."""

from collections.abc import Sequence

import torch
from torch import Tensor

from robotorchan.temporal.axis import validate_times


def observed_rows(values: Tensor) -> Tensor:
    """Return a row mask for complete, finite observations without imputation."""
    if values.ndim != 2:
        raise ValueError("values must have shape (n, d)")
    return torch.isfinite(values).all(dim=-1)


def lagged_rows(
    times: Tensor,
    values: Tensor,
    *,
    lags: Sequence[int],
    horizon: int = 1,
    series_ids: Tensor | None = None,
    available_times: Tensor | None = None,
    exogenous: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return features, targets, decision row indices and target row indices.

    Lags and horizon count observations within a series, not elapsed time.
    Lag zero means the decision row. Exogenous values are taken only at that
    row and must be known at decision time. Rows with missing inputs or targets
    are omitted; no interpolation or future backfill is performed.
    """
    validate_times(times)
    n = times.numel()
    if values.ndim != 2 or values.shape[0] != n or values.device != times.device:
        raise ValueError("values must have shape (n, d) on the time device")
    if not lags or any(type(lag) is not int or lag < 0 for lag in lags):
        raise ValueError("lags must be nonnegative integers")
    if type(horizon) is not int or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    if len(set(lags)) != len(lags):
        raise ValueError("lags must be unique")
    ids = torch.zeros(n, dtype=torch.long, device=times.device) if series_ids is None else series_ids
    if ids.shape != (n,) or ids.device != times.device:
        raise ValueError("series_ids must align with times")
    if ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError("series_ids must be integer identifiers")
    available = times if available_times is None else available_times
    validate_times(available)
    if available.shape != times.shape or available.device != times.device:
        raise ValueError("available_times must align with times")
    if (available < times).any():
        raise ValueError("availability cannot precede observation")
    if exogenous is not None and (
        exogenous.ndim != 2
        or exogenous.shape[0] != n
        or exogenous.device != times.device
        or exogenous.dtype != values.dtype
    ):
        raise ValueError("exogenous must align with values in rows, device and dtype")
    features, targets, decisions, target_indices = [], [], [], []
    for group in ids.unique():
        idx = (ids == group).nonzero(as_tuple=True)[0]
        t = times[idx]
        if (t[1:] <= t[:-1]).any():
            raise ValueError("times must be strictly increasing within each series")
        for k in range(max(lags), len(idx) - horizon):
            decision = idx[k]
            target = idx[k + horizon]
            history = idx[k - torch.tensor(lags, device=times.device)]
            if (available[history] > times[decision]).any():
                continue
            x = values[history].flatten()
            if exogenous is not None:
                x = torch.cat((x, exogenous[decision]))
            y = values[target]
            if torch.isfinite(x).all() and torch.isfinite(y).all():
                features.append(x)
                targets.append(y)
                decisions.append(decision)
                target_indices.append(target)
    if not features:
        width = len(lags) * values.shape[1]
        width += 0 if exogenous is None else exogenous.shape[1]
        empty_idx = torch.empty(0, dtype=torch.long, device=times.device)
        return (
            values.new_empty((0, width)),
            values.new_empty((0, values.shape[1])),
            empty_idx,
            empty_idx.clone(),
        )
    decision_rows = torch.stack(decisions)
    order = torch.argsort(times[decision_rows], stable=True)
    return (
        torch.stack(features)[order],
        torch.stack(targets)[order],
        decision_rows[order],
        torch.stack(target_indices)[order],
    )
