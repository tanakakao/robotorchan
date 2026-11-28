"""As-of lag lookup and target-free forecast features for irregular time series."""

import math
from collections.abc import Sequence

import torch
from torch import Tensor

from robotorchan.temporal.axis import validate_times


def elapsed_lag_features(
    times: Tensor,
    values: Tensor,
    decision_times: Tensor,
    *,
    lags: Sequence[float],
    series_ids: Tensor | None = None,
    decision_series_ids: Tensor | None = None,
    available_times: Tensor | None = None,
    max_age: float | None = None,
    exogenous: Tensor | None = None,
    exogenous_available_times: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Return valid feature rows and their indices in decision_times.

    Each lag uses the latest finite observation at or before decision minus
    lag, available by decision time. max_age bounds additional staleness
    relative to the requested lag time. No target data are required.
    Exogenous rows correspond to decisions, and must be known by then.
    """
    validate_times(times)
    validate_times(decision_times)
    if times.device != decision_times.device or times.dtype != decision_times.dtype:
        raise ValueError("decision times must share time dtype and device")
    if values.ndim != 2 or values.shape[0] != len(times) or values.device != times.device:
        raise ValueError("values must align with observation times")
    if not lags or any(not math.isfinite(lag) or lag < 0 for lag in lags):
        raise ValueError("lags must be finite and nonnegative")
    if len(set(lags)) != len(lags):
        raise ValueError("lags must be unique")
    if max_age is not None and (not math.isfinite(max_age) or max_age < 0):
        raise ValueError("max_age must be finite and nonnegative")
    if (series_ids is None) != (decision_series_ids is None):
        raise ValueError("supply both observation and decision series identifiers")
    ids = _series_ids(times, series_ids)
    query_ids = _series_ids(decision_times, decision_series_ids)
    available = times if available_times is None else available_times
    validate_times(available)
    if available.shape != times.shape or available.device != times.device:
        raise ValueError("availability must align with observation times")
    if (available < times).any():
        raise ValueError("availability cannot precede observation")
    if exogenous is not None and (
        exogenous.ndim != 2
        or exogenous.shape[0] != len(decision_times)
        or exogenous.device != values.device
        or exogenous.dtype != values.dtype
    ):
        raise ValueError("exogenous must align with decisions and value dtype/device")
    if exogenous_available_times is not None:
        if exogenous is None:
            raise ValueError("exogenous is required for exogenous availability")
        validate_times(exogenous_available_times)
        if (
            exogenous_available_times.shape != decision_times.shape
            or exogenous_available_times.device != times.device
        ):
            raise ValueError("exogenous availability must align with decisions")
    finite = torch.isfinite(values).all(dim=-1)
    rows, indices = [], []
    for i, decision in enumerate(decision_times):
        eligible = (ids == query_ids[i]) & (available <= decision) & finite
        parts = []
        for lag in lags:
            requested = decision - lag
            candidates = eligible & (times <= requested)
            if max_age is not None:
                candidates = candidates & (times >= requested - max_age)
            idx = candidates.nonzero(as_tuple=True)[0]
            if not idx.numel():
                break
            latest = idx[times[idx].argmax()]
            parts.append(values[latest])
        if len(parts) != len(lags):
            continue
        if exogenous is not None:
            if not torch.isfinite(exogenous[i]).all():
                continue
            if exogenous_available_times is not None and exogenous_available_times[i] > decision:
                continue
            parts.append(exogenous[i])
        rows.append(torch.cat(parts))
        indices.append(i)
    width = len(lags) * values.shape[1]
    width += 0 if exogenous is None else exogenous.shape[1]
    x = torch.stack(rows) if rows else values.new_empty((0, width))
    return x, torch.tensor(indices, dtype=torch.long, device=times.device)


def _series_ids(times: Tensor, ids: Tensor | None) -> Tensor:
    if ids is None:
        return torch.zeros(len(times), dtype=torch.long, device=times.device)
    if ids.shape != times.shape or ids.device != times.device:
        raise ValueError("series identifiers must align with time coordinates")
    if ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError("series identifiers must be integers")
    return ids


def forecast_times(origins: Tensor, horizons: Sequence[float]) -> Tensor:
    """Return (n, h) target times for positive physical-duration horizons."""
    validate_times(origins)
    if not horizons or any(not math.isfinite(h) or h <= 0 for h in horizons):
        raise ValueError("horizons must be finite and positive")
    targets = origins[:, None] + origins.new_tensor(horizons)
    if not torch.isfinite(targets).all() or (targets <= origins[:, None]).any():
        raise ValueError("horizons exceed time precision or range")
    return targets
