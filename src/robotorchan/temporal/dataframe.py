"""Optional pandas conversion without importing pandas at package import time."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from robotorchan.temporal.axis import datetime_seconds, validate_times

if TYPE_CHECKING:
    from pandas import DataFrame


def dataframe_to_tensors(
    frame: DataFrame,
    *,
    time_column: str,
    value_columns: Sequence[str],
    origin: datetime | None = None,
    dtype: torch.dtype = torch.float64,
) -> tuple[Tensor, Tensor]:
    """Convert numeric or timezone-aware datetime columns, retaining row order.

    Datetimes require an explicit origin; numeric times retain their units.
    Missing numeric values become NaN. Missing time coordinates are rejected.
    """
    import pandas as pd

    if not value_columns:
        raise ValueError("value_columns must be nonempty")
    column = frame[time_column]
    if pd.api.types.is_datetime64_any_dtype(column.dtype):
        if origin is None or column.isna().any():
            raise ValueError("datetime times require an origin and no missing values")
        times = datetime_seconds(column.tolist(), origin=origin).to(dtype=dtype)
    else:
        if origin is not None:
            raise ValueError("origin is only supported for datetime columns")
        times = torch.tensor(column.to_numpy(dtype=float, na_value=float("nan")), dtype=dtype)
    validate_times(times)
    values = torch.tensor(
        frame[list(value_columns)].to_numpy(dtype=float, na_value=float("nan")), dtype=dtype
    )
    return times, values
