"""Irregular as-of features, future targets and optional tabular input."""

from datetime import UTC, datetime

import pytest
import torch

from robotorchan.temporal import elapsed_lag_features, forecast_times, lagged_rows
from robotorchan.temporal.dataframe import dataframe_to_tensors


def test_asof_lags_use_only_available_history():
    times = torch.tensor([0.0, 2.0, 5.0, 9.0])
    values = times[:, None]
    x, rows = elapsed_lag_features(
        times,
        values,
        torch.tensor([4.0, 6.0, 10.0]),
        lags=[0.0, 3.0],
        available_times=torch.tensor([0.0, 2.0, 8.0, 9.0]),
    )
    assert rows.tolist() == [0, 1, 2]
    assert x.tolist() == [[2.0, 0.0], [2.0, 2.0], [9.0, 5.0]]


def test_series_staleness_and_exogenous():
    times = torch.tensor([0.0, 0.0, 3.0, 4.0])
    values = torch.tensor([[1.0], [10.0], [2.0], [20.0]])
    x, rows = elapsed_lag_features(
        times,
        values,
        torch.tensor([5.0, 5.0, 5.0]),
        lags=[0.0],
        series_ids=torch.tensor([0, 1, 0, 1]),
        decision_series_ids=torch.tensor([0, 1, 1]),
        max_age=1.0,
        exogenous=torch.ones(3, 1),
        exogenous_available_times=torch.tensor([5.0, 6.0, 4.0]),
    )
    assert rows.tolist() == [2]
    assert x.tolist() == [[20.0, 1.0]]


def test_step_lags_exogenous_availability():
    times = torch.arange(4, dtype=torch.float64)
    _, _, decisions, _ = lagged_rows(
        times,
        times[:, None],
        lags=[0],
        exogenous=times[:, None],
        exogenous_available_times=times + torch.tensor([0.0, 1.0, 0.0, 0.0]),
    )
    assert decisions.tolist() == [0, 2]


def test_future_targets_without_observations():
    times = torch.tensor([2.0, 5.0])
    assert forecast_times(times, [1.0, 4.0]).tolist() == [[3.0, 6.0], [6.0, 9.0]]
    with pytest.raises(ValueError, match="positive"):
        forecast_times(times, [0.0])


def test_dataframe():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(
        {"t": pd.to_datetime(["2026-01-01", "2026-01-02"], utc=True), "y": [1.0, None]}
    )
    times, values = dataframe_to_tensors(
        frame, time_column="t", value_columns=["y"], origin=datetime(2026, 1, 1, tzinfo=UTC)
    )
    assert times.tolist() == [0.0, 86400.0]
    assert torch.isnan(values[1, 0])
