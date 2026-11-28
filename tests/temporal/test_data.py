"""Temporal data contracts and leakage regression tests."""

from datetime import UTC, datetime, timedelta

import pytest
import torch

from robotorchan.temporal import (
    TimeAxis,
    datetime_seconds,
    lagged_rows,
    observed_rows,
    rolling_origin_splits,
)


def test_axis_training_only():
    t = torch.tensor([10.0, 12.0, 17.0], dtype=torch.float64)
    axis = TimeAxis.fit(t[:2])
    torch.testing.assert_close(axis.transform(t), torch.tensor([0.0, 1.0, 3.5]).double())
    torch.testing.assert_close(axis.inverse_transform(axis.transform(t)), t)
    assert TimeAxis.fit(t[:1]).scale.item() == 1


def test_datetime_conversion():
    origin = datetime(2026, 1, 1, tzinfo=UTC)
    assert datetime_seconds([origin + timedelta(seconds=3)], origin=origin).item() == 3
    with pytest.raises(ValueError, match="timezone"):
        datetime_seconds([datetime(2026, 1, 1)], origin=origin)


def test_irregular_multiple_series():
    t = torch.tensor([0.0, 0.0, 2.0, 3.0, 5.0, 9.0])
    ids = torch.tensor([0, 1, 0, 1, 0, 1])
    y = torch.tensor([[1.0], [10.0], [2.0], [20.0], [3.0], [30.0]])
    x, target, decision, rows = lagged_rows(t, y, lags=[0, 1], series_ids=ids)
    torch.testing.assert_close(x, torch.tensor([[2.0, 1.0], [20.0, 10.0]]))
    torch.testing.assert_close(target, torch.tensor([[3.0], [30.0]]))
    assert decision.tolist() == [2, 3]
    assert rows.tolist() == [4, 5]


def test_missing_delayed_and_exogenous():
    t = torch.arange(6, dtype=torch.double)
    y = t[:, None].clone()
    y[1] = torch.nan
    available = t.clone()
    available[3] = 5
    x, target, decision, _ = lagged_rows(
        t, y, lags=[0], available_times=available, exogenous=y
    )
    assert observed_rows(y).tolist() == [True, False, True, True, True, True]
    assert decision.tolist() == [2, 4]
    assert x.tolist() == [[2.0, 2.0], [4.0, 4.0]]
    assert target.tolist() == [[3.0], [5.0]]


def test_split_availability_and_gap():
    t = torch.arange(10, dtype=torch.double)
    target = t + 1
    available = target + 1
    folds = list(
        rolling_origin_splits(
            t, target, origins=[5.0, 7.0], test_duration=2.0, label_available_times=available
        )
    )
    assert folds[0][0].tolist() == [0, 1, 2]
    assert folds[0][1].tolist() == [5, 6]
    train, _ = next(
        rolling_origin_splits(
            t, target, origins=[7.0], test_duration=2.0, gap=1.0, train_duration=4.0
        )
    )
    assert train.tolist() == [2, 3, 4]


def test_reject_duplicate_times():
    with pytest.raises(ValueError, match="strictly increasing"):
        lagged_rows(torch.tensor([0.0, 0.0]), torch.ones(2, 1), lags=[0])


def test_empty_lag_shape():
    x, y, d, r = lagged_rows(torch.tensor([0.0]), torch.ones(1, 2), lags=[0, 1])
    assert x.shape == (0, 4)
    assert y.shape == (0, 2)
    assert d.numel() == r.numel() == 0
