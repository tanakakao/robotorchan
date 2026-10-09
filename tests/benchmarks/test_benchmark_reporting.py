"""Benchmark report serialization and metric aggregation tests."""

import csv
import io
import json

import pytest
import torch

from robotorchan.benchmarks.reporting import make_metric_report


def _curves() -> dict[int, torch.Tensor]:
    return {
        7: torch.tensor([2.0, 4.0], dtype=torch.double),
        3: torch.tensor([0.0, 2.0], dtype=torch.double),
    }


def test_normal_report_csv_and_json() -> None:
    report = make_metric_report("random", "regret", _curves())
    assert report.seeds == (3, 7)
    assert report.mean == (1.0, 3.0)
    assert report.interval_method == "normal"
    record = json.loads(report.to_json())
    assert record["schema_version"] == 1
    assert record["seeds"] == [3, 7]
    rows = list(csv.DictReader(io.StringIO(report.to_csv())))
    assert len(rows) == 2
    assert rows[0]["evaluation_index"] == "0"
    assert float(rows[1]["mean"]) == 3.0


def test_bootstrap_report_reproducible() -> None:
    first = make_metric_report(
        "sobol",
        "hypervolume",
        _curves(),
        interval_method="bootstrap",
        n_resamples=100,
        seed=11,
    )
    second = make_metric_report(
        "sobol",
        "hypervolume",
        _curves(),
        interval_method="bootstrap",
        n_resamples=100,
        seed=11,
    )
    assert first.to_json() == second.to_json()
    assert first.lower[0] <= first.mean[0] <= first.upper[0]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"strategy": ""},
        {"metric": ""},
        {"interval_method": "unknown"},
        {"confidence": 1.0},
    ],
)
def test_invalid_report_settings(kwargs: dict) -> None:
    parameters = {"strategy": "random", "metric": "regret", "curves": _curves()}
    parameters.update(kwargs)
    with pytest.raises(ValueError):
        make_metric_report(**parameters)


def test_nonfinite_curves_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        make_metric_report("random", "regret", {0: torch.tensor([float("nan")])})
