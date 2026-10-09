"""Portable benchmark reports built from validated seeded metric curves."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from typing import Any

from torch import Tensor

from robotorchan.benchmarks.comparison import summarize_curves
from robotorchan.benchmarks.multiseed_statistics import bootstrap_curves


@dataclass(frozen=True)
class BenchmarkMetricReport:
    """Evaluation-aligned mean and uncertainty for one strategy and metric."""

    strategy: str
    metric: str
    seeds: tuple[int, ...]
    mean: tuple[float, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    confidence: float
    interval_method: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible, schema-labeled report."""
        return {
            "schema_version": 1,
            "strategy": self.strategy,
            "metric": self.metric,
            "seeds": list(self.seeds),
            "mean": list(self.mean),
            "lower": list(self.lower),
            "upper": list(self.upper),
            "confidence": self.confidence,
            "interval_method": self.interval_method,
        }

    def to_json(self) -> str:
        """Serialize a deterministic JSON report."""
        return json.dumps(self.to_dict(), indent=2, allow_nan=False)

    def to_csv(self) -> str:
        """Serialize one row per zero-based curve index."""
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(("strategy", "metric", "evaluation_index", "mean", "lower", "upper"))
        for index, (mean, lower, upper) in enumerate(
            zip(self.mean, self.lower, self.upper, strict=True)
        ):
            writer.writerow((self.strategy, self.metric, index, mean, lower, upper))
        return buffer.getvalue()


def make_metric_report(
    strategy: str,
    metric: str,
    curves: dict[int, Tensor],
    *,
    confidence: float = 0.95,
    interval_method: str = "normal",
    n_resamples: int = 2000,
    seed: int = 0,
) -> BenchmarkMetricReport:
    """Aggregate seed curves for plotting, CSV export and JSON archiving.

    Curve index zero is the first element of the supplied metric series,
    not necessarily the first newly acquired evaluation.
    """
    if not isinstance(strategy, str) or not strategy.strip():
        raise ValueError("strategy must be a nonempty string.")
    if not isinstance(metric, str) or not metric.strip():
        raise ValueError("metric must be a nonempty string.")
    if interval_method == "normal":
        summary = summarize_curves(curves, confidence=confidence)
    elif interval_method == "bootstrap":
        summary = bootstrap_curves(
            curves,
            confidence=confidence,
            n_resamples=n_resamples,
            seed=seed,
        )
    else:
        raise ValueError("interval_method must be normal or bootstrap.")

    def values(tensor: Tensor) -> tuple[float, ...]:
        return tuple(float(value) for value in tensor.detach().cpu().tolist())

    return BenchmarkMetricReport(
        strategy=strategy,
        metric=metric,
        seeds=summary.seeds,
        mean=values(summary.mean),
        lower=values(summary.lower),
        upper=values(summary.upper),
        confidence=confidence,
        interval_method=interval_method,
    )
