"""Phase 19 controlled sensitivity sweeps and checkpoint-aligned summaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal

import torch
from torch import Tensor

from robotorchan.benchmarks.config import BenchmarkExperimentConfig

SensitivityAxis = Literal["initial_points", "q", "evaluation_budget"]


@dataclass(frozen=True)
class SensitivitySummary:
    """Seed-paired metric means at evaluation counts shared by all settings."""

    axis: SensitivityAxis
    values: tuple[int, ...]
    seeds: tuple[int, ...]
    evaluations: Tensor
    metric: str
    mean_by_value: dict[int, Tensor]
    standard_error_by_value: dict[int, Tensor]


def sensitivity_configs(
    base: BenchmarkExperimentConfig,
    *,
    axis: SensitivityAxis,
    values: Sequence[int],
) -> tuple[BenchmarkExperimentConfig, ...]:
    """Create a one-factor-at-a-time sweep while preserving other settings."""
    if axis not in ("initial_points", "q", "evaluation_budget"):
        raise ValueError("Unsupported sensitivity axis.")
    if not values or any(type(value) is not int or value < 1 for value in values):
        raise ValueError("Sensitivity values must be positive integers.")
    if len(set(values)) != len(values):
        raise ValueError("Sensitivity values must be unique.")
    return tuple(replace(base, **{axis: value}) for value in values)


def summarize_sensitivity(
    *,
    axis: SensitivityAxis,
    scores: Mapping[int, Tensor],
    evaluations: Mapping[int, Tensor],
    seeds: Mapping[int, tuple[int, ...]],
    metric: str,
) -> SensitivitySummary:
    """Align completed-evaluation checkpoints without interpolating scores.

    Scores have shape (seed, checkpoint). Different budgets are compared only
    at checkpoints present in every setting. Seed ordering must be identical.
    For initial-design sweeps, seed identity does not imply paired initial X.
    """
    if axis not in ("initial_points", "q", "evaluation_budget"):
        raise ValueError("Unsupported sensitivity axis.")
    if metric not in (
        "simple_regret",
        "regret",
        "feasible_regret",
        "hypervolume",
        "feasible_hypervolume",
    ):
        raise ValueError("Unsupported sensitivity metric.")
    if not scores or set(scores) != set(evaluations) or set(scores) != set(seeds):
        raise ValueError("Scores, checkpoints, and seeds must cover identical settings.")
    values = tuple(sorted(scores))
    if any(type(value) is not int or value < 1 for value in values):
        raise ValueError("Sensitivity setting values must be positive integers.")
    reference_seeds = seeds[values[0]]
    if not reference_seeds or len(set(reference_seeds)) != len(reference_seeds):
        raise ValueError("Seeds must be nonempty and unique.")
    common: set[int] | None = None
    for value in values:
        points = evaluations[value]
        data = scores[value]
        if (
            not isinstance(points, Tensor)
            or points.ndim != 1
            or points.dtype not in (torch.int32, torch.int64)
            or points.numel() == 0
            or points[0].item() != 0
            or (points[1:] <= points[:-1]).any()
        ):
            raise ValueError("Checkpoints must start at zero and strictly increase.")
        if (
            not isinstance(data, Tensor)
            or data.shape != (len(reference_seeds), points.numel())
            or not torch.is_floating_point(data)
            or torch.isnan(data).any()
            or torch.isneginf(data).any()
            or (metric != "feasible_regret" and torch.isposinf(data).any())
        ):
            raise ValueError("Scores must be finite floating-point [seed, checkpoint] tensors.")
        if seeds[value] != reference_seeds:
            raise ValueError("Seed ordering must match across sensitivity settings.")
        point_set = set(points.tolist())
        common = point_set if common is None else common & point_set
    assert common is not None
    if not common:
        raise ValueError("No common evaluation checkpoints.")
    aligned = torch.tensor(sorted(common), dtype=torch.long)
    mean = {}
    stderr = {}
    for value in values:
        indices = [evaluations[value].tolist().index(point) for point in aligned.tolist()]
        selected = scores[value][:, indices]
        mean[value] = selected.mean(dim=0)
        stderr[value] = (
            selected.std(dim=0, unbiased=True) / len(reference_seeds) ** 0.5
            if len(reference_seeds) > 1
            else torch.zeros_like(selected[0])
        )
    return SensitivitySummary(
        axis=axis,
        values=values,
        seeds=reference_seeds,
        evaluations=aligned,
        metric=metric,
        mean_by_value=mean,
        standard_error_by_value=stderr,
    )
