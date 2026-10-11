"""Exact paired sign tests and Holm correction for benchmark comparisons."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import comb, isfinite

import torch

from robotorchan.benchmarks.curve_statistics import paired_metric_difference
from robotorchan.benchmarks.metric_curves import MetricCurves


@dataclass(frozen=True)
class PairedInference:
    """Final-evaluation paired inference against one reference strategy."""

    strategy: str
    reference: str
    n_pairs: int
    n_non_ties: int
    mean_difference: float
    standardized_mean_difference: float | None
    p_value: float
    adjusted_p_value: float
    significant: bool
    lower_is_better: bool


def _sign_test_p_value(positive: int, negative: int) -> float:
    """Exact two-sided sign-test probability conditional on nonzero differences."""
    n = positive + negative
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(min(positive, negative) + 1))
    return min(1.0, 2.0 * tail / (2**n))


def infer_paired_strategies(
    curves: Mapping[str, MetricCurves],
    *,
    reference: str,
    alpha: float = 0.05,
) -> tuple[PairedInference, ...]:
    """Apply exact two-sided paired sign tests with Holm family-wise correction.

    Each hypothesis concerns the final evaluation only. The sign test assumes
    independent seed pairs and a symmetric null of equally likely difference
    signs; it does not test the mean difference. Ties are excluded from the
    effective sign-test sample. Standardized mean differences are descriptive.
    """
    if not isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be finite and between zero and one.")
    if reference not in curves or len(curves) < 2:
        raise ValueError("A reference and at least one alternative are required.")
    baseline = curves[reference]
    records = []
    for name, candidate in curves.items():
        if name == reference:
            continue
        differences = paired_metric_difference(candidate, baseline).final_values
        if differences.ndim != 1 or differences.numel() < 2:
            raise ValueError("At least two paired seeds are required.")
        if not torch.isfinite(differences).all():
            raise ValueError("Paired differences must be finite.")
        positive = int((differences > 0).sum().item())
        negative = int((differences < 0).sum().item())
        std = differences.std(correction=1).item()
        records.append(
            (
                name,
                differences.numel(),
                positive + negative,
                differences.mean().item(),
                differences.mean().item() / std if std > 0 else None,
                _sign_test_p_value(positive, negative),
            )
        )
    m = len(records)
    order = sorted(range(m), key=lambda index: records[index][-1])
    adjusted = [0.0] * m
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * records[index][-1]))
        adjusted[index] = running
    return tuple(
        PairedInference(
            strategy=name,
            reference=reference,
            n_pairs=n_pairs,
            n_non_ties=n_non_ties,
            mean_difference=mean,
            standardized_mean_difference=effect,
            p_value=p,
            adjusted_p_value=adjusted[index],
            significant=adjusted[index] <= alpha,
            lower_is_better=baseline.metric == "simple_regret",
        )
        for index, (name, n_pairs, n_non_ties, mean, effect, p) in enumerate(records)
    )
