"""Phase 17 paired statistical comparisons for benchmark trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from robotorchan.benchmarks.comparative_constrained import ConstrainedComparison


@dataclass(frozen=True)
class PairedStatistics:
    """Positive improvements favor the challenger over the reference."""

    reference: str
    challenger: str
    evaluations: Tensor
    improvement_by_seed: Tensor
    mean_improvement: Tensor
    confidence_lower: Tensor
    confidence_upper: Tensor
    win_rate: Tensor
    tie_rate: Tensor
    confidence_level: float
    bootstrap_samples: int


def paired_statistics(
    comparison: ConstrainedComparison,
    *,
    reference: str,
    challenger: str,
    confidence_level: float = 0.95,
    bootstrap_samples: int = 2000,
    seed: int = 0,
    tie_tolerance: float = 0.0,
) -> PairedStatistics:
    """Estimate paired bootstrap intervals over seeds, not evaluation steps.

    A single resampled seed index is shared across all checkpoints, retaining
    each trajectory's temporal dependence. Intervals are descriptive percentile
    intervals, not evidence of statistical significance.
    """
    if reference == challenger:
        raise ValueError("Reference and challenger must be different.")
    if reference not in comparison.scores_by_method:
        raise ValueError("Unknown reference method.")
    if challenger not in comparison.scores_by_method:
        raise ValueError("Unknown challenger method.")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be strictly between 0 and 1.")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive.")
    if tie_tolerance < 0 or not torch.isfinite(torch.tensor(tie_tolerance)):
        raise ValueError("tie_tolerance must be finite and nonnegative.")

    baseline = comparison.scores_by_method[reference]
    candidate = comparison.scores_by_method[challenger]
    if baseline.ndim != 2 or candidate.shape != baseline.shape:
        raise ValueError("Paired scores must have matching [seed, checkpoint] shapes.")
    if baseline.shape != (len(comparison.seeds), comparison.evaluations.numel()):
        raise ValueError("Score dimensions must match seeds and checkpoints.")
    if baseline.shape[0] < 2:
        raise ValueError("At least two paired seeds are required for bootstrap intervals.")
    if not torch.is_floating_point(baseline) or not torch.is_floating_point(candidate):
        raise ValueError("Scores must be floating-point tensors.")
    if not torch.isfinite(baseline).all() or not torch.isfinite(candidate).all():
        raise ValueError("Paired scores must be finite.")
    if comparison.metric in ("feasible_hypervolume", "hypervolume"):
        improvement = candidate - baseline
    elif comparison.metric in ("feasible_regret", "regret", "simple_regret"):
        improvement = baseline - candidate
    else:
        raise ValueError("Unknown metric direction.")

    n = improvement.shape[0]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randint(n, (bootstrap_samples, n), generator=generator)
    boot_means = improvement.index_select(0, indices.flatten().to(improvement.device))
    boot_means = boot_means.reshape(bootstrap_samples, n, -1).mean(dim=1)
    alpha = (1.0 - confidence_level) / 2.0
    lower = torch.quantile(boot_means, alpha, dim=0)
    upper = torch.quantile(boot_means, 1.0 - alpha, dim=0)
    wins = improvement > tie_tolerance
    ties = improvement.abs() <= tie_tolerance
    return PairedStatistics(
        reference=reference,
        challenger=challenger,
        evaluations=comparison.evaluations.clone(),
        improvement_by_seed=improvement,
        mean_improvement=improvement.mean(dim=0),
        confidence_lower=lower,
        confidence_upper=upper,
        win_rate=wins.to(improvement.dtype).mean(dim=0),
        tie_rate=ties.to(improvement.dtype).mean(dim=0),
        confidence_level=confidence_level,
        bootstrap_samples=bootstrap_samples,
    )
