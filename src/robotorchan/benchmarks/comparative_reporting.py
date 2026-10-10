"""Phase 22 reproducible Markdown summaries for benchmark comparisons."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from robotorchan.benchmarks.comparative_constrained import ConstrainedComparison
from robotorchan.benchmarks.comparative_single_objective import SingleObjectiveComparison

_DIRECTIONS = {
    "simple_regret": "minimize",
    "regret": "minimize",
    "feasible_regret": "minimize",
    "hypervolume": "maximize",
    "feasible_hypervolume": "maximize",
}


@dataclass(frozen=True)
class BenchmarkReport:
    """Machine-readable metadata and a human-readable Markdown report."""

    metric: str
    direction: str
    seeds: tuple[int, ...]
    final_evaluations: int
    markdown: str


def _format(value: Tensor) -> str:
    number = float(value.item())
    if torch.isnan(value):
        return "undefined"
    if torch.isposinf(value):
        return "+inf"
    if torch.isneginf(value):
        return "-inf"
    return f"{number:.6g}"


def _report(
    *,
    metric: str,
    seeds: tuple[int, ...],
    evaluations: Tensor,
    means: dict[str, Tensor],
    errors: dict[str, Tensor],
    title: str,
) -> BenchmarkReport:
    if metric not in _DIRECTIONS:
        raise ValueError("Unknown benchmark metric.")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Report requires unique, nonempty seeds.")
    if (
        evaluations.ndim != 1
        or evaluations.numel() == 0
        or evaluations.dtype not in (torch.int32, torch.int64)
        or evaluations[0].item() != 0
        or (evaluations[1:] <= evaluations[:-1]).any()
    ):
        raise ValueError("Evaluation checkpoints must start at zero and increase.")
    if not means or set(means) != set(errors):
        raise ValueError("Every method needs mean and standard error.")
    rows = []
    for method in sorted(means):
        mean, error = means[method], errors[method]
        if mean.shape != evaluations.shape or error.shape != evaluations.shape:
            raise ValueError("Summary arrays must match evaluation checkpoints.")
        if torch.isnan(mean).any() or torch.isneginf(mean).any():
            raise ValueError("Mean scores must not contain NaN or negative infinity.")
        if (error < 0).any() or torch.isinf(error).any():
            raise ValueError("Standard errors must be nonnegative or undefined.")
        if (torch.isnan(error) & torch.isfinite(mean)).any():
            raise ValueError("Finite scores require defined standard errors.")
        final_error = "undefined" if torch.isposinf(mean[-1]) else _format(error[-1])
        rows.append(f"| {method} | {_format(mean[-1])} | {final_error} |")
    final = int(evaluations[-1].item())
    direction = _DIRECTIONS[metric]
    lines = [
        f"# {title}",
        "",
        f"- Metric: `{metric}` ({direction})",
        f"- Seeds: {len(seeds)} ({', '.join(map(str, seeds))})",
        f"- Final candidate evaluations: {final}",
        "- Comparison: matched seeds and evaluation checkpoints are required.",
        "- Uncertainty: between-seed standard error; not a significance test.",
        "- Undefined pre-feasibility regret is shown as +inf.",
        "",
        "## Final checkpoint",
        "",
        "| Method | Mean | Standard error |",
        "| --- | ---: | ---: |",
        *rows,
        "",
    ]
    return BenchmarkReport(
        metric=metric,
        direction=direction,
        seeds=seeds,
        final_evaluations=final,
        markdown="\n".join(lines),
    )


def report_single_objective(comparison: SingleObjectiveComparison) -> BenchmarkReport:
    """Summarize matched single-objective regret trajectories."""
    return _report(
        metric="simple_regret",
        seeds=comparison.seeds,
        evaluations=comparison.evaluations,
        means=comparison.mean_regret_by_method,
        errors=comparison.standard_error_by_method,
        title="Single-objective benchmark",
    )


def report_constrained(comparison: ConstrainedComparison) -> BenchmarkReport:
    """Summarize matched constrained regret or hypervolume trajectories."""
    return _report(
        metric=comparison.metric,
        seeds=comparison.seeds,
        evaluations=comparison.evaluations,
        means=comparison.mean_score_by_method,
        errors=comparison.standard_error_by_method,
        title="Constrained benchmark",
    )
