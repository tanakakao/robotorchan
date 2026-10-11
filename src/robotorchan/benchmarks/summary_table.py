"""CSV-ready final-evaluation summaries for empirical strategy comparisons."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

from robotorchan.benchmarks.comparison_report import EmpiricalComparisonReport

_COLUMNS = (
    "strategy",
    "metric",
    "reference",
    "lower_is_better",
    "n_seeds",
    "final_mean",
    "final_ci_lower",
    "final_ci_upper",
    "paired_mean_difference",
    "paired_ci_lower",
    "paired_ci_upper",
    "final_win_rate",
    "adjusted_p_value",
    "significant",
)


@dataclass(frozen=True)
class StrategySummaryRow:
    """Final performance and reference-relative statistics for one strategy."""

    strategy: str
    metric: str
    reference: str
    lower_is_better: bool
    n_seeds: int
    final_mean: float
    final_ci_lower: float
    final_ci_upper: float
    paired_mean_difference: float | None
    paired_ci_lower: float | None
    paired_ci_upper: float | None
    final_win_rate: float | None
    adjusted_p_value: float | None
    significant: bool | None

    def to_dict(self) -> dict[str, str | int | float | bool | None]:
        """Return fields in stable CSV column order."""
        return {column: getattr(self, column) for column in _COLUMNS}


def _final_number(value: object) -> float:
    if not isinstance(value, (tuple, list)) or not value:
        raise ValueError("Expected a nonempty metric history.")
    number = float(value[-1])
    if not isfinite(number):
        raise ValueError("Summary statistics must be finite.")
    return number


def summarize_comparison_table(
    report: EmpiricalComparisonReport,
) -> tuple[StrategySummaryRow, ...]:
    """Summarize final evaluations without interpreting significance as ranking."""
    comparisons = {item["first"]: item for item in report.comparisons}
    inference = {item["strategy"]: item for item in report.inference}
    lower_is_better = report.metric == "simple_regret"
    rows = []
    for name, stats in sorted(report.strategies.items()):
        comparison = comparisons.get(name)
        test = inference.get(name)
        paired_stats = comparison["statistics"] if comparison is not None else None
        rows.append(
            StrategySummaryRow(
                strategy=name,
                metric=report.metric,
                reference=report.reference,
                lower_is_better=lower_is_better,
                n_seeds=int(stats["n_seeds"]),
                final_mean=_final_number(stats["mean"]),
                final_ci_lower=_final_number(stats["ci_lower"]),
                final_ci_upper=_final_number(stats["ci_upper"]),
                paired_mean_difference=(
                    _final_number(paired_stats["mean"]) if paired_stats is not None else None
                ),
                paired_ci_lower=(
                    _final_number(paired_stats["ci_lower"]) if paired_stats is not None else None
                ),
                paired_ci_upper=(
                    _final_number(paired_stats["ci_upper"]) if paired_stats is not None else None
                ),
                final_win_rate=(
                    float(comparison["final_win_rate"]) if comparison is not None else None
                ),
                adjusted_p_value=(
                    float(test["adjusted_p_value"]) if test is not None else None
                ),
                significant=bool(test["significant"]) if test is not None else None,
            )
        )
    return tuple(rows)


def save_comparison_table_csv(
    rows: tuple[StrategySummaryRow, ...],
    path: str | Path,
) -> None:
    """Export a stable CSV schema suitable for spreadsheets and reports."""
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_COLUMNS)
        writer.writeheader()
        writer.writerows(row.to_dict() for row in rows)
