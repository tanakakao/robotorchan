"""Create a reproducible Phase 4 summary from four saved Phase 3 reports.

Usage: python scripts/render_empirical_four_cell_report.py ARTIFACT_ROOT OUTPUT.md
Expected inputs: ARTIFACT_ROOT/{branin,hartmann6}-q{1,3}/report.json.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

PROBLEMS = ("branin", "hartmann6")
BATCH_SIZES = (1, 3)
STRATEGIES = ("random", "sobol", "qEI", "qNEI")


def render(root: Path) -> str:
    """Reject incomplete or inconsistent reports rather than filling missing cells."""
    reports = {}
    for problem in PROBLEMS:
        for q in BATCH_SIZES:
            path = root / f"{problem}-q{q}" / "report.json"
            report = json.loads(path.read_text(encoding="utf-8"))
            if (
                report.get("status") != "complete"
                or report.get("problem") != problem
                or report.get("q") != q
                or report.get("metric") != "simple_regret"
                or report.get("seed_count") != 10
                or not report.get("git_sha")
                or list(report.get("final_metric_by_seed", {})) != list(STRATEGIES)
                or list(report.get("elapsed_seconds_by_seed", {})) != list(STRATEGIES)
            ):
                raise ValueError(f"Invalid or incomplete report: {path}")
            for field in ("final_metric_by_seed", "elapsed_seconds_by_seed"):
                for strategy in STRATEGIES:
                    values = report[field][strategy]
                    if (
                        not isinstance(values, list)
                        or len(values) != 10
                        or any(
                            not isinstance(v, (int, float))
                            or not math.isfinite(v)
                            or (field == "elapsed_seconds_by_seed" and v < 0)
                            for v in values
                        )
                    ):
                        raise ValueError(f"Invalid {field}: {path}, {strategy}")
            reports[problem, q] = report

    lines = [
        "# Phase 4: measured single-objective comparison",
        "",
        "Ten paired seeds per cell; CPU / float64; 20 initial and 50 additional evaluations.",
        "Lower final simple regret is better. Runtimes are descriptive and environment-dependent.",
        "q=1 and q=3 use different numbers of model updates and are **not** a controlled",
        "parallel wall-clock comparison. No cross-q significance is claimed.",
        "",
        "## Final simple regret (mean across 10 seeds)",
        "",
        "| Problem | q | Random | Sobol | qEI | qNEI |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for problem in PROBLEMS:
        for q in BATCH_SIZES:
            report = reports[problem, q]
            means = [
                sum(report["final_metric_by_seed"][strategy]) / 10
                for strategy in STRATEGIES
            ]
            lines.append(
                f"| {problem} | {q} | " + " | ".join(f"{v:.6f}" for v in means) + " |"
            )
    lines += [
        "",
        "## Mean elapsed seconds (descriptive only)",
        "",
        "| Problem | q | Random | Sobol | qEI | qNEI |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for problem in PROBLEMS:
        for q in BATCH_SIZES:
            report = reports[problem, q]
            means = [
                sum(report["elapsed_seconds_by_seed"][strategy]) / 10
                for strategy in STRATEGIES
            ]
            lines.append(
                f"| {problem} | {q} | " + " | ".join(f"{v:.3f}" for v in means) + " |"
            )
    lines += ["", "## Paired improvements over baselines", ""]
    for problem in PROBLEMS:
        for q in BATCH_SIZES:
            report = reports[problem, q]
            lines += [
                f"### {problem}, q={q}",
                "",
                f"Source experiment SHA: `{report['git_sha']}`.",
                "",
                "| Strategy | Baseline | Mean improvement | 95% paired bootstrap CI |",
                "| --- | --- | ---: | --- |",
            ]
            for comparison in report.get("comparisons", []):
                if (
                    comparison.get("strategy") not in STRATEGIES
                    or comparison.get("baseline") not in ("random", "sobol")
                    or comparison.get("n_paired") != 10
                    or len(comparison.get("bootstrap_ci95", [])) != 2
                ):
                    raise ValueError(f"Invalid paired comparison: {problem}, q={q}")
                lo, hi = comparison["bootstrap_ci95"]
                mean = comparison["mean_paired_improvement"]
                if not all(math.isfinite(v) for v in (lo, hi, mean)) or lo > hi:
                    raise ValueError(f"Nonfinite or inverted interval: {problem}, q={q}")
                lines.append(
                    f"| {comparison['strategy']} | {comparison['baseline']} | "
                    f"{mean:+.6f} | [{lo:+.6f}, {hi:+.6f}] |"
                )
            lines.append("")
    lines += [
        "## Interpretation and limitations",
        "",
        "These are four synthetic, noiseless, single-objective cells, not a general",
        "ranking of acquisition functions. The intervals compare strategies within",
        "each cell, not q=1 against q=3. Report provenance and raw trajectories",
        "must be retained alongside this derived Markdown.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    result = render(args.artifact_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result, encoding="utf-8")


if __name__ == "__main__":
    main()
