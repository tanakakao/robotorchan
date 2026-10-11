"""Render evidence-backed empirical performance and timing charts."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def render_empirical_charts(artifacts: str | Path, destination: str | Path) -> dict[str, object]:
    """Render standalone SVG charts only from complete Phase 31 CSV artifacts."""
    root = Path(artifacts)
    output = Path(destination)
    if output.exists():
        raise FileExistsError("Refusing to overwrite existing empirical charts.")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "complete" or summary.get("seed_count") != 10:
        raise ValueError("Incomplete empirical summary.")
    with (root / "curves.csv").open(newline="", encoding="utf-8") as handle:
        curves = list(csv.DictReader(handle))
    with (root / "timing.csv").open(newline="", encoding="utf-8") as handle:
        timings = list(csv.DictReader(handle))
    strategies = tuple(summary["final_metric_by_seed"])
    if len(curves) != len(strategies) * 10 * 70 or len(timings) != len(strategies) * 10:
        raise ValueError("Missing or duplicate empirical CSV rows.")
    grouped: dict[str, dict[int, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    for row in curves:
        strategy = row["strategy"]
        seed = int(row["seed"])
        total = int(row["total_points"])
        if (
            strategy not in strategies
            or seed not in range(10)
            or total not in range(1, 71)
            or row["problem"] != summary["problem"]
            or int(row["q"]) != summary["q"]
            or row["metric"] != summary["metric"]
            or total in grouped[strategy][seed]
        ):
            raise ValueError("Invalid empirical curve row.")
        value = float(row["value"])
        if not math.isfinite(value):
            raise ValueError("Nonfinite empirical curve.")
        if int(row["evaluations"]) != max(0, total - 20):
            raise ValueError("Incorrect additional evaluation index.")
        if total == 70 and not math.isclose(
            value,
            summary["final_metric_by_seed"][strategy][seed],
            rel_tol=1e-10,
            abs_tol=1e-12,
        ):
            raise ValueError("Curve final value disagrees with summary.")
        grouped[strategy][seed][total] = value
    if any(
        set(grouped[strategy][seed]) != set(range(1, 71))
        for strategy in strategies
        for seed in range(10)
    ):
        raise ValueError("Incomplete per-seed empirical curve.")
    seen: set[tuple[str, int]] = set()
    for row in timings:
        key = (row["strategy"], int(row["seed"]))
        if (
            key in seen
            or key[0] not in strategies
            or key[1] not in range(10)
            or row["problem"] != summary["problem"]
            or int(row["q"]) != summary["q"]
        ):
            raise ValueError("Invalid timing row.")
        components = [
            float(row[name])
            for name in (
                "candidate_seconds",
                "evaluation_seconds",
                "initial_evaluation_seconds",
            )
        ]
        total_seconds = float(row["total_seconds"])
        expected_seconds = summary["elapsed_seconds_by_seed"][key[0]][key[1]]
        if (
            not all(math.isfinite(value) and value >= 0 for value in components)
            or not math.isfinite(total_seconds)
            or not math.isclose(sum(components), total_seconds, rel_tol=1e-10, abs_tol=1e-12)
            or not math.isclose(
                total_seconds, expected_seconds, rel_tol=1e-10, abs_tol=1e-12
            )
        ):
            raise ValueError("Timing CSV disagrees with empirical summary.")
        seen.add(key)
    if len(seen) != len(strategies) * 10:
        raise ValueError("Incomplete timing rows.")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Install matplotlib to render empirical charts.") from exc

    output.mkdir(parents=True)
    try:
        fig, ax = plt.subplots(figsize=(9, 5))
        for strategy in strategies:
            values = [
                sum(grouped[strategy][seed][point] for seed in range(10)) / 10
                for point in range(20, 71)
            ]
            ax.plot(range(0, 51), values, label=strategy)
        ax.set(
            title=f"{summary['problem']} (q={summary['q']})",
            xlabel="Additional evaluations",
            ylabel=summary["metric"],
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / "performance.svg")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 5))
        means = [sum(summary["elapsed_seconds_by_seed"][strategy]) / 10 for strategy in strategies]
        ax.bar(strategies, means)
        ax.set(ylabel="Elapsed seconds", title="Mean wall-clock time (10 seeds)")
        fig.tight_layout()
        fig.savefig(output / "timing.svg")
        plt.close(fig)
        (output / "provenance.json").write_text(
            json.dumps(
                {
                    "source": str(root),
                    "git_sha": summary["git_sha"],
                    "problem": summary["problem"],
                    "q": summary["q"],
                    "seed_count": 10,
                    "metric": summary["metric"],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    except Exception:
        plt.close("all")
        for path in output.iterdir():
            path.unlink()
        output.rmdir()
        raise
    return {"status": "rendered", "files": ["performance.svg", "timing.svg", "provenance.json"]}
