"""Summarize one decoupled benchmark configuration from JSONL records."""

import argparse
import json
from pathlib import Path

from robotorchan.benchmarks.decoupled_reporting import summarize_cost_curves


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--stop", type=int, required=True)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--baseline")
    parser.add_argument("--kind")
    parser.add_argument("--weight", type=float)
    parser.add_argument("--problem")
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    if args.step <= 0 or args.stop < args.start:
        parser.error("Require positive step and stop >= start.")
    records = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    filters = {
        "kind": args.kind,
        "strength_weight": args.weight,
        "problem": args.problem,
    }
    records = [
        row
        for row in records
        if all(value is None or row.get(key) == value for key, value in filters.items())
    ]
    for key in filters:
        if len({row.get(key) for row in records}) > 1:
            parser.error(f"Select a single configuration for {key}.")
    rows = summarize_cost_curves(
        records,
        args.metric,
        list(range(args.start, args.stop + 1, args.step)),
        baseline=args.baseline,
    )
    for row in rows:
        print(json.dumps(row))
    if args.plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        for method in sorted({row["method"] for row in rows}):
            selected = [row for row in rows if row["method"] == method]
            x = [row["cost"] for row in selected]
            ax.step(x, [row["mean"] for row in selected], where="post", label=method)
            ax.fill_between(
                x,
                [row["lower"] for row in selected],
                [row["upper"] for row in selected],
                step="post",
                alpha=0.2,
            )
        ax.set(xlabel="Cumulative evaluation cost", ylabel=args.metric)
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.plot)
        plt.close(fig)


if __name__ == "__main__":
    main()
