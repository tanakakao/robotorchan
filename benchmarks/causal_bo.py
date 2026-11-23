"""Run the known-DAG CBO versus fixed-set GP-BO versus Random comparison.

Example:
    python benchmarks/causal_bo.py --seeds 0,1,2 --iterations 5
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from robotorchan.benchmarks import run_causal_benchmark, summarize_curves


def _seeds(text: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part.strip()) for part in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated integer seeds.") from exc
    if not values or any(value < 0 for value in values) or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("Seeds must be unique nonnegative integers.")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=_seeds, default=(0, 1, 2))
    parser.add_argument("--n-initial-per-set", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--optimizer", choices=("botorch", "sobol"), default="botorch")
    parser.add_argument("--num-restarts", type=int, default=3)
    parser.add_argument("--raw-samples", type=int, default=32)
    parser.add_argument(
        "--include-gp-scm",
        action="store_true",
        help="Include GP-SCM trained only on the same measured initial interventions.",
    )
    parser.add_argument("--mc-samples", type=int, default=64)
    parser.add_argument("--mc-seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("benchmark_results/causal_bo.csv"))
    args = parser.parse_args()
    methods = (
        ("cbo", "gp_bo", "random", "gp_scm") if args.include_gp_scm else ("cbo", "gp_bo", "random")
    )
    trajectories = run_causal_benchmark(
        seeds=args.seeds,
        n_initial_per_set=args.n_initial_per_set,
        n_iterations=args.iterations,
        optimizer=args.optimizer,
        num_restarts=args.num_restarts,
        raw_samples=args.raw_samples,
        methods=methods,
        mc_samples=args.mc_samples,
        mc_seed=args.mc_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "method",
        "seed",
        "evaluation",
        "is_initial",
        "intervention_set",
        "intervention_values",
        "outcome_truth",
        "best_truth",
        "simple_regret",
        "evaluation_cost",
        "cumulative_cost",
    )
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for trajectory in trajectories:
            for step in trajectory.steps:
                writer.writerow(
                    {
                        "method": trajectory.method,
                        "seed": trajectory.seed,
                        "evaluation": step.evaluation,
                        "is_initial": step.is_initial,
                        "intervention_set": "+".join(step.intervention_nodes),
                        "intervention_values": json.dumps(
                            dict(
                                zip(
                                    step.intervention_nodes,
                                    step.intervention_values,
                                    strict=True,
                                )
                            )
                        ),
                        "outcome_truth": step.outcome_truth,
                        "best_truth": step.best_truth,
                        "simple_regret": step.simple_regret,
                        "evaluation_cost": step.evaluation_cost,
                        "cumulative_cost": step.cumulative_cost,
                    }
                )
    summary = {
        "setup": {
            "seeds": list(args.seeds),
            "n_initial_per_set": args.n_initial_per_set,
            "initial_evaluations": trajectories[0].initial_points,
            "additional_evaluations": args.iterations,
            "optimizer": args.optimizer,
            "num_restarts": args.num_restarts,
            "raw_samples": args.raw_samples,
            "methods": list(methods),
            "mc_samples": args.mc_samples if args.include_gp_scm else None,
            "mc_seed": args.mc_seed if args.include_gp_scm else None,
            "gp_scm_data_policy": (
                "Same performed initial interventions, no extra observational data"
                if args.include_gp_scm
                else "not included"
            ),
            "truth_optimum": 0.0,
            "costs": {"X": 1.0, "Z": 1.5, "X+Z": 3.0},
            "comparison": "equal evaluation count, not equal cumulative cost",
        },
        "methods": {},
    }
    for method in methods:
        selected = [item for item in trajectories if item.method == method]
        metrics = {}
        for field, prop in (
            ("best_truth", "best_curve"),
            ("simple_regret", "regret_curve"),
            ("cumulative_cost", "cost_curve"),
        ):
            curves = {item.seed: getattr(item, prop) for item in selected}
            report = summarize_curves(curves)
            metrics[field] = {
                "mean": report.mean.tolist(),
                "std": report.std.tolist(),
            }
        summary["methods"][method] = metrics
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Per-evaluation records: {args.output}")
    print(f"Aggregated curves: {summary_path}")


if __name__ == "__main__":
    main()
