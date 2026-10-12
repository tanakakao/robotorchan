"""Run a real CPU benchmark smoke or ten-seed baseline cohort."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch

from robotorchan.benchmarks.comparative_baselines import run_comparative_baseline
from robotorchan.benchmarks.comparative_matrix import ComparativeExperimentCell
from robotorchan.benchmarks.config import BenchmarkExperimentConfig
from robotorchan.benchmarks.empirical_runtime_evidence import runtime_breakdown


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=("branin", "hartmann6"), required=True)
    parser.add_argument("--q", type=int, choices=(1, 3), required=True)
    parser.add_argument("--seeds", type=int, choices=(2, 10), default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--git-sha", required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output directory already exists.")
    args.output.mkdir(parents=True)
    manifest = {
        "problem": args.problem,
        "q": args.q,
        "seeds": list(range(args.seeds)),
        "strategies": ["random", "sobol"],
        "initial_points": 20,
        "evaluation_budget": 50,
        "device": "cpu",
        "dtype": "float64",
        "git_sha": args.git_sha,
        "status": "running",
        "scope": "measured_random_sobol_baselines_not_bayesian_optimization",
    }
    rows = []
    try:
        for strategy in manifest["strategies"]:
            config = BenchmarkExperimentConfig(
                problem=args.problem,
                strategy=strategy,
                seeds=tuple(range(args.seeds)),
                initial_points=20,
                evaluation_budget=50,
                q=args.q,
            )
            runs = run_comparative_baseline(
                ComparativeExperimentCell(tier="empirical_standard", config=config)
            )
            if {run.seed for run in runs} != set(range(args.seeds)):
                raise ValueError("Missing or duplicated seeds.")
            for run in runs:
                if run.evaluation_count != 50 or run.X.dtype != torch.float64:
                    raise ValueError("Unexpected budget or dtype.")
                timings = runtime_breakdown(run)
                if not math.isfinite(timings["total_measured_seconds"]):
                    raise ValueError("Invalid runtime.")
                rows.append({"strategy": strategy, "seed": run.seed, **timings})
        with (args.output / "runtime_by_seed.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        manifest["status"] = "measured"
        manifest["row_count"] = len(rows)
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        (args.output / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
