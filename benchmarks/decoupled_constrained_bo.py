"""Run constrained coupled and decoupled KG experiments."""

import argparse
import json

from robotorchan.benchmarks.constrained_problems import (
    constrained_annulus,
    constrained_disconnected,
    constrained_narrow_band,
    constrained_quadratic,
)
from robotorchan.benchmarks.decoupled_constrained import run_constrained_decoupled_benchmark


def main():
    factories = {
        "quadratic": constrained_quadratic,
        "annulus": constrained_annulus,
        "disconnected": constrained_disconnected,
        "narrow_band": constrained_narrow_band,
    }
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=tuple(factories), default="quadratic")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--budget", type=int, default=40)
    parser.add_argument("--no-fit", action="store_true")
    args = parser.parse_args()
    problem = factories[args.problem]()
    for seed in map(int, args.seeds.split(",")):
        for decoupled in (False, True):
            for row in run_constrained_decoupled_benchmark(
                problem,
                output_costs=(3,) + (1,) * problem.n_constraints,
                infeasible_utility=-5.0,
                budget=args.budget,
                seed=seed,
                decoupled=decoupled,
                fit_model=not args.no_fit,
            ):
                print(json.dumps({"problem": problem.name, **row}))


if __name__ == "__main__":
    main()
