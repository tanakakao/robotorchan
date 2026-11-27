"""Run paired equal-budget experiments and emit JSON Lines."""

import argparse
import json

from robotorchan.benchmarks.decoupled import run_decoupled_benchmark
from robotorchan.benchmarks.multiobjective_problems import branin_currin, dtlz2, zdt1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--problem", choices=("branin_currin", "zdt1", "dtlz2"), default="branin_currin"
    )
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--budget", type=int, default=40)
    parser.add_argument("--policies", default="epig,kg,hvkg,random,sobol")
    parser.add_argument("--no-fit", action="store_true")
    args = parser.parse_args()
    problem = {"branin_currin": branin_currin, "zdt1": zdt1, "dtlz2": dtlz2}[args.problem]()
    costs = tuple(2 * i + 1 for i in range(problem.n_objectives))
    for seed in map(int, args.seeds.split(",")):
        for policy in args.policies.split(","):
            for decoupled in (False, True):
                for row in run_decoupled_benchmark(
                    problem,
                    seed=seed,
                    budget=args.budget,
                    output_costs=costs,
                    decoupled=decoupled,
                    policy=policy,
                    fit_model=not args.no_fit,
                ):
                    print(json.dumps({"problem": problem.name, **row}))


if __name__ == "__main__":
    main()
