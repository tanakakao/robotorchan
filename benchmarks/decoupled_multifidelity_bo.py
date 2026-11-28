"""Compare high-fidelity and multi-fidelity masked KG policies."""

import argparse
import json

from robotorchan.benchmarks.decoupled_multifidelity import run_multifidelity_decoupled_benchmark


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--budget", type=int, default=72)
    parser.add_argument("--no-fit", action="store_true")
    args = parser.parse_args()
    for seed in map(int, args.seeds.split(",")):
        for multifidelity in (False, True):
            for decoupled in (False, True):
                for row in run_multifidelity_decoupled_benchmark(
                    seed=seed,
                    budget=args.budget,
                    decoupled=decoupled,
                    multifidelity=multifidelity,
                    fit_model=not args.no_fit,
                ):
                    print(json.dumps({"problem": "multifidelity_two_response", **row}))


if __name__ == "__main__":
    main()
