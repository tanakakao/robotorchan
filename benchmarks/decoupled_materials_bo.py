"""Compare masked and coupled material-property evaluations."""

import argparse
import json

from robotorchan.benchmarks.decoupled_materials import run_materials_decoupled_benchmark


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--weights", default="0.25,0.5,0.75")
    parser.add_argument("--budget", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    for seed in map(int, args.seeds.split(",")):
        for weight in map(float, args.weights.split(",")):
            for decoupled in (False, True):
                for row in run_materials_decoupled_benchmark(
                    seed=seed,
                    budget=args.budget,
                    epochs=args.epochs,
                    decoupled=decoupled,
                    strength_weight=weight,
                ):
                    print(json.dumps({"problem": "strength_conductivity_pass_tradeoff", **row}))


if __name__ == "__main__":
    main()
