"""Compare coupled and decoupled regression-classification optimization."""

import argparse
import json

from robotorchan.benchmarks.decoupled_heterogeneous import run_heterogeneous_decoupled_benchmark


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("binary", "multiclass", "ordinal"), default="binary")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--budget", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    for seed in map(int, args.seeds.split(",")):
        for decoupled in (False, True):
            for row in run_heterogeneous_decoupled_benchmark(
                kind=args.kind,
                seed=seed,
                budget=args.budget,
                epochs=args.epochs,
                decoupled=decoupled,
            ):
                print(json.dumps({"problem": "heterogeneous_quality", **row}))


if __name__ == "__main__":
    main()
