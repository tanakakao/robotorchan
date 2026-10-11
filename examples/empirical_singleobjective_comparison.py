"""Run a frozen ten-seed empirical single-objective cohort from the command line."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from robotorchan.benchmarks.empirical_noiseless import (
    EXTENDED_STRATEGIES,
    STRATEGIES,
    run_noiseless_singleobjective,
)


def main() -> None:
    """Run one problem/q stratum and persist all raw trajectories and provenance."""
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    parser.add_argument("--problem", choices=("branin", "hartmann6"), required=True)
    parser.add_argument("--q", type=int, choices=(1, 3), required=True)
    parser.add_argument("--extended", action="store_true")
    args = parser.parse_args()
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True))
    if dirty:
        raise RuntimeError("Run measured evidence from a clean git checkout.")
    run_noiseless_singleobjective(
        args.destination,
        problem=args.problem,
        q=args.q,
        git_sha=sha,
        dirty=dirty,
        strategies=STRATEGIES + EXTENDED_STRATEGIES if args.extended else STRATEGIES,
    )


if __name__ == "__main__":
    main()
