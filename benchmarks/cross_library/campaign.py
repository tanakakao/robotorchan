"""Run comparison cells in independent CPU processes with per-trial timeouts."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from time import monotonic

from run import DISTRIBUTIONS, Cell, git_commit

ROOT = Path(__file__).resolve().parent
THREAD_ENV = dict.fromkeys(("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"), "1")


def execute(directory: Path, cell: Cell, seed: int, timeout: int, index: int) -> dict:
    """Retain the worker result and its exit condition separately."""
    trial = directory / f"cell-{index:03d}-s{seed}"
    trial.mkdir()
    config, output = trial / "cell.json", trial / "result.jsonl"
    config.write_text(json.dumps([asdict(cell)]), encoding="utf-8")
    command = [
        sys.executable,
        str(ROOT / "run.py"),
        "--config",
        str(config),
        "--profile",
        "standard",
        "--output",
        str(output),
        "--seeds",
        str(seed),
    ]
    started = monotonic()
    with (
        (trial / "worker.log").open("w", encoding="utf-8") as log,
        subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=os.environ | THREAD_ENV,
            start_new_session=True,
        ) as process,
    ):
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            code = None
    return {
        **asdict(cell),
        "seed": seed,
        "trial": trial.name,
        "worker_wall_seconds": monotonic() - started,
        "returncode": code,
        "status": "timeout" if code is None else "exited",
        "result_record": output.is_file() and output.stat().st_size > 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "phase4.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=720)
    parser.add_argument("--libraries", nargs="+", choices=tuple(DISTRIBUTIONS))
    parser.add_argument("--problems", nargs="+")
    parser.add_argument("--modes", nargs="+", choices=("A", "B", "C"))
    args = parser.parse_args()
    cells = [Cell(**row) for row in json.loads(args.config.read_text(encoding="utf-8"))]
    selected = [
        cell
        for cell in cells
        if (not args.libraries or cell.library in args.libraries)
        and (not args.problems or cell.problem in args.problems)
        and (not args.modes or cell.mode in args.modes)
    ]
    names = [(cell.problem, cell.q, cell.mode, cell.library) for cell in selected]
    if not selected or len(set(names)) != len(names):
        parser.error("Select at least one cell with unique problem/q/mode/library keys.")
    if min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Seeds must be unique and nonnegative.")
    if args.workers < 1 or args.timeout < 1:
        parser.error("Workers and timeout must be positive.")
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {
        "commit": git_commit(),
        "workers": args.workers,
        "timeout_seconds": args.timeout,
        "seeds": args.seeds,
        "cells": [asdict(cell) for cell in selected],
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    failed = False
    with (
        (directory / "jobs.jsonl").open("x", encoding="utf-8") as jobs,
        (directory / "results.jsonl").open("x", encoding="utf-8") as results,
        ThreadPoolExecutor(max_workers=args.workers) as executor,
    ):
        tasks = [
            executor.submit(execute, directory, cell, seed, args.timeout, index)
            for seed in args.seeds
            for index, cell in enumerate(selected)
        ]
        for task in as_completed(tasks):
            job = task.result()
            jobs.write(json.dumps(job, allow_nan=False) + "\n")
            jobs.flush()
            if job["result_record"]:
                path = directory / job["trial"] / "result.jsonl"
                results.write(path.read_text(encoding="utf-8"))
                results.flush()
            failed |= job["returncode"] != 0 or not job["result_record"]
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
