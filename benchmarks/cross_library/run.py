"""Run paired CPU benchmark cells and write one JSONL record per run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import random
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from math import isfinite
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from adapters import (
    GP,
    Adapter,
    Ax,
    Baseline,
    BayBE,
    BoFire,
    Optuna,
    RobotorchanStudy,
    Space,
    UnsupportedConfiguration,
)
from botorch.test_functions.multi_objective import C2DTLZ2
from torch import Tensor

from robotorchan.benchmarks.candidate_view import CandidateProblemView
from robotorchan.benchmarks.config import BenchmarkExperimentConfig
from robotorchan.benchmarks.constrained_problems import register_constrained_problems
from robotorchan.benchmarks.metrics import hypervolume_curve, simple_regret_curve
from robotorchan.benchmarks.mixed_problems import register_mixed_problems
from robotorchan.benchmarks.multiobjective_problems import register_multiobjective_problems
from robotorchan.benchmarks.noisy_problems import register_noisy_problems
from robotorchan.benchmarks.observation_noise import SeededObservationNoise
from robotorchan.benchmarks.persistence import trajectory_to_record
from robotorchan.benchmarks.problem import BenchmarkProblem
from robotorchan.benchmarks.random_streams import derive_stream_seed
from robotorchan.benchmarks.registry import BenchmarkProblemRegistry
from robotorchan.benchmarks.runner import BenchmarkTrajectory, run_benchmark
from robotorchan.benchmarks.standard_problems import register_standard_problems

DISTRIBUTIONS = {
    "robotorchan": "robotorchan",
    "botorch": "botorch",
    "ax": "ax-platform",
    "optuna": "optuna",
    "optuna-gp": "optuna",
    "baybe": "baybe",
    "bofire": "bofire",
    "random": "robotorchan",
    "sobol": "robotorchan",
}
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Cell:
    problem: str
    library: str
    q: int
    initial: int
    total: int
    mode: str

    def __post_init__(self) -> None:
        if self.library not in DISTRIBUTIONS or self.mode not in ("A", "B", "C"):
            raise ValueError("Unknown adapter or comparison mode.")
        if self.q < 1 or self.initial < 3 or self.total <= self.initial:
            raise ValueError("Require q >= 1 and 3 <= initial < total.")


def c2_dtlz2() -> BenchmarkProblem:
    function = C2DTLZ2(dim=6, num_objectives=2)
    return BenchmarkProblem(
        name="c2_dtlz2",
        bounds=function.bounds,
        objective=function.evaluate_true,
        directions=("minimize", "minimize"),
        variable_types=("continuous",) * 6,
        constraints=lambda x: function.evaluate_slack(x, noise=False),
        n_constraints=1,
        reference_point=function.ref_point,
    )


def registry() -> BenchmarkProblemRegistry:
    result = BenchmarkProblemRegistry()
    register_standard_problems(result)
    register_mixed_problems(result)
    register_multiobjective_problems(result)
    register_constrained_problems(result)
    register_noisy_problems(result)
    result.register("c2_dtlz2", c2_dtlz2)
    return result


def initial_design(space: Space, n: int, seed: int) -> Tensor:
    """Sample continuous Sobol coordinates and put discrete axes in equal-width bins."""
    problem = space.problem
    stream = derive_stream_seed(seed, "initial_design") % (2**32)
    unit = torch.quasirandom.SobolEngine(problem.dimension, scramble=True, seed=stream).draw(
        n, dtype=torch.float64
    )
    low, high = problem.bounds
    design = low + unit * (high - low)
    for i, kind in enumerate(problem.variable_types):
        if kind != "continuous":
            levels = space.levels(i)
            design[:, i] = levels.start + (unit[:, i] * len(levels)).floor()
    return design


@contextmanager
def trial_rng(seed: int):
    """Restore the process RNGs changed by external library calls."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    with torch.random.fork_rng(devices=[]):
        try:
            random.seed(seed)
            np.random.seed(seed % (2**32))
            torch.random.default_generator.manual_seed(seed)
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)


def adapter_for(cell: Cell, space: Space, seed: int, profile: str) -> Adapter:
    if profile == "smoke":
        restarts, raw, mc = 2, 16, 16
    else:
        restarts, raw, mc = 10, 512, 128
    if cell.library in ("random", "sobol"):
        return Baseline(space, cell.library)
    if cell.mode == "B" and cell.library not in ("robotorchan", "botorch"):
        raise UnsupportedConfiguration("Mode B currently matches BoTorch and robotorchan only.")
    if cell.library in ("robotorchan", "botorch"):
        if (
            cell.mode == "A"
            and cell.library == "robotorchan"
            and not space.constraints
            and len(space.objectives) == 1
            and cell.problem != "noisy_quadratic"
            and all(kind == "continuous" for kind in space.problem.variable_types)
        ):
            return RobotorchanStudy(space, seed, restarts, raw)
        return GP(
            space,
            robotorchan=cell.library == "robotorchan",
            noisy=cell.problem == "noisy_quadratic",
            restarts=restarts,
            raw=raw,
            mc=mc,
        )
    if cell.library == "ax":
        return Ax(space, seed)
    if cell.library in ("optuna", "optuna-gp"):
        return Optuna(space, seed, gp=cell.library == "optuna-gp")
    if cell.library == "baybe":
        return BayBE(space)
    return BoFire(space, seed)


def normal_noise(y: Tensor, rng: torch.Generator) -> Tensor:
    return 0.1 * torch.randn(y.shape, generator=rng, dtype=y.dtype, device=y.device)


def package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def metrics(problem: BenchmarkProblem, trajectory: BenchmarkTrajectory) -> dict:
    feasible = (trajectory.constraints >= 0).all(dim=-1)
    best = (
        problem.to_maximization(trajectory.Y_observed[feasible]).amax(0) if feasible.any() else ()
    )
    signs = [1 if d == "maximize" else -1 for d in problem.directions]
    result = {
        "feasible_count": int(feasible.sum()),
        "constraint_violation_rate": float((~feasible).double().mean()),
        "first_feasible_evaluation": next(
            (i for i, valid in enumerate(feasible.tolist(), 1) if valid), None
        ),
        "best_observed": [float(value) * sign for value, sign in zip(best, signs, strict=True)]
        if len(best)
        else None,
    }
    if problem.n_objectives == 1 and problem.optimal_value is not None:
        result["simple_regret"] = [
            x if isfinite(x) else None for x in simple_regret_curve(problem, trajectory).tolist()
        ]
    if problem.n_objectives == 2 and problem.reference_point is not None:
        result["hypervolume"] = hypervolume_curve(problem, trajectory).tolist()
    return result


def run_cell(
    cell: Cell, seed: int, *, profile: str, commit: str | None, problem_registry=None
) -> dict:
    problems = registry() if problem_registry is None else problem_registry
    problem = problems.create(cell.problem)
    space = Space(CandidateProblemView.from_problem(problem))
    design = initial_design(space, cell.initial, seed)
    config = BenchmarkExperimentConfig(
        problem=cell.problem,
        strategy=cell.library,
        seeds=(seed,),
        initial_points=cell.initial,
        evaluation_budget=cell.total - cell.initial,
        q=cell.q,
    )
    batches = []
    record = {
        "schema_version": SCHEMA_VERSION,
        "library": cell.library,
        "library_version": package_version(DISTRIBUTIONS[cell.library]),
        "algorithm": None,
        "batch_method": None,
        "config": config.to_dict(),
        "problem_metadata": {
            "bounds": problem.bounds.tolist(),
            "directions": list(problem.directions),
            "variable_types": list(problem.variable_types),
            "n_constraints": problem.n_constraints,
            "reference_point": (
                problem.reference_point.tolist() if problem.reference_point is not None else None
            ),
        },
        "evaluation_budget_total": cell.total,
        "mode": cell.mode,
        "profile": profile,
        "commit": commit,
        "seed": seed,
        "initial_data_id": None,
        "initial_data": None,
        "evaluation_count": 0,
        "cumulative_cost": 0.0,
        "metrics": None,
        "trajectory": None,
        "observed_batches": batches,
        "timing": {"fit_seconds": None, "update_seconds": 0.0, "total_seconds": None},
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "botorch": package_version("botorch"),
            "torch_threads": torch.get_num_threads(),
            "dtype": "float64",
            "device": "cpu",
        },
        "status": "ok",
        "error": None,
    }
    started = perf_counter()
    stage = "initialization"
    adapter: Adapter | None = None
    try:
        with trial_rng(seed):
            adapter = adapter_for(cell, space, seed, profile)
            record.update(algorithm=adapter.algorithm, batch_method=adapter.batch_method)

            def propose(view, x, y, q, rng):
                nonlocal stage
                stage = "candidate"
                candidates = adapter.ask(q, rng)
                if candidates.shape != (q, view.dimension):
                    raise ValueError(f"Adapter returned {tuple(candidates.shape)} for q={q}.")
                stage = "evaluation"
                return candidates

            def consume(x, y, constraints, costs, initial):
                nonlocal stage
                stage = "initial_update" if initial else "update"
                batch = {
                    "X": x.tolist(),
                    "Y_observed": y.tolist(),
                    "constraints": constraints.tolist(),
                    "costs": costs.tolist(),
                }
                record["evaluation_count"] += len(x)
                record["cumulative_cost"] += float(costs.sum())
                if initial:
                    record["initial_data"] = batch
                    record["initial_data_id"] = hashlib.sha256(
                        json.dumps(batch, sort_keys=True, allow_nan=False).encode()
                    ).hexdigest()
                else:
                    batches.append(batch)
                update_started = perf_counter()
                try:
                    adapter.tell(x, y, constraints, initial=initial)
                finally:
                    record["timing"]["update_seconds"] += perf_counter() - update_started

            stage = "initial_evaluation"
            (trajectory,) = run_benchmark(
                config,
                propose,
                registry=problems,
                restrict_candidate_problem=True,
                independent_rng_streams=True,
                isolate_strategy_rng=False,
                initial_design=design,
                observation_consumer=consume,
                observation_noise=(
                    SeededObservationNoise(normal_noise)
                    if cell.problem == "noisy_quadratic"
                    else None
                ),
            )
            stage = "metrics"
            record["trajectory"] = trajectory_to_record(trajectory)
            record["metrics"] = metrics(problem, trajectory)
    except UnsupportedConfiguration as error:
        record.update(status="unsupported", error=str(error))
    except ImportError as error:
        record.update(status="dependency_error", error=f"{type(error).__name__}: {error}")
    except Exception as error:  # External libraries expose different exception types.
        record.update(status=f"{stage}_error", error=f"{type(error).__name__}: {error}")
    if isinstance(adapter, GP):
        record["timing"]["fit_seconds"] = adapter.fit_seconds
    record["timing"]["total_seconds"] = perf_counter() - started
    return record


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def summary_row(record: dict) -> dict:
    values = record["metrics"] or {}
    regret = values.get("simple_regret", [None])
    hypervolume = values.get("hypervolume", [None])
    trajectory = record["trajectory"] or {}
    return {
        "problem": record["config"]["problem"],
        "library": record["library"],
        "version": record["library_version"],
        "algorithm": record["algorithm"],
        "batch_method": record["batch_method"],
        "mode": record["mode"],
        "seed": record["seed"],
        "q": record["config"]["q"],
        "initial_data_id": record["initial_data_id"],
        "evaluation_count": record["evaluation_count"],
        "cumulative_cost": record["cumulative_cost"],
        "final_simple_regret": regret[-1],
        "final_hypervolume": hypervolume[-1],
        "constraint_violation_rate": values.get("constraint_violation_rate"),
        "fit_seconds": record["timing"]["fit_seconds"],
        "candidate_seconds": (
            sum(trajectory["candidate_seconds"]) if "candidate_seconds" in trajectory else None
        ),
        "total_seconds": record["timing"]["total_seconds"],
        "status": record["status"],
        "error": record["error"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("smoke.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, help="Optional one-row-per-run CSV.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--libraries", choices=tuple(DISTRIBUTIONS), nargs="+")
    parser.add_argument("--problems", nargs="+")
    parser.add_argument("--profile", choices=("smoke", "standard"), default="smoke")
    args = parser.parse_args()
    cells = [Cell(**row) for row in json.loads(args.config.read_text(encoding="utf-8"))]
    cells = [
        cell
        for cell in cells
        if (not args.libraries or cell.library in args.libraries)
        and (not args.problems or cell.problem in args.problems)
    ]
    if not cells or any(seed < 0 for seed in args.seeds) or len(set(args.seeds)) != len(args.seeds):
        parser.error("Select at least one cell and unique nonnegative seeds.")
    torch.set_num_threads(1)
    commit = git_commit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.summary and args.summary.resolve() == args.output.resolve():
        parser.error("The output and summary paths must be different.")
    failed = False
    summaries = []
    with args.output.open("x", encoding="utf-8") as output:
        for cell in cells:
            for seed in args.seeds:
                record = run_cell(cell, seed, profile=args.profile, commit=commit)
                output.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                output.flush()
                if args.summary:
                    summaries.append(summary_row(record))
                failed |= record["status"] != "ok"
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        with args.summary.open("x", encoding="utf-8", newline="") as summary:
            writer = csv.DictWriter(summary, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
