"""Execute and certify paired noiseless single-objective empirical comparisons."""

from __future__ import annotations

import json
import platform
import sys
from collections.abc import Callable
from pathlib import Path

import botorch
import gpytorch
import torch

from robotorchan.benchmarks.comparative_baselines import run_comparative_baseline
from robotorchan.benchmarks.comparative_matrix import ComparativeExperimentCell
from robotorchan.benchmarks.comparative_qei import run_comparative_qei
from robotorchan.benchmarks.comparative_qlogei import run_comparative_qlogei
from robotorchan.benchmarks.comparative_qlognei import run_comparative_qlognei
from robotorchan.benchmarks.comparative_qnei import run_comparative_qnei
from robotorchan.benchmarks.comparative_qts import run_comparative_qts
from robotorchan.benchmarks.comparative_qucb_qpi import run_comparative_qucb_qpi
from robotorchan.benchmarks.config import BenchmarkExperimentConfig
from robotorchan.benchmarks.initial_pairing import validate_initial_pairing
from robotorchan.benchmarks.persistence import save_trajectory
from robotorchan.benchmarks.runner import BenchmarkTrajectory

STRATEGIES = ("random", "sobol", "qEI", "qNEI")
EXTENDED_STRATEGIES = ("qLogEI", "qLogNEI", "qPI", "qUCB", "qTS")
PROBLEMS = ("branin", "hartmann6")
RUNNERS: dict[str, Callable] = {
    "random": run_comparative_baseline,
    "sobol": run_comparative_baseline,
    "qEI": run_comparative_qei,
    "qNEI": run_comparative_qnei,
    "qLogEI": run_comparative_qlogei,
    "qLogNEI": run_comparative_qlognei,
    "qPI": run_comparative_qucb_qpi,
    "qUCB": run_comparative_qucb_qpi,
    "qTS": run_comparative_qts,
}


def run_noiseless_singleobjective(
    destination: str | Path,
    *,
    problem: str,
    q: int,
    git_sha: str,
    dirty: bool,
    num_restarts: int = 3,
    raw_samples: int = 64,
    strategies: tuple[str, ...] = STRATEGIES,
) -> dict[str, object]:
    """Persist full seed-paired runs; raise rather than silently omit failures.

    Run separately for each problem and q. A failed run leaves an error record
    and must not be reported as a completed comparison.
    """
    if problem not in PROBLEMS:
        raise ValueError("Unsupported noiseless single-objective problem.")
    if q not in (1, 3):
        raise ValueError("q must be 1 or 3.")
    if not git_sha or type(dirty) is not bool:
        raise ValueError("Explicit git SHA and dirty state are required.")
    if type(num_restarts) is not int or num_restarts < 1:
        raise ValueError("num_restarts must be positive.")
    if type(raw_samples) is not int or raw_samples < 1:
        raise ValueError("raw_samples must be positive.")

    if not strategies or len(set(strategies)) != len(strategies):
        raise ValueError("Strategies must be nonempty and unique.")
    if any(strategy not in RUNNERS for strategy in strategies):
        raise ValueError("Unknown empirical strategy.")
    if "random" not in strategies or "sobol" not in strategies:
        raise ValueError("Empirical comparison requires Random and Sobol baselines.")

    root = Path(destination)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Refusing to overwrite an existing experiment directory.")
    root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "protocol": "empirical-evaluation-protocol",
        "problem": problem,
        "strategies": list(strategies),
        "q": q,
        "seeds": list(range(10)),
        "initial_points": 20,
        "evaluation_budget": 50,
        "dtype": "float64",
        "device": "cpu",
        "git_sha": git_sha,
        "dirty": dirty,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "botorch": botorch.__version__,
        "gpytorch": gpytorch.__version__,
        "num_restarts": num_restarts,
        "raw_samples": raw_samples,
        "observation_noise": "problem_default_noiseless",
        "rng": "independent_streams_and_isolated_strategy_torch_rng",
        "status": "running",
    }
    manifest_path = root / "manifest.json"

    def persist_manifest() -> None:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    persist_manifest()
    trajectories: dict[str, tuple[BenchmarkTrajectory, ...]] = {}
    try:
        for strategy in strategies:
            config = BenchmarkExperimentConfig(
                problem=problem,
                strategy=strategy,
                seeds=tuple(range(10)),
                initial_points=20,
                evaluation_budget=50,
                q=q,
            )
            cell = ComparativeExperimentCell(tier="empirical_standard", config=config)
            if strategy == "qTS" or strategy in ("random", "sobol"):
                runs = RUNNERS[strategy](cell)
            else:
                runs = RUNNERS[strategy](cell, num_restarts=num_restarts, raw_samples=raw_samples)
            if len(runs) != 10 or {run.seed for run in runs} != set(range(10)):
                raise ValueError("Missing or duplicate seed trajectories.")
            for run in runs:
                if run.initial_points != 20 or run.evaluation_count != 50:
                    raise ValueError("Incorrect empirical evaluation budget.")
                if not run.completed_batches or run.completed_batches[-1] != 50:
                    raise ValueError("Incomplete candidate batch history.")
                if q == 3 and (
                    len(run.completed_batches) != 17
                    or run.completed_batches[-1] - run.completed_batches[-2] != 2
                ):
                    raise ValueError("Incorrect final q=3 batch.")
                if not torch.isfinite(run.Y_truth).all():
                    raise ValueError("Nonfinite truth outcomes.")
                save_trajectory(run, root / f"{strategy}_seed{run.seed}.json")
            trajectories[strategy] = runs
        pairing = validate_initial_pairing(trajectories)
        manifest["pairing"] = {
            "design": pairing.design_paired,
            "truth": pairing.truth_paired,
            "observation": pairing.observation_paired,
        }
        manifest["status"] = "runtime_validated"
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
        persist_manifest()
        raise
    persist_manifest()
    return manifest
