"""Execute and certify paired multiobjective empirical comparisons."""

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
from robotorchan.benchmarks.comparative_qehvi import run_comparative_qehvi
from robotorchan.benchmarks.comparative_qnehvi import run_comparative_qnehvi
from robotorchan.benchmarks.config import BenchmarkExperimentConfig
from robotorchan.benchmarks.initial_pairing import validate_initial_pairing
from robotorchan.benchmarks.multiobjective_problems import branin_currin
from robotorchan.benchmarks.persistence import save_trajectory
from robotorchan.benchmarks.runner import BenchmarkTrajectory

STRATEGIES = ("random", "sobol", "qEHVI", "qNEHVI")
PROBLEMS = ("branin_currin",)
RUNNERS: dict[str, Callable] = {
    "random": run_comparative_baseline,
    "sobol": run_comparative_baseline,
    "qEHVI": run_comparative_qehvi,
    "qNEHVI": run_comparative_qnehvi,
}


def run_multiobjective(
    destination: str | Path,
    *,
    problem: str,
    q: int,
    git_sha: str,
    dirty: bool,
    num_restarts: int = 3,
    raw_samples: int = 64,
    mc_samples: int = 128,
) -> dict[str, object]:
    """Persist full seed-paired runs; raise rather than silently omit failures.

    Run separately for each problem and q. A failed run leaves an error record
    and must not be reported as a completed comparison.
    """
    if problem not in PROBLEMS:
        raise ValueError("Unsupported multiobjective problem.")
    if q not in (1, 3):
        raise ValueError("q must be 1 or 3.")
    if not git_sha or type(dirty) is not bool:
        raise ValueError("Explicit git SHA and dirty state are required.")
    if type(num_restarts) is not int or num_restarts < 1:
        raise ValueError("num_restarts must be positive.")
    if type(raw_samples) is not int or raw_samples < 1:
        raise ValueError("raw_samples must be positive.")

    if type(mc_samples) is not int or mc_samples < 1:
        raise ValueError("mc_samples must be positive.")

    root = Path(destination)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Refusing to overwrite an existing experiment directory.")
    root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "protocol": "empirical-evaluation-protocol",
        "problem": problem,
        "strategies": list(STRATEGIES),
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
        "mc_samples": mc_samples,
        "reference_point": branin_currin().reference_point.tolist(),
        "objective_directions": list(branin_currin().directions),
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
        for strategy in STRATEGIES:
            config = BenchmarkExperimentConfig(
                problem=problem,
                strategy=strategy,
                seeds=tuple(range(10)),
                initial_points=20,
                evaluation_budget=50,
                q=q,
            )
            cell = ComparativeExperimentCell(tier="empirical_standard", config=config)
            if strategy in ("qEHVI", "qNEHVI"):
                runs = RUNNERS[strategy](
                    cell,
                    num_restarts=num_restarts,
                    raw_samples=raw_samples,
                    mc_samples=mc_samples,
                )
            else:
                runs = RUNNERS[strategy](cell)
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
                if run.Y_truth.shape != (70, 2):
                    raise ValueError("Incorrect multiobjective outcome shape.")
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
