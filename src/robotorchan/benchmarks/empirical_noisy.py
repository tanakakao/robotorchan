"""Execute and certify paired noisy single-objective empirical comparisons."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import botorch
import gpytorch
import torch

from robotorchan.benchmarks.comparative_baselines import sobol_candidates
from robotorchan.benchmarks.comparative_matrix import ComparativeExperimentCell
from robotorchan.benchmarks.comparative_qei import make_comparative_qei_strategy
from robotorchan.benchmarks.comparative_qnei import make_comparative_qnei_strategy
from robotorchan.benchmarks.config import BenchmarkExperimentConfig
from robotorchan.benchmarks.initial_pairing import validate_initial_pairing
from robotorchan.benchmarks.observation_noise import SeededObservationNoise
from robotorchan.benchmarks.persistence import save_trajectory
from robotorchan.benchmarks.registry import BenchmarkProblemRegistry
from robotorchan.benchmarks.runner import BenchmarkTrajectory, random_candidates, run_benchmark
from robotorchan.benchmarks.standard_problems import register_standard_problems

STRATEGIES = ("random", "sobol", "qEI", "qNEI")
PROBLEMS = ("branin", "hartmann6")


def run_noisy_singleobjective(
    destination: str | Path,
    *,
    problem: str,
    q: int,
    git_sha: str,
    dirty: bool,
    num_restarts: int = 3,
    raw_samples: int = 64,
    noise_std: float = 0.1,
    noise_policy: str = "paired",
) -> dict[str, object]:
    """Persist full seed-paired runs; raise rather than silently omit failures.

    Run separately for each problem and q. A failed run leaves an error record
    and must not be reported as a completed comparison.
    """
    if problem not in PROBLEMS:
        raise ValueError("Unsupported noisy single-objective problem.")
    if q not in (1, 3):
        raise ValueError("q must be 1 or 3.")
    if not git_sha or type(dirty) is not bool:
        raise ValueError("Explicit git SHA and dirty state are required.")
    if type(num_restarts) is not int or num_restarts < 1:
        raise ValueError("num_restarts must be positive.")
    if type(raw_samples) is not int or raw_samples < 1:
        raise ValueError("raw_samples must be positive.")

    if not isinstance(noise_std, (int, float)) or not 0 < noise_std < float("inf"):
        raise ValueError("noise_std must be positive and finite.")
    if noise_policy not in ("paired", "independent"):
        raise ValueError("Unknown noise policy.")

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
        "observation_noise": {
            "distribution": "Gaussian",
            "std": noise_std,
            "policy": noise_policy,
        },
        "rng": "independent_streams_and_isolated_strategy_torch_rng",
        "status": "running",
    }
    manifest_path = root / "manifest.json"

    def persist_manifest() -> None:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    persist_manifest()
    trajectories: dict[str, tuple[BenchmarkTrajectory, ...]] = {}
    registry = BenchmarkProblemRegistry()
    register_standard_problems(registry)

    def gaussian_noise(truth: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        return (
            torch.randn(truth.shape, dtype=truth.dtype, device=truth.device, generator=generator)
            * noise_std
        )

    observation_noise = SeededObservationNoise(gaussian_noise, policy=noise_policy)
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
            ComparativeExperimentCell(tier="empirical_standard", config=config)
            if strategy == "random":
                candidate_generator = random_candidates
            elif strategy == "sobol":

                def candidate_generator(
                    problem_view, X, Y, batch_q, generator, *, initial_points=20
                ):
                    return sobol_candidates(
                        problem_view,
                        X,
                        Y,
                        batch_q,
                        generator,
                        initial_points=initial_points,
                    )
            elif strategy == "qEI":
                candidate_generator = make_comparative_qei_strategy(
                    num_restarts=num_restarts, raw_samples=raw_samples
                )
            else:
                candidate_generator = make_comparative_qnei_strategy(
                    num_restarts=num_restarts,
                    raw_samples=raw_samples,
                )
            runs = run_benchmark(
                config,
                candidate_generator,
                registry=registry,
                restrict_candidate_problem=True,
                independent_rng_streams=True,
                observation_noise=observation_noise,
                isolate_strategy_rng=True,
            )
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
        pairing = validate_initial_pairing(
            trajectories, require_observation_pairing=noise_policy == "paired"
        )
        manifest["pairing"] = {
            "design": pairing.design_paired,
            "truth": pairing.truth_paired,
            "observation": pairing.observation_paired,
        }
        if not any(
            not torch.equal(run.Y_observed, run.Y_truth)
            for runs in trajectories.values()
            for run in runs
        ):
            raise ValueError("Noisy cohort unexpectedly has no observation perturbation.")
        manifest["status"] = "runtime_validated"
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
        persist_manifest()
        raise
    persist_manifest()
    return manifest
