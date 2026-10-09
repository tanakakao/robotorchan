"""Resumable seeded benchmark execution with explicit RNG checkpoints."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from robotorchan.benchmarks.config import BenchmarkExperimentConfig
from robotorchan.benchmarks.registry import BenchmarkProblemRegistry
from robotorchan.benchmarks.runner import (
    BenchmarkTrajectory,
    CandidateGenerator,
    sobol_initial_design,
)
from robotorchan.benchmarks.storage import _decode_tensor, _encode_tensor

_CHECKPOINT_VERSION = 1
_FIELDS = ("X", "Y_observed", "Y_truth", "constraints", "costs")


@dataclass(frozen=True)
class BenchmarkCheckpoint:
    """A complete trajectory prefix and candidate-generator RNG state."""

    config: BenchmarkExperimentConfig
    trajectory: BenchmarkTrajectory
    generator_state: Tensor


def save_benchmark_checkpoint(path: str | Path, checkpoint: BenchmarkCheckpoint) -> None:
    """Write a validated, versioned checkpoint without Python object deserialization."""
    config, run = checkpoint.config, checkpoint.trajectory
    if len(config.seeds) != 1 or run.seed != config.seeds[0]:
        raise ValueError("Checkpoint requires one matching configured seed.")
    count = run.X.shape[0]
    if not config.initial_points <= count <= config.initial_points + config.evaluation_budget:
        raise ValueError("Checkpoint evaluation count exceeds configured limits.")
    if run.initial_points != config.initial_points or run.X.ndim != 2:
        raise ValueError("Invalid checkpoint initial design.")
    for name in _FIELDS:
        tensor = getattr(run, name)
        if (
            not isinstance(tensor, Tensor)
            or tensor.ndim != 2
            or tensor.shape[0] != count
            or tensor.dtype != config.torch_dtype
        ):
            raise ValueError(f"Invalid checkpoint tensor: {name}.")
    state = checkpoint.generator_state
    if not isinstance(state, Tensor) or state.dtype != torch.uint8 or state.ndim != 1:
        raise ValueError("Generator state must be a one-dimensional uint8 tensor.")
    payload = {
        "schema_version": _CHECKPOINT_VERSION,
        "config": config.to_dict(),
        "seed": run.seed,
        "initial_points": run.initial_points,
        "generator_state": state.cpu().tolist(),
        "trajectory": {name: _encode_tensor(getattr(run, name)) for name in _FIELDS},
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_benchmark_checkpoint(path: str | Path) -> BenchmarkCheckpoint:
    """Load and validate a checkpoint, rejecting unknown schema versions."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "config",
        "seed",
        "initial_points",
        "generator_state",
        "trajectory",
    }:
        raise ValueError("Invalid checkpoint schema.")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("Unsupported checkpoint version.")
    config = BenchmarkExperimentConfig.from_dict(payload["config"])
    if type(payload["seed"]) is not int or type(payload["initial_points"]) is not int:
        raise ValueError("Invalid checkpoint seed or initial count.")
    record = payload["trajectory"]
    if not isinstance(record, dict) or set(record) != set(_FIELDS):
        raise ValueError("Invalid checkpoint trajectory record.")
    state_values: Any = payload["generator_state"]
    if (
        not isinstance(state_values, list)
        or not state_values
        or any(type(value) is not int or not 0 <= value <= 255 for value in state_values)
    ):
        raise ValueError("Invalid generator state.")
    run = BenchmarkTrajectory(
        seed=payload["seed"],
        initial_points=payload["initial_points"],
        **{name: _decode_tensor(record[name]) for name in _FIELDS},
    )
    checkpoint = BenchmarkCheckpoint(
        config=config,
        trajectory=run,
        generator_state=torch.tensor(state_values, dtype=torch.uint8),
    )
    # Reuse save-time validation without writing a file.
    if len(config.seeds) != 1 or run.seed != config.seeds[0]:
        raise ValueError("Checkpoint seed mismatch.")
    count = run.X.shape[0]
    if (
        run.initial_points != config.initial_points
        or run.X.ndim != 2
        or not config.initial_points <= count <= config.initial_points + config.evaluation_budget
    ):
        raise ValueError("Invalid checkpoint evaluation count.")
    for name in _FIELDS:
        tensor = getattr(run, name)
        if tensor.ndim != 2 or tensor.shape[0] != count or tensor.dtype != config.torch_dtype:
            raise ValueError(f"Invalid checkpoint tensor: {name}.")
    return checkpoint


def run_resumable_benchmark(
    config: BenchmarkExperimentConfig,
    candidate_generator: CandidateGenerator,
    *,
    registry: BenchmarkProblemRegistry | None = None,
    checkpoint: BenchmarkCheckpoint | None = None,
    max_batches: int | None = None,
) -> tuple[BenchmarkTrajectory, BenchmarkCheckpoint]:
    """Run one seed, optionally stopping after a whole number of new batches.

    Resumption is deterministic for candidate generators that use only the
    supplied generator and trajectory history; external model state is not saved.
    """
    if len(config.seeds) != 1:
        raise ValueError("Resumable execution requires exactly one seed.")
    if max_batches is not None and (type(max_batches) is not int or max_batches < 0):
        raise ValueError("max_batches must be a nonnegative integer.")
    problem = config.resolve_problem(registry)
    seed = config.seeds[0]
    generator = torch.Generator(device=config.torch_device).manual_seed(seed)
    if checkpoint is None:
        X = sobol_initial_design(
            problem,
            config.initial_points,
            seed,
            dtype=config.torch_dtype,
            device=config.torch_device,
        )
        observed = problem.evaluate_observation(X)
        truth = problem.evaluate_truth(X)
        constraints = problem.evaluate_constraints(X)
        costs = problem.evaluate_cost(X)
    else:
        if checkpoint.config != config:
            raise ValueError("Checkpoint configuration mismatch.")
        run = checkpoint.trajectory
        if run.seed != seed or run.initial_points != config.initial_points:
            raise ValueError("Checkpoint seed or initial points mismatch.")
        if run.X.ndim != 2 or run.X.shape[1] != problem.dimension:
            raise ValueError("Checkpoint problem dimension mismatch.")
        if (
            not config.initial_points
            <= run.X.shape[0]
            <= (config.initial_points + config.evaluation_budget)
        ):
            raise ValueError("Checkpoint evaluation count mismatch.")
        for name in _FIELDS:
            value = getattr(run, name)
            if (
                value.ndim != 2
                or value.shape[0] != run.X.shape[0]
                or value.dtype != config.torch_dtype
                or value.device != config.torch_device
            ):
                raise ValueError(f"Checkpoint tensor mismatch: {name}.")
        problem._validate_X(run.X)
        generator.set_state(checkpoint.generator_state.cpu())
        X, observed, truth = run.X.clone(), run.Y_observed.clone(), run.Y_truth.clone()
        constraints, costs = run.constraints.clone(), run.costs.clone()
    remaining = config.initial_points + config.evaluation_budget - X.shape[0]
    batches = 0
    while remaining and (max_batches is None or batches < max_batches):
        q = min(config.q, remaining)
        candidates = candidate_generator(problem, X.clone(), observed.clone(), q, generator)
        if not isinstance(candidates, Tensor) or candidates.shape != (q, problem.dimension):
            raise ValueError("Candidate generator must return a tensor of shape (q, d).")
        problem._validate_X(candidates)
        X = torch.cat((X, candidates), dim=0)
        observed = torch.cat((observed, problem.evaluate_observation(candidates)), dim=0)
        truth = torch.cat((truth, problem.evaluate_truth(candidates)), dim=0)
        constraints = torch.cat((constraints, problem.evaluate_constraints(candidates)), dim=0)
        costs = torch.cat((costs, problem.evaluate_cost(candidates)), dim=0)
        remaining -= q
        batches += 1
    trajectory = BenchmarkTrajectory(
        seed=seed,
        X=X,
        Y_observed=observed,
        Y_truth=truth,
        constraints=constraints,
        costs=costs,
        initial_points=config.initial_points,
    )
    result = BenchmarkCheckpoint(
        config=config,
        trajectory=trajectory,
        generator_state=generator.get_state().cpu(),
    )
    return trajectory, result
