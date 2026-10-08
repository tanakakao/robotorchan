"""Validated, serializable configuration for benchmark experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch

from robotorchan.benchmarks.problem import BenchmarkProblem
from robotorchan.benchmarks.registry import BenchmarkProblemRegistry, default_problem_registry

DeviceKind = Literal["cpu", "cuda"]
DTypeName = Literal["float32", "float64"]


@dataclass(frozen=True)
class BenchmarkExperimentConfig:
    """Immutable experiment settings independent of models and acquisitions.

    The evaluation budget counts new observations, excluding initial points.
    A batch may be smaller than q at the end of an experiment.
    """

    problem: str
    strategy: str
    seeds: tuple[int, ...] = (0,)
    initial_points: int = 8
    evaluation_budget: int = 40
    q: int = 1
    dtype: DTypeName = "float64"
    device: DeviceKind = "cpu"

    def __post_init__(self) -> None:
        if not isinstance(self.problem, str) or not self.problem.strip():
            raise ValueError("problem must be a nonempty string.")
        if not isinstance(self.strategy, str) or not self.strategy.strip():
            raise ValueError("strategy must be a nonempty string.")
        if not isinstance(self.seeds, tuple) or not self.seeds:
            raise ValueError("seeds must be a nonempty tuple.")
        if any(type(seed) is not int or seed < 0 for seed in self.seeds):
            raise ValueError("seeds must contain nonnegative integers.")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique.")
        for name in ("initial_points", "evaluation_budget", "q"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.dtype not in ("float32", "float64"):
            raise ValueError("dtype must be float32 or float64.")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda.")

    @property
    def torch_dtype(self) -> torch.dtype:
        """Return the configured torch floating-point dtype."""
        return torch.float32 if self.dtype == "float32" else torch.float64

    @property
    def torch_device(self) -> torch.device:
        """Return the configured torch device without allocating a tensor."""
        return torch.device(self.device)

    def resolve_problem(
        self,
        registry: BenchmarkProblemRegistry | None = None,
    ) -> BenchmarkProblem:
        """Resolve and validate a named benchmark against the selected device."""
        source = default_problem_registry if registry is None else registry
        problem = source.create(self.problem)
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is requested but unavailable.")
        return problem

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible configuration record."""
        data = asdict(self)
        data["seeds"] = list(self.seeds)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkExperimentConfig:
        """Load a record and reject unknown keys rather than silently ignoring them."""
        if not isinstance(data, dict):
            raise TypeError("Configuration must be a dictionary.")
        allowed = set(cls.__dataclass_fields__)
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        payload = dict(data)
        if "seeds" in payload:
            if not isinstance(payload["seeds"], (tuple, list)):
                raise ValueError("seeds must be a sequence.")
            payload["seeds"] = tuple(payload["seeds"])
        return cls(**payload)
