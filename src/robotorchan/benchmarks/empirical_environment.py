"""Capture reproducibility provenance for empirical benchmark runs."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from robotorchan.benchmarks.config import BenchmarkExperimentConfig


def _version(package: str) -> str | None:
    """Return an installed distribution version, or None when unavailable."""
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_sha() -> str | None:
    """Resolve the current repository commit without assuming Git is installed."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def capture_empirical_environment(config: BenchmarkExperimentConfig) -> dict[str, Any]:
    """Capture a JSON-compatible execution manifest without modifying RNG state."""
    cuda_available = torch.cuda.is_available()
    return {
        "schema_version": 1,
        "git_sha": _git_sha(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "cuda_available": cuda_available,
        "cuda_device": (
            torch.cuda.get_device_name(torch.cuda.current_device())
            if cuda_available and config.device == "cuda"
            else None
        ),
        "versions": {
            package: _version(package)
            for package in ("robotorchan", "torch", "botorch", "gpytorch")
        },
        "config": config.to_dict(),
    }


def save_empirical_environment(
    path: str | Path,
    config: BenchmarkExperimentConfig,
) -> dict[str, Any]:
    """Persist the environment manifest alongside benchmark result artifacts."""
    manifest = capture_empirical_environment(config)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
