"""Tests for empirical benchmark environment provenance."""

from __future__ import annotations

import json

from robotorchan.benchmarks.config import BenchmarkExperimentConfig
from robotorchan.benchmarks.empirical_environment import (
    capture_empirical_environment,
    save_empirical_environment,
)


def test_environment_manifest_records_required_provenance() -> None:
    """Record versions, device, precision, seeds and resource settings."""
    config = BenchmarkExperimentConfig(
        problem="branin",
        strategy="random",
        seeds=(0, 1),
        initial_points=10,
        evaluation_budget=10,
        q=1,
    )
    manifest = capture_empirical_environment(config)

    assert manifest["schema_version"] == 1
    assert manifest["config"] == config.to_dict()
    assert manifest["python"]
    assert manifest["platform"]
    assert manifest["torch_threads"] >= 1
    assert set(manifest["versions"]) == {"robotorchan", "torch", "botorch", "gpytorch"}
    assert manifest["config"]["dtype"] == "float64"
    assert manifest["config"]["device"] == "cpu"


def test_environment_manifest_json_roundtrip(tmp_path) -> None:
    """Write an independently readable environment manifest."""
    config = BenchmarkExperimentConfig(problem="branin", strategy="random")
    destination = tmp_path / "results" / "environment.json"

    manifest = save_empirical_environment(destination, config)

    assert json.loads(destination.read_text(encoding="utf-8")) == manifest
