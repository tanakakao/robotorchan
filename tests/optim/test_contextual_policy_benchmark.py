"""Lightweight smoke checks for the synthetic contextual policy benchmark."""

import importlib.util
from pathlib import Path

import pytest


def test_contextual_policy_benchmark_is_deterministic() -> None:
    path = Path(__file__).resolve().parents[2] / "benchmarks" / "contextual_policy.py"
    spec = importlib.util.spec_from_file_location("contextual_policy_benchmark", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    first = module.run(seed=17, n_samples=1024)
    second = module.run(seed=17, n_samples=1024)
    for key in ("fixed_expected", "oracle_expected", "oracle_gain", "fixed_mc", "oracle_mc"):
        assert first[key] == second[key]
    assert first["oracle_expected"] == pytest.approx(1.0)
    assert first["oracle_gain"] > 0.0
    assert first["oracle_mc"] >= first["fixed_mc"]
    assert first["evaluation_seconds"] >= 0.0
