"""Tests for versioned empirical report artifacts."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from robotorchan.benchmarks.report_persistence import (
    REPORT_SCHEMA_VERSION,
    ReportArtifact,
    load_report_artifact,
    save_report_artifact,
)


def _artifact() -> ReportArtifact:
    return ReportArtifact(
        schema_version=REPORT_SCHEMA_VERSION,
        provenance={"python": "3.12.0", "revision": "abc123"},
        report={
            "metric": "simple_regret",
            "reference": "random",
            "fairness": {},
            "strategies": {},
            "comparisons": [],
            "inference": [],
        },
    )


def test_report_round_trip(tmp_path) -> None:
    path = tmp_path / "comparison.json"
    original = _artifact()
    save_report_artifact(original, path)
    loaded = load_report_artifact(path)
    assert loaded == original
    assert json.loads(path.read_text(encoding="utf-8")) == original.to_dict()


def test_unknown_schema_is_rejected(tmp_path) -> None:
    path = tmp_path / "comparison.json"
    with pytest.raises(ValueError, match="schema version"):
        save_report_artifact(replace(_artifact(), schema_version=999), path)
    save_report_artifact(_artifact(), path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["schema_version"] = 999
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="schema version"):
        load_report_artifact(path)


def test_nonfinite_values_are_rejected(tmp_path) -> None:
    artifact = _artifact()
    artifact.report["strategies"]["random"] = {"mean": [float("inf")]}
    with pytest.raises(ValueError):
        save_report_artifact(artifact, tmp_path / "comparison.json")


def test_invalid_envelope_is_rejected(tmp_path) -> None:
    path = tmp_path / "comparison.json"
    path.write_text('{"schema_version": 1}', encoding="utf-8")
    with pytest.raises(ValueError, match="envelope"):
        load_report_artifact(path)
