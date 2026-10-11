"""Versioned JSON persistence for reproducible empirical comparison reports."""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import botorch
import torch

from robotorchan.benchmarks.comparison_report import EmpiricalComparisonReport

REPORT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ReportArtifact:
    """Validated, versioned report payload and provenance metadata."""

    schema_version: int
    provenance: dict[str, str]
    report: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return the validated persistence envelope."""
        return {
            "schema_version": self.schema_version,
            "provenance": self.provenance,
            "report": self.report,
        }


def create_report_artifact(
    report: EmpiricalComparisonReport,
    *,
    revision: str = "",
) -> ReportArtifact:
    """Capture available runtime versions and optional caller-supplied revision."""
    return ReportArtifact(
        schema_version=REPORT_SCHEMA_VERSION,
        provenance={
            "python": platform.python_version(),
            "torch": torch.__version__,
            "botorch": botorch.__version__,
            "revision": revision,
        },
        report=report.to_dict(),
    )


def save_report_artifact(artifact: ReportArtifact, path: str | Path) -> None:
    """Write a strict JSON artifact; reject unsupported schema versions."""
    if artifact.schema_version != REPORT_SCHEMA_VERSION:
        raise ValueError("Unsupported report schema version.")
    destination = Path(path)
    content = json.dumps(artifact.to_dict(), indent=2, sort_keys=True, allow_nan=False)
    destination.write_text(content + "\n", encoding="utf-8")


def load_report_artifact(path: str | Path) -> ReportArtifact:
    """Read a report envelope and validate required top-level fields."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data) != {
        "schema_version",
        "provenance",
        "report",
    }:
        raise ValueError("Invalid report artifact envelope.")
    if type(data["schema_version"]) is not int or data["schema_version"] != REPORT_SCHEMA_VERSION:
        raise ValueError("Unsupported report schema version.")
    provenance = data["provenance"]
    report = data["report"]
    if not isinstance(provenance, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in provenance.items()
    ):
        raise ValueError("Invalid provenance metadata.")
    if not isinstance(report, dict) or not {
        "metric",
        "reference",
        "fairness",
        "strategies",
        "comparisons",
        "inference",
    }.issubset(report):
        raise ValueError("Invalid comparison report payload.")
    return ReportArtifact(
        schema_version=REPORT_SCHEMA_VERSION,
        provenance=provenance,
        report=report,
    )
