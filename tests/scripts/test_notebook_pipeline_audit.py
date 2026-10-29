"""Regression tests for static notebook pipeline evidence."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts/audit_notebook_pipeline.py"
_SPEC = importlib.util.spec_from_file_location("audit_notebook_pipeline", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
audit = _MODULE.audit
calls_in = _MODULE.calls_in


def test_calls_in_recognizes_functions_and_methods() -> None:
    source = "model.fit()\nposterior = model.posterior(X)\noptimize_acqf(acq, bounds)"
    assert {"fit", "posterior", "optimize_acqf"} <= calls_in(source)


def test_audit_reports_evidence_without_claiming_execution(tmp_path: Path) -> None:
    notebook = {
        "cells": [
            {"cell_type": "markdown", "source": ["model.fit() is described here"]},
            {
                "cell_type": "code",
                "source": [
                    "model.fit()\n",
                    "posterior = model.posterior(X)\n",
                    "candidate, value = optimize_acqf(acq, bounds)\n",
                ],
            },
            {"cell_type": "code", "source": ["def broken(:\n"]},
        ]
    }
    path = tmp_path / "example.ipynb"
    path.write_text(json.dumps(notebook), encoding="utf-8")
    row = audit(path)
    assert row["parseable_code_cells"] == "1"
    assert row["invalid_code_cells"] == "1"
    assert row["model_fit_calls"] == "true"
    assert row["native_fit_calls"] == "false"
    assert row["posterior_calls"] == "true"
    assert row["optimizer_calls"] == "true"
    assert row["candidate_mentions"] == "true"
    assert row["execution"] == "unverified"


def test_markdown_only_is_not_code_evidence(tmp_path: Path) -> None:
    path = tmp_path / "markdown.ipynb"
    path.write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "markdown",
                        "source": ["model.fit(); optimize_acqf(); candidate"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    row = audit(path)
    assert row["model_fit_calls"] == "false"
    assert row["optimizer_calls"] == "false"
    assert row["candidate_mentions"] == "false"


def test_real_single_task_notebook_has_candidate_pipeline() -> None:
    notebook = Path(__file__).resolve().parents[2] / "examples/notebooks/01_single_task_gp.ipynb"
    row = audit(notebook)
    assert int(row["parseable_code_cells"]) > 0
    assert row["invalid_code_cells"] == "0"
    assert row["native_fit_calls"] == "true"
    assert row["posterior_calls"] == "true"
    assert row["optimizer_calls"] == "true"
    assert row["candidate_mentions"] == "true"
    assert row["execution"] == "unverified"
