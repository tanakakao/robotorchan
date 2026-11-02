"""Regression tests for notebook schema validation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import nbformat

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts/check_notebook_schema.py"
_SPEC = importlib.util.spec_from_file_location("check_notebook_schema", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
validate_notebooks = _MODULE.validate_notebooks


def test_valid_notebook_passes(tmp_path: Path) -> None:
    path = tmp_path / "valid.ipynb"
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("print(1)")])
    nbformat.write(notebook, path)

    assert validate_notebooks(tmp_path) == (1, [])


def test_missing_outputs_fails(tmp_path: Path) -> None:
    path = tmp_path / "missing_outputs.ipynb"
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("print(1)")])
    del notebook.cells[0]["outputs"]
    path.write_text(json.dumps(notebook), encoding="utf-8")

    count, failures = validate_notebooks(tmp_path)
    assert count == 1
    assert len(failures) == 1
    assert "missing_outputs.ipynb" in failures[0]
    assert "outputs" in failures[0]


def test_ignored_virtual_environment_not_counted(tmp_path: Path) -> None:
    hidden = tmp_path / ".venv"
    hidden.mkdir()
    (hidden / "broken.ipynb").write_text("{invalid", encoding="utf-8")

    assert validate_notebooks(tmp_path) == (0, [])


def test_malformed_json_fails(tmp_path: Path) -> None:
    (tmp_path / "broken.ipynb").write_text("{invalid", encoding="utf-8")

    count, failures = validate_notebooks(tmp_path)
    assert count == 1
    assert len(failures) == 1
