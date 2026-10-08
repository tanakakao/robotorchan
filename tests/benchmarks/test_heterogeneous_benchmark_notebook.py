"""Validate the reproducible Phase 19 notebook's static execution contract."""

from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK = (
    Path(__file__).resolve().parents[2]
    / "notebooks"
    / "heterogeneous_bo_benchmark_phase19.ipynb"
)


def test_benchmark_notebook_has_valid_cells_and_clean_outputs() -> None:
    """Notebook stays readable, executable in order, and free of stored outputs."""
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["kernelspec"]["language"] == "python"
    cells = notebook["cells"]
    assert cells
    assert cells[0]["cell_type"] == "markdown"
    code_cells = [cell for cell in cells if cell["cell_type"] == "code"]
    assert len(code_cells) >= 3
    assert all(cell["execution_count"] is None for cell in code_cells)
    assert all(cell["outputs"] == [] for cell in code_cells)
    for cell in cells:
        assert isinstance(cell["source"], list)
        assert all(isinstance(line, str) for line in cell["source"])
    source = "\n".join("".join(cell["source"]) for cell in code_cells)
    assert "compare_strategies(" in source
    assert "summarize_trajectories(" in source
    assert "plt.show()" in source
    assert "difference = curves[" in source
    for cell in code_cells:
        compile("".join(cell["source"]), str(NOTEBOOK), "exec")
